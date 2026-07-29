"""The sacred property, and the release gate that enforces it.

Constitution principle III: a policy may change the cost of a sweep, never
its values or its shape. Users tune policy for performance, so a policy that
altered science output would be silent data corruption.

The suite is parametrised rather than enumerated, so adding a policy field
means adding a parameter, not writing a new test.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsweep import SweepPolicy

from ._callees import scalar_engine, vector_engine


@pytest.fixture
def space() -> xr.Dataset:
    """Return a map with deliberate duplicates, so dedup has work to do."""
    aot = np.array([[0.1, 0.2, 0.1], [0.3, 0.2, 0.1], [0.1, 0.4, 0.2]])
    rh = np.array([[30.0, 70.0, 30.0], [50.0, 70.0, 30.0], [30.0, 10.0, 70.0]])
    return xr.Dataset({"aot": (("y", "x"), aot), "rh": (("y", "x"), rh)})


def _policies(tmp_path) -> dict[str, SweepPolicy]:
    """Return every policy the result must be invariant under."""
    return {
        "baseline": SweepPolicy(),
        "dedup": SweepPolicy(dedup=True),
        "dedup_partial": SweepPolicy(dedup=("y", "x")),
        "store": SweepPolicy(store=str(tmp_path / "store.zarr")),
        "store_dedup": SweepPolicy(store=str(tmp_path / "sd.zarr"), dedup=True),
        "store_loaded": SweepPolicy(store=str(tmp_path / "sl.zarr"), load=True),
        "process": SweepPolicy(executor="process", max_workers=2),
        "process_dedup": SweepPolicy(executor="process", max_workers=2, dedup=True),
        "retries": SweepPolicy(retries=2),
    }


@pytest.mark.parametrize(
    "name",
    [
        "dedup",
        "dedup_partial",
        "store",
        "store_dedup",
        "store_loaded",
        "process",
        "process_dedup",
        "retries",
    ],
)
def test_result_is_bit_identical_under_any_policy(
    name: str, space: xr.Dataset, tmp_path
) -> None:
    """Values, dims and coordinates must not move when the policy does."""
    policies = _policies(tmp_path)
    reference = scalar_engine(space, policy=policies["baseline"])
    candidate = scalar_engine(space, policy=policies[name])

    assert candidate.rho.dims == reference.rho.dims
    assert candidate.rho.shape == reference.rho.shape
    assert np.array_equal(candidate.rho.values, reference.rho.values)
    assert np.array_equal(candidate.status.values, reference.status.values)


@pytest.mark.parametrize("batch", [1, 2, 3, 5, 7, 20])
def test_batch_size_never_changes_the_result(batch: int) -> None:
    """Batching a vec axis is safe by definition, so it must be provably so."""
    space = xr.Dataset({"aot": ("aot", [0.1, 0.2]), "wl": ("wl", np.arange(20.0))})
    reference = vector_engine(space)
    candidate = vector_engine(space, policy=SweepPolicy(chunks={"wl": batch}))

    assert candidate.t.dims == reference.t.dims
    assert np.array_equal(candidate.t.values, reference.t.values)


def test_dedup_reduces_calls_while_preserving_everything_else(
    space: xr.Dataset,
) -> None:
    """The optimisation must be visible in cost, invisible in output."""
    plain = scalar_engine.explain(space)
    deduped = scalar_engine.explain(space, policy=SweepPolicy(dedup=True))

    assert deduped.n_calls < plain.n_calls
    assert deduped.n_points == plain.n_points
    assert [v.dims for v in deduped.result] == [v.dims for v in plain.result]
    assert [v.sizes for v in deduped.result] == [v.sizes for v in plain.result]


def test_result_dims_are_the_union_of_loop_variable_dims(
    space: xr.Dataset,
) -> None:
    """FR-002 stated as a test: the shape comes from the space, not the policy."""
    for policy in (SweepPolicy(), SweepPolicy(dedup=True)):
        result = scalar_engine(space, policy=policy)
        assert result.rho.dims == ("y", "x")
        assert result.rho.shape == (3, 3)
