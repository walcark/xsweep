"""Batched delivery: fewer calls, same values, dedup and resume intact.

The point of `batch` is that it changes only how a point reaches the
callable. Everything a `loop` variable buys — the sweep space, dedup, the
store, resumption — has to keep working untouched, and the result has to
be indistinguishable from the scalar-delivered run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xsweep import SweepPolicy, sweep
from xsweep.errors import ContractError


@pytest.fixture
def states() -> xr.Dataset:
    """Return 24 distinct optical thicknesses."""
    return xr.Dataset({"aot": ("aot", np.linspace(0.05, 0.9, 24))})


@pytest.fixture
def scene() -> xr.Dataset:
    """Return a 6x6 map holding only three distinct values."""
    values = np.repeat(np.repeat([0.1, 0.3, 0.5], 2), 6).reshape(6, 6)
    return xr.Dataset({"aot": (("y", "x"), values)})


# --- Delivery ---


def test_a_group_arrives_as_one_array(states: xr.Dataset) -> None:
    """Eight points reach the callable as one 1-D array, not eight calls."""
    widths: list[int] = []

    @sweep("batch(aot) -> rho()")
    def f(aot: xr.DataArray) -> xr.DataArray:
        widths.append(aot.sizes["point"])
        return xr.DataArray(np.exp(-aot.values), dims=["point"])

    f(states, policy=SweepPolicy(batch_size=8))

    assert widths == [8, 8, 8]


def test_the_result_matches_the_scalar_run(states: xr.Dataset) -> None:
    """Batching is a cost decision: it moves no value and no dim."""

    @sweep("batch(aot) -> rho()")
    def batched(aot: xr.DataArray) -> xr.DataArray:
        return xr.DataArray(np.exp(-aot.values), dims=["point"])

    @sweep("loop(aot) -> rho()")
    def scalar(aot: float) -> float:
        return float(np.exp(-aot))

    xr.testing.assert_allclose(
        batched(states, policy=SweepPolicy(batch_size=8)), scalar(states)
    )


def test_a_partial_group_is_handed_only_what_is_left(states: xr.Dataset) -> None:
    """24 points at 10 per group leaves a final group of 4."""
    widths: list[int] = []

    @sweep("batch(aot) -> rho()")
    def f(aot: xr.DataArray) -> xr.DataArray:
        widths.append(aot.sizes["point"])
        return xr.DataArray(np.exp(-aot.values), dims=["point"])

    f(states, policy=SweepPolicy(batch_size=10))

    assert widths == [10, 10, 4]


def test_scalar_and_batched_variables_mix(states: xr.Dataset) -> None:
    """A loop variable shared by the group still arrives as a scalar."""
    seen: list[tuple[object, int]] = []
    space = states.assign(rh=50.0)

    @sweep("loop(rh) batch(aot) -> rho()")
    def f(rh: float, aot: xr.DataArray) -> xr.DataArray:
        seen.append((rh, aot.sizes["point"]))
        return xr.DataArray(np.exp(-aot.values) + rh, dims=["point"])

    f(space, policy=SweepPolicy(batch_size=12))

    assert seen == [(50.0, 12), (50.0, 12)]


# --- What the sweep space still buys ---


def test_dedup_still_collapses_repeated_points(scene: xr.Dataset) -> None:
    """36 pixels holding 3 distinct values cost one call of 3."""
    widths: list[int] = []

    @sweep("batch(aot) -> rho()")
    def f(aot: xr.DataArray) -> xr.DataArray:
        widths.append(aot.sizes["point"])
        return xr.DataArray(np.exp(-aot.values), dims=["point"])

    out = f(scene, policy=SweepPolicy(batch_size=8, dedup=True))

    assert widths == [3]
    assert dict(out.sizes) == {"y": 6, "x": 6}
    np.testing.assert_allclose(out["rho"].values, np.exp(-scene["aot"].values))


def test_a_finished_store_makes_no_call(states: xr.Dataset, tmp_path: Path) -> None:
    """The second run of a completed sweep reads rather than computes."""
    calls: list[int] = []

    @sweep("batch(aot) -> rho()")
    def f(aot: xr.DataArray) -> xr.DataArray:
        calls.append(aot.sizes["point"])
        return xr.DataArray(np.exp(-aot.values), dims=["point"])

    policy = SweepPolicy(batch_size=8, store=str(tmp_path / "s.zarr"))
    f(states, policy=policy)
    calls.clear()
    f(states, policy=policy)

    assert calls == []


def test_a_crash_costs_one_group_not_the_whole_run(
    states: xr.Dataset, tmp_path: Path
) -> None:
    """Resuming recomputes the interrupted group only.

    This is the property `vec` cannot offer: it holds the whole sweep as
    one job, so any interruption discards everything computed so far.
    """
    calls: list[int] = []
    alive = [2]

    @sweep("batch(aot) -> rho()")
    def f(aot: xr.DataArray) -> xr.DataArray:
        calls.append(aot.sizes["point"])
        if len(calls) > alive[0]:
            raise RuntimeError("the cluster went away")
        return xr.DataArray(np.exp(-aot.values), dims=["point"])

    policy = SweepPolicy(batch_size=8, store=str(tmp_path / "s.zarr"))
    f(states, policy=policy)

    alive[0] = 10_000
    calls.clear()
    f(states, policy=policy)

    assert sum(calls) == 8


# --- What the callable owes in return ---


def test_a_return_without_the_group_dim_is_refused(states: xr.Dataset) -> None:
    """Returning one value for a group of many must not pass silently."""

    @sweep("batch(aot) -> rho()")
    def f(aot: xr.DataArray) -> float:
        return float(np.exp(-aot.values[0]))

    with pytest.raises(ContractError, match="without a 'point' dim"):
        f(states, policy=SweepPolicy(batch_size=8, on_error="raise"))


def test_a_return_of_the_wrong_width_is_refused(states: xr.Dataset) -> None:
    """A group of 8 answered with 3 values is a bug, not a shape to guess."""

    @sweep("batch(aot) -> rho()")
    def f(aot: xr.DataArray) -> xr.DataArray:
        return xr.DataArray(np.exp(-aot.values[:3]), dims=["point"])

    with pytest.raises(ContractError, match="handed 8 point"):
        f(states, policy=SweepPolicy(batch_size=8, on_error="raise"))
