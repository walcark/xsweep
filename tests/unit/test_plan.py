"""Planning tests.

The central guarantee is that planning never calls the wrapped function: the
whole point of inspecting a plan is to decide whether to spend engine hours,
so spending one to build the report would defeat it.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsweep import Contract, OutVar, Plan, Sweeper, SweepPolicy, VecVar, sweep
from xsweep.errors import ContractError, PolicyError


@sweep("loop(a, b) -> out()")
def _product(a: float, b: float) -> float:
    """Return the product; the counter lives on the wrapper."""
    _product_calls.append((a, b))
    return a * b


_product_calls: list[tuple[float, float]] = []


def test_planning_never_calls_the_function(cartesian_space: xr.Dataset) -> None:
    """A plan costs nothing, which is why it is worth consulting."""
    _product_calls.clear()
    plan = _product.explain(cartesian_space)
    assert isinstance(plan, Plan)
    assert _product_calls == []
    assert plan.n_calls == 12


def test_plan_reports_axes_and_their_origin(map_space: xr.Dataset) -> None:
    """Zip and product must be distinguishable in the report."""

    @sweep("loop(aot, rh) -> out()")
    def f(aot: float, rh: float) -> float:
        return aot

    plan = f.explain(map_space)
    assert plan.loop_dims == ("y", "x")
    assert all(axis.is_zipped for axis in plan.axes)
    assert plan.n_points == 6


def test_one_work_item_per_point_without_batching(
    cartesian_space: xr.Dataset,
) -> None:
    """With nothing batched, work items and loop points coincide."""
    plan = _product.explain(cartesian_space)
    assert len(plan.work_items) == plan.n_points
    assert all(item.slices == {} for item in plan.work_items)


def test_batching_multiplies_work_items() -> None:
    """A batched dim splits each point into several calls."""

    @sweep("loop(a) vec(wl @ 8) -> t(wl)")
    def f(a: float, wl: xr.DataArray) -> xr.DataArray:
        return wl * a

    space = xr.Dataset({"a": ("a", [1.0, 2.0]), "wl": ("wl", np.arange(20.0))})
    plan = f.explain(space)
    widths = [sl.stop - sl.start for sl in plan.batches["wl"]]
    assert widths == [8, 8, 4]
    assert len(plan.work_items) == 2 * 3


def test_unknown_chunk_dim_fails_before_any_call() -> None:
    """Policy validation runs during planning, so it costs no engine time."""

    @sweep("loop(a) -> out()")
    def f(a: float) -> float:
        return a

    space = xr.Dataset({"a": ("a", [1.0])})
    with pytest.raises(PolicyError, match="available dims: a"):
        f.explain(space, policy=SweepPolicy(chunks={"z": 2}))


def test_chunking_a_loop_dim_is_rejected() -> None:
    """Only vec dims are batchable; a loop dim is a different concept."""

    @sweep("loop(a) -> out()")
    def f(a: float) -> float:
        return a

    space = xr.Dataset({"a": ("a", [1.0, 2.0])})
    with pytest.raises(PolicyError, match="only vec dims can be batched"):
        f.explain(space, policy=SweepPolicy(chunks={"a": 1}))


def test_batch_marker_on_a_multi_dim_vec_is_rejected() -> None:
    """'@ N' cannot say which dim it means once the variable has several.

    This fires at decoration, before any space exists, so the message points
    at the policy form without being able to confirm the variable's rank.
    """
    with pytest.raises(ContractError, match="chunks="):

        @sweep("vec(A @ 500) -> C(y, x)")
        def f(A: xr.DataArray) -> xr.DataArray:
            return A


def test_batch_marker_on_a_multi_dim_vec_is_rejected_with_a_space() -> None:
    """Built programmatically, the same error is caught at planning.

    There the space is known, so the message can name the actual dims.
    """
    contract = Contract(
        vec=(VecVar("A", 500),), out=(OutVar("C", ("y", "x"), (None, None)),)
    )
    sweeper = Sweeper(contract, lambda A: A)
    space = xr.Dataset({"A": (("y", "x"), np.zeros((4, 4)))})
    with pytest.raises(ContractError, match="chunks=\\{'y': 500\\}"):
        sweeper.explain(space)


def test_undetermined_output_size_is_reported_not_probed() -> None:
    """An unknown size is announced, never discovered by calling."""
    calls: list[float] = []

    @sweep("loop(a) -> band_int(band)")
    def f(a: float) -> xr.DataArray:
        calls.append(a)
        return xr.DataArray([a, a], dims="band")

    space = xr.Dataset({"a": ("a", [1.0, 2.0])})
    plan = f.explain(space)
    assert not plan.determined
    assert calls == []
    assert "undetermined" in repr(plan)


def test_report_mentions_the_costly_misconfigurations(
    map_space: xr.Dataset,
) -> None:
    """The report exists to make expensive mistakes visible before launching."""

    @sweep("loop(aot, rh) -> out()")
    def f(aot: float, rh: float) -> float:
        return aot

    text = repr(f.explain(map_space))
    assert "dedup        disabled" in text
    assert "points       6" in text
