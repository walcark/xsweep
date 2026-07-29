"""Quickstart scenario 2: failures never kill a sweep, and runs resume."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsweep import PointFailed, SweepPolicy, sweep


def test_a_failing_point_is_recorded_and_the_sweep_continues(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """One bad point must not cost the other eleven."""

    @sweep("loop(a, b) -> out()")
    def f(a: float, b: float) -> float:
        if a == 0.2:
            raise RuntimeError("engine blew up")
        return a * b

    result = f(cartesian_space, policy=SweepPolicy(store=str(tmp_path / "s.zarr")))
    failed = result.status.values == 2
    assert failed.sum() == 4
    assert np.isnan(result.out.values[failed]).all()
    assert not np.isnan(result.out.values[~failed]).any()


def test_fail_fast_surfaces_the_original_error(
    cartesian_space: xr.Dataset,
) -> None:
    """When the user asks to stop, they must learn why."""

    @sweep("loop(a, b) -> out()")
    def f(a: float, b: float) -> float:
        raise RuntimeError("engine blew up")

    with pytest.raises(PointFailed, match="engine blew up"):
        f(cartesian_space, policy=SweepPolicy(on_error="raise"))


def test_retries_give_a_flaky_call_another_chance(
    cartesian_space: xr.Dataset,
) -> None:
    """Engines fail transiently; a retry is cheaper than a lost sweep."""
    attempts: dict[tuple[float, float], int] = {}

    @sweep("loop(a, b) -> out()")
    def f(a: float, b: float) -> float:
        key = (a, b)
        attempts[key] = attempts.get(key, 0) + 1
        if attempts[key] < 2:
            raise RuntimeError("transient")
        return a * b

    result = f(cartesian_space, policy=SweepPolicy(retries=1))
    assert (result.status.values == 1).all()
    assert set(attempts.values()) == {2}


def test_only_missing_points_are_recomputed_on_resume(
    tmp_path, cartesian_space: xr.Dataset
) -> None:
    """The whole reason the store exists: never pay twice for a point."""
    fail = {"active": True}
    calls: list[tuple[float, float]] = []

    @sweep("loop(a, b) -> out()", version="1")
    def f(a: float, b: float) -> float:
        calls.append((a, b))
        if fail["active"] and a == 0.2:
            raise RuntimeError("engine blew up")
        return a * b

    store = str(tmp_path / "s.zarr")
    f(cartesian_space, policy=SweepPolicy(store=store))
    assert len(calls) == 12

    fail["active"] = False
    calls.clear()
    result = f(cartesian_space, policy=SweepPolicy(store=store))
    assert len(calls) == 4
    assert (result.status.values == 1).all()


def test_skip_predicate_never_calls_the_function(
    cartesian_space: xr.Dataset,
) -> None:
    """Skipping is explicit, and it is decided during planning."""
    calls: list[float] = []

    @sweep("loop(a, b) -> out()")
    def f(a: float, b: float) -> float:
        calls.append(a)
        return a * b

    result = f(
        cartesian_space,
        policy=SweepPolicy(skip_where=lambda p: p["a"] == 0.2),
    )
    assert len(calls) == 8
    assert (result.status.values == 3).sum() == 4


def test_nan_loop_values_are_not_skipped_by_default() -> None:
    """An unexpected NaN is more often a bug than a mask, so it is not magic."""
    seen: list[float] = []

    @sweep("loop(a) -> out()")
    def f(a: float) -> float:
        seen.append(a)
        return a

    space = xr.Dataset({"a": ("a", [1.0, np.nan, 3.0])})
    f(space)
    assert len(seen) == 3
    assert np.isnan(seen[1])
