"""Quickstart scenario 3: dedup collapses calls, never results.

The sacred property is what these tests really check: a policy may change
the cost of a sweep, never its values or its shape.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsweep import SweepPolicy, sweep
from xsweep.errors import PolicyError


def test_dedup_reduces_calls_and_preserves_the_result(
    map_space: xr.Dataset,
) -> None:
    """Four unique rows among six pixels means four calls, same answer."""
    calls: list[tuple[float, float]] = []

    @sweep("loop(aot, rh) -> rho()")
    def f(aot: float, rh: float) -> float:
        calls.append((aot, rh))
        return aot * 100 + rh

    plain = f(map_space)
    n_plain = len(calls)

    calls.clear()
    deduped = f(map_space, policy=SweepPolicy(dedup=True))
    n_dedup = len(calls)

    assert n_plain == 6
    assert n_dedup == len(
        {
            (a, r)
            for a, r in zip(
                map_space.aot.values.ravel(), map_space.rh.values.ravel(), strict=True
            )
        }
    )
    assert n_dedup < n_plain
    assert np.array_equal(plain.rho.values, deduped.rho.values)
    assert plain.rho.dims == deduped.rho.dims


def test_dedup_handles_mixed_dtypes() -> None:
    """Factorising per column is what makes strings and floats coexist."""
    calls: list[tuple[float, str]] = []

    @sweep("loop(aot, profile) -> rho()")
    def f(aot: float, profile: str) -> float:
        calls.append((aot, profile))
        return aot * len(profile)

    space = xr.Dataset(
        {
            "aot": (("y", "x"), np.array([[0.1, 0.2], [0.1, 0.3]])),
            "profile": (("y", "x"), np.array([["ms", "t"], ["ms", "t"]])),
        }
    )
    plain = f(space)
    calls.clear()
    deduped = f(space, policy=SweepPolicy(dedup=True))
    assert len(calls) == 3
    assert np.array_equal(plain.rho.values, deduped.rho.values)


def test_partial_dedup_keeps_the_other_axis_fully_swept() -> None:
    """Collapsing space while keeping time is the documented use case."""
    calls: list[float] = []

    @sweep("loop(aot, hour) -> rho()")
    def f(aot: float, hour: float) -> float:
        calls.append(aot)
        return aot + hour

    space = xr.Dataset(
        {
            "aot": (("y", "x"), np.array([[0.1, 0.1], [0.1, 0.2]])),
            "hour": ("time", np.array([6.0, 12.0])),
        }
    )
    plain = f(space)
    calls.clear()
    deduped = f(space, policy=SweepPolicy(dedup=("y", "x")))

    # Two unique spatial rows times two hours, instead of eight points.
    assert len(calls) == 4
    assert np.array_equal(plain.rho.values, deduped.rho.values)
    assert deduped.sizes["time"] == 2


def test_dedup_over_an_unknown_dim_fails_before_any_call() -> None:
    """Policy errors must cost nothing, so they are raised while planning."""

    @sweep("loop(a) -> out()")
    def f(a: float) -> float:
        return a

    space = xr.Dataset({"a": ("a", [1.0, 2.0])})
    with pytest.raises(PolicyError, match="available dims: a"):
        f(space, policy=SweepPolicy(dedup=("z",)))


def test_plan_reports_the_unique_count(map_space: xr.Dataset) -> None:
    """The number that decides whether a map sweep is affordable."""

    @sweep("loop(aot, rh) -> rho()")
    def f(aot: float, rh: float) -> float:
        return aot

    plan = f.explain(map_space, policy=SweepPolicy(dedup=True))
    assert plan.n_points == 6
    assert plan.n_unique is not None and plan.n_unique < 6
    assert "unique" in repr(plan)
