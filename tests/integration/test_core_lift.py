"""Quickstart scenario 1: the core lift.

Every assertion here is a call count or a shape, because that is what the
specification promises: the same result whatever the policy, computed once
per point.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr

from xsweep import SweepPolicy, sweep


def test_cartesian_sweep_shape_and_call_count(
    cartesian_space: xr.Dataset,
) -> None:
    """Distinct dims multiply, and each point is computed exactly once."""
    calls: list[tuple[float, float]] = []

    @sweep("loop(a, b) -> out()")
    def f(a: float, b: float) -> float:
        calls.append((a, b))
        return a * b

    result = f(cartesian_space)
    assert dict(result.sizes) == {"a": 3, "b": 4}
    assert len(calls) == 12
    assert result.out.sel(a=0.2, b=30.0).item() == pytest.approx(6.0)


def test_loop_values_arrive_as_native_python(
    cartesian_space: xr.Dataset,
) -> None:
    """External engines want plain floats, not 0-d arrays."""
    seen: list[type] = []

    @sweep("loop(a, b) -> out()")
    def f(a: float, b: float) -> float:
        seen.append(type(a))
        return a + b

    f(cartesian_space)
    assert set(seen) == {float}


def test_shared_dims_zip_instead_of_multiplying(map_space: xr.Dataset) -> None:
    """Six pixels give six calls, not the thirty-six a product would."""
    calls: list[tuple[float, float]] = []

    @sweep("loop(aot, rh) -> rho()")
    def f(aot: float, rh: float) -> float:
        calls.append((aot, rh))
        return aot + rh

    result = f(map_space)
    assert dict(result.sizes) == {"y": 2, "x": 3}
    assert len(calls) == 6


def test_second_run_hits_the_cache(tmp_path, map_space: xr.Dataset) -> None:
    """Re-running an identical sweep must not spend a single call."""
    calls: list[float] = []

    @sweep("loop(aot, rh) -> rho()", version="1")
    def f(aot: float, rh: float) -> float:
        calls.append(aot)
        return aot + rh

    store = str(tmp_path / "r.zarr")
    first = f(map_space, policy=SweepPolicy(store=store))
    assert len(calls) == 6

    calls.clear()
    second = f(map_space, policy=SweepPolicy(store=store))
    assert calls == []
    assert np.allclose(first.rho.values, second.rho.values)


def test_zero_dimensional_variable_creates_no_axis() -> None:
    """A fixed-but-present value is delivered without becoming a dim."""
    seen: list[float] = []

    @sweep("loop(a, sza) -> out()")
    def f(a: float, sza: float) -> float:
        seen.append(sza)
        return a * sza

    space = xr.Dataset({"a": ("a", [1.0, 2.0]), "sza": ((), 35.0)})
    result = f(space)
    assert dict(result.sizes) == {"a": 2}
    assert seen == [35.0, 35.0]


def test_status_variable_accompanies_the_result(
    cartesian_space: xr.Dataset,
) -> None:
    """Resume and failure must be distinguishable, hence the sidecar."""

    @sweep("loop(a, b) -> out()")
    def f(a: float, b: float) -> float:
        return a

    result = f(cartesian_space)
    assert "status" in result
    assert (result.status.values == 1).all()


def test_in_memory_mode_is_announced(
    caplog: pytest.LogCaptureFixture, cartesian_space: xr.Dataset
) -> None:
    """Running without persistence must never be silent."""

    @sweep("loop(a, b) -> out()")
    def f(a: float, b: float) -> float:
        return a

    with caplog.at_level(logging.INFO, logger="xsweep"):
        f(cartesian_space)
    assert "not persisted" in caplog.text
    assert "sweep done ok=12" in caplog.text


def test_store_and_memory_modes_agree(tmp_path, cartesian_space: xr.Dataset) -> None:
    """The two modes share one code path, so they must agree exactly."""

    @sweep("loop(a, b) -> out()")
    def f(a: float, b: float) -> float:
        return a * b

    memory = f(cartesian_space)
    persisted = f(cartesian_space, policy=SweepPolicy(store=str(tmp_path / "s.zarr")))
    assert np.array_equal(memory.out.values, persisted.out.values)
