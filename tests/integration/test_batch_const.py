"""Quickstart scenario 4: batched axes and context data."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsweep import SweepPolicy, sweep
from xsweep.errors import ContractError


def test_batches_respect_the_declared_maximum() -> None:
    """Twenty wavelengths at eight per call means 8, 8 and 4."""
    widths: list[int] = []

    @sweep("loop(a) vec(wl @ 8) -> t(wl)")
    def f(a: float, wl: xr.DataArray) -> xr.DataArray:
        widths.append(wl.sizes["wl"])
        return wl * a

    space = xr.Dataset({"a": ("a", [1.0]), "wl": ("wl", np.arange(20.0))})
    result = f(space)
    assert widths == [8, 8, 4]
    assert np.array_equal(result.t.values[0], np.arange(20.0))


def test_a_short_last_batch_is_normal_and_correct() -> None:
    """The documented golden case: sizes that do not divide evenly."""

    @sweep("loop(a) vec(wl @ 7) -> t(wl)")
    def f(a: float, wl: xr.DataArray) -> xr.DataArray:
        return wl * a

    space = xr.Dataset({"a": ("a", [2.0]), "wl": ("wl", np.arange(20.0))})
    result = f(space)
    assert np.array_equal(result.t.values[0], np.arange(20.0) * 2.0)


def test_const_arrives_whole_in_every_call() -> None:
    """Context data is never chunked, which is what makes reduction safe."""
    shapes: list[tuple[int, ...]] = []

    @sweep("loop(aot) vec(wl) const(srf) -> band_int(band)")
    def f(aot: float, wl: xr.DataArray, srf: xr.DataArray) -> xr.DataArray:
        shapes.append(srf.shape)
        return ((aot * wl) * srf).sum("wl")

    space = xr.Dataset(
        {
            "aot": ("aot", [0.1, 0.2]),
            "wl": ("wl", np.linspace(400.0, 900.0, 10)),
            "srf": (("band", "wl"), np.ones((3, 10)) / 10),
        }
    )
    result = f(space)
    assert shapes == [(3, 10), (3, 10)]
    assert dict(result.sizes) == {"aot": 2, "band": 3}


def test_batching_a_reduced_dim_is_refused_at_decoration() -> None:
    """Splitting an axis the callable integrates over corrupts the integral."""
    with pytest.raises(ContractError, match="not among the declared output dims"):

        @sweep("loop(aot) vec(wl @ 8) const(srf) -> band_int(band)")
        def f(aot: float, wl: xr.DataArray, srf: xr.DataArray) -> xr.DataArray:
            return ((aot * wl) * srf).sum("wl")


def test_multi_dim_vec_runs_whole_then_tiled_with_the_same_result() -> None:
    """The elementwise case: batching is a cost choice, never a value one."""
    calls = {"n": 0}

    @sweep("vec(A, B) -> C(y, x)")
    def combine(A: xr.DataArray, B: xr.DataArray, *, k: float) -> xr.DataArray:
        calls["n"] += 1
        return A + k * B

    rng = np.random.default_rng(0)
    space = xr.Dataset(
        {
            "A": (("y", "x"), rng.random((8, 8))),
            "B": (("y", "x"), rng.random((8, 8))),
        }
    )
    whole = combine(space, k=2.0)
    assert calls["n"] == 1

    calls["n"] = 0
    tiled = combine(space, policy=SweepPolicy(chunks={"y": 4, "x": 4}), k=2.0)
    assert calls["n"] == 4
    assert np.array_equal(whole.C.values, tiled.C.values)


def test_a_scalar_engine_and_a_vector_engine_agree_on_the_same_data() -> None:
    """Same physics, two contracts: the choice is about the callee, not data."""
    rng = np.random.default_rng(1)
    space = xr.Dataset(
        {
            "A": (("y", "x"), rng.random((4, 4))),
            "B": (("y", "x"), rng.random((4, 4))),
        }
    )

    @sweep("vec(A, B) -> C(y, x)")
    def vectorised(A: xr.DataArray, B: xr.DataArray) -> xr.DataArray:
        return A + 2.0 * B

    @sweep("loop(A, B) -> C()")
    def scalar(A: float, B: float) -> float:
        return A + 2.0 * B

    assert np.allclose(vectorised(space).C.values, scalar(space).C.values)
