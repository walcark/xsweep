"""The ten worked examples of the design reference, section 8.

Nine were claimed expressible in v0 without library changes. This asserts it
by planning each one: planning resolves the contract against the space and
the policy, so a case that could not be expressed fails here. The tenth
(dependent, ragged axes) is covered by its documented workaround.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xsweep import SweepModule, SweepPolicy, sweep


def _map(size: int = 8) -> xr.Dataset:
    """Return a small pixel map with repeated values."""
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "aot": (("y", "x"), rng.choice([0.1, 0.2, 0.3], size=(size, size))),
            "rh": (("y", "x"), rng.choice([30.0, 70.0], size=(size, size))),
        }
    )


def test_1_cartesian_sensitivity_study() -> None:
    """Vary three parameters independently over a spectral axis."""

    @sweep("loop(aot, rh, sza) vec(wl @ 8) -> tdir(wl)")
    def f(aot: float, rh: float, sza: float, wl: xr.DataArray) -> xr.DataArray:
        return wl * aot

    space = xr.Dataset(
        {
            "aot": ("aot", [0.05, 0.1, 0.3]),
            "rh": ("rh", [30.0, 70.0]),
            "sza": ("sza", [0.0, 30.0]),
            "wl": ("wl", np.arange(24.0)),
        }
    )
    plan = f.explain(space)
    assert plan.n_points == 12
    assert plan.n_calls == 12 * 3


def test_2_pixel_map_with_dedup() -> None:
    """The map case: zipped dims, and duplicates collapsed."""

    @sweep("loop(aot, rh) -> rho()")
    def f(aot: float, rh: float) -> float:
        return aot + rh

    space = _map()
    plan = f.explain(space, policy=SweepPolicy(dedup=True))
    assert plan.loop_dims == ("y", "x")
    assert plan.n_unique is not None and plan.n_unique <= 6


def test_3_map_crossed_with_a_diurnal_axis() -> None:
    """A zipped map multiplied by an independent time axis."""

    @sweep("loop(aot, rh, hour) -> rho()")
    def f(aot: float, rh: float, hour: float) -> float:
        return aot + hour

    space = _map().assign({"hour": ("time", [6.0, 12.0, 18.0])})
    plan = f.explain(space)
    assert plan.loop_dims == ("y", "x", "time")
    assert plan.n_points == 8 * 8 * 3


def test_4_multi_geometry_folded_into_one_call() -> None:
    """Several geometry axes handed to the engine at once."""

    @sweep("loop(aot) vec(wl @ 8, vza, phi) -> rad(wl, vza, phi)")
    def f(
        aot: float, wl: xr.DataArray, vza: xr.DataArray, phi: xr.DataArray
    ) -> xr.DataArray:
        return wl * aot

    space = xr.Dataset(
        {
            "aot": ("aot", [0.1, 0.2]),
            "wl": ("wl", np.arange(16.0)),
            "vza": ("vza", np.arange(5.0)),
            "phi": ("phi", np.arange(4.0)),
        }
    )
    plan = f.explain(space)
    assert plan.n_points == 2
    assert plan.n_calls == 2 * 2


def test_5_monte_carlo_convergence() -> None:
    """Replication through a seed carrier variable."""

    @sweep("loop(aot, seed) -> rho()")
    def f(aot: float, seed: int) -> float:
        return float(aot + seed)

    space = xr.Dataset({"aot": ("aot", [0.1, 0.3]), "seed": ("rep", np.arange(20))})
    result = f(space)
    assert dict(result.sizes) == {"aot": 2, "rep": 20}


def test_6_band_integration_with_a_response_function() -> None:
    """A reduction over the spectral axis, with the response passed whole."""

    @sweep("loop(aot) vec(wl) const(srf) -> band_int(band)")
    def f(aot: float, wl: xr.DataArray, srf: xr.DataArray) -> xr.DataArray:
        return ((aot * wl) * srf).sum("wl")

    space = xr.Dataset(
        {
            "aot": ("aot", [0.1, 0.2]),
            "wl": ("wl", np.arange(10.0)),
            "srf": (("band", "wl"), np.ones((3, 10)) / 10),
        }
    )
    result = f(space)
    assert dict(result.sizes) == {"aot": 2, "band": 3}


def test_7_three_albedo_inversion() -> None:
    """An albedo axis swept, then consumed by ordinary xarray downstream."""

    @sweep("loop(aot, albedo) -> toa()")
    def f(aot: float, albedo: float) -> float:
        return aot + albedo

    space = xr.Dataset(
        {"aot": ("aot", [0.1, 0.2]), "albedo": ("albedo", [0.0, 0.5, 1.0])}
    )
    result = f(space)
    # The inversion itself is plain xarray, which is the point: xsweep stops
    # where the pipeline begins.
    slope = result.toa.diff("albedo").isel(albedo=0)
    assert slope.sizes == {"aot": 2}


def test_8_parametrised_profiles_by_label() -> None:
    """String axes, the documented alternative to object-valued parameters."""

    @sweep("loop(aot, profile) -> rho()")
    def f(aot: float, profile: str) -> float:
        return aot * len(profile)

    space = xr.Dataset(
        {
            "aot": ("aot", [0.1, 0.2]),
            "profile": ("profile", ["afgl_ms", "afgl_t"]),
        }
    )
    result = f(space)
    assert dict(result.sizes) == {"aot": 2, "profile": 2}


def test_9_cross_engine_comparison() -> None:
    """Two modules, one space, results concatenated on an explicit axis."""

    class EngineA(SweepModule):
        contract = "loop(aot) -> rho()"

        def forward(self, aot: float) -> float:
            return aot * 2.0

    class EngineB(SweepModule):
        contract = "loop(aot) -> rho()"

        def forward(self, aot: float) -> float:
            return aot * 3.0

    space = xr.Dataset({"aot": ("aot", [0.1, 0.2])})
    compared = xr.concat([EngineA()(space).rho, EngineB()(space).rho], dim="engine")
    assert compared.sizes == {"engine": 2, "aot": 2}


def test_10_ragged_axes_use_the_union_grid_workaround() -> None:
    """The one example v0 cannot express directly.

    A wavelength support that depends on the band does not fit rectangular
    axes. The documented workaround is a union grid with zero weights, which
    is physically clean and needs no library change.
    """

    @sweep("loop(aot) vec(wl) const(srf) -> band_int(band)")
    def f(aot: float, wl: xr.DataArray, srf: xr.DataArray) -> xr.DataArray:
        return ((aot * wl) * srf).sum("wl")

    # Band 0 covers the first half, band 1 the second: zero weights outside.
    srf = np.zeros((2, 10))
    srf[0, :5] = 0.2
    srf[1, 5:] = 0.2
    space = xr.Dataset(
        {
            "aot": ("aot", [1.0]),
            "wl": ("wl", np.arange(10.0)),
            "srf": (("band", "wl"), srf),
        }
    )
    result = f(space)
    assert result.band_int.values[0].tolist() == pytest.approx([2.0, 7.0])
