"""
Band integration against a spectral response function
========================================================

TOA atmospheric path radiance, integrated against six synthetic sensor
bands. This is the case that needs both ``vec`` and ``const``: the response
function integration reduces over the whole wavelength axis, so batching
``wl`` would integrate each batch separately and silently corrupt the sum
(the contract refuses that combination at decoration time; see
``scripts/04_band_integration.py`` for the refusal itself).

The atmosphere is Rayleigh (Bodhaine et al., 1999, as in the Rayleigh
example) plus an Angstrom power-law aerosol term, ``tau_aer = aot *
(wl/0.55)^-alpha`` (Angstrom, 1929). Path radiance reuses the
single-scattering approximation from the dedup example, now wavelength
dependent. The solar source term is a blackbody at 5778 K (Planck's law),
normalised to its own peak, i.e. a relative spectral shape, not an absolute,
measured solar spectrum. Surface-reflected radiance is neglected (a dark
surface), so only the atmospheric path signal is modelled. The six band
response functions are synthetic Gaussians, not a real sensor's.

Run with::

    pixi run -e dev python benchmarks/examples/04_band_integration_srf.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import xsweep

# sphinx-gallery executes examples without `__file__` set, so the repo root
# is located from the already-installed xsweep package instead.
sys.path.insert(0, str(Path(xsweep.__file__).resolve().parents[2]))

from benchmarks._lib.timing import record  # noqa: E402
from xsweep import sweep  # noqa: E402

CALLS = {"n": 0}

_ALPHA = 1.3  # representative Angstrom exponent for continental aerosols
_MU0 = np.cos(np.radians(30.0))  # fixed 30 degree solar zenith angle
_H, _C, _KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23  # exact SI constants
_T_SUN = 5778.0  # K, blackbody approximation of the solar spectrum
_WIEN_B = 2.8977719e-3  # m*K, Wien's displacement law constant


def _tau_rayleigh(wl: xr.DataArray) -> xr.DataArray:
    """Bodhaine et al. (1999) Rayleigh optical depth at sea level, wl in um."""
    tau = 0.002152 * (1.0455996 - 341.29061 * wl**-2 - 0.90230850 * wl**2)
    return tau / (1 + 0.0027059889 * wl**-2 - 85.968563 * wl**2)


def _solar_shape(wl_um: xr.DataArray | float) -> xr.DataArray | float:
    """Planck's law at 5778 K, wl in micrometers, unnormalised."""
    wl_m = wl_um * 1e-6
    x = _H * _C / (wl_m * _KB * _T_SUN)
    return (2.0 * _H * _C**2 / wl_m**5) / (np.exp(x) - 1.0)


_E0_PEAK = _solar_shape(_WIEN_B / _T_SUN * 1e6)  # normalises the shape to 1


@sweep("loop(aot, rh) vec(wl) const(srf) -> radiance(band)")
def radiance(
    aot: float, rh: float, wl: xr.DataArray, srf: xr.DataArray
) -> xr.DataArray:
    """TOA path radiance integrated against a sensor's spectral response.

    ``wl`` arrives whole because the integration reduces over it, and
    ``srf`` arrives whole because it is context rather than a swept
    parameter.
    """
    CALLS["n"] += 1
    tau_total = _tau_rayleigh(wl) + aot * (wl / 0.55) ** (-_ALPHA)
    omega0 = min(0.80 + 0.0015 * rh, 0.999)
    rho_atm = (omega0 / (4.0 * (1.0 + _MU0))) * (
        1.0 - np.exp(-tau_total * (1.0 + 1.0 / _MU0))
    )
    l_toa = (_solar_shape(wl) / _E0_PEAK) / np.pi * rho_atm
    return (l_toa * srf).sum("wl")


def _srf(wl: np.ndarray, centres: np.ndarray, width: float) -> np.ndarray:
    """Synthetic, normalised Gaussian response functions, one row per band."""
    weights = np.exp(-(((wl[None, :] - centres[:, None]) / width) ** 2))
    return weights / weights.sum(axis=1, keepdims=True)


def main() -> None:
    """Sweep band radiance over aot x rh, wl handed whole to each call."""
    wl = np.linspace(0.35, 2.5, 300)
    centres = np.array([0.47, 0.56, 0.65, 0.86, 1.61, 2.20])
    weights = _srf(wl, centres, width=0.03)

    space = xr.Dataset(
        {
            "aot": ("aot", np.linspace(0.02, 0.6, 10)),
            "rh": ("rh", np.linspace(10.0, 90.0, 8)),
            "wl": ("wl", wl),
            "srf": (("band", "wl"), weights),
        },
        coords={"band": centres},
    )

    print("=== What it will cost ===\n")
    print(radiance.explain(space))

    print("\n=== Running ===\n")
    start = time.perf_counter()
    result = radiance(space)
    elapsed = time.perf_counter() - start

    print(f"calls made: {CALLS['n']}")
    print(f"result: {dict(result.sizes)}")
    print("\nradiance by band, at aot=0.2, rh=50:")
    print(result.radiance.sel(aot=0.2, rh=50.0, method="nearest").round(4).values)

    fig, ax = plt.subplots()
    ax.bar([f"{c:.2f}" for c in centres], result.radiance.mean(("aot", "rh")))
    ax.set_xlabel("band centre (µm)")
    ax.set_ylabel("mean radiance (relative units)")
    fig.tight_layout()

    bench = record("04_band_integration_srf", n_calls=CALLS["n"], wall_time_s=elapsed)
    print(f"\nrecorded: {bench.wall_time_s:.4f}s for {bench.n_calls} calls")


if __name__ == "__main__":
    main()
