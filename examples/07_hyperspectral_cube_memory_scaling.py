"""
Hyperspectral cube on a pixel map: what dedup buys at scale
==============================================================

The same path-radiance/SRF physics as the band-integration example, now on
an 80x80 pixel map (6400 pixels) with an 800-point wavelength axis instead
of a small aot x rh loop grid: a heavier per-call cost (each call processes
an 800-element spectrum against a 6x800 response table), run once without
and once with deduplication, to show what dedup buys when a single call is
no longer cheap. See ``04_band_integration_srf.py`` for the atmosphere
formula and its citations; this case reuses it unchanged.

Run with::

    pixi run -e dev python benchmarks/examples/07_hyperspectral_cube_memory_scaling.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

import xsweep

# sphinx-gallery executes examples without `__file__` set, so the repo root
# is located from the already-installed xsweep package instead.
sys.path.insert(0, str(Path(xsweep.__file__).resolve().parents[2]))

from benchmarks._lib.timing import record  # noqa: E402
from xsweep import SweepPolicy, sweep  # noqa: E402

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
    """TOA path radiance integrated against a sensor's spectral response."""
    CALLS["n"] += 1
    tau_total = _tau_rayleigh(wl) + aot * (wl / 0.55) ** (-_ALPHA)
    omega0 = min(0.80 + 0.0015 * rh, 0.999)
    rho_atm = (omega0 / (4.0 * (1.0 + _MU0))) * (
        1.0 - np.exp(-tau_total * (1.0 + 1.0 / _MU0))
    )
    l_toa = (_solar_shape(wl) / _E0_PEAK) / np.pi * rho_atm
    return (l_toa * srf).sum("wl")


def _pixel_map(size: int) -> xr.Dataset:
    """Build a size x size map covered by 8 distinct (aot, rh) scene classes."""
    classes = np.array(
        [
            (0.03, 20.0),
            (0.05, 40.0),
            (0.08, 60.0),
            (0.12, 30.0),
            (0.15, 70.0),
            (0.25, 50.0),
            (0.35, 20.0),
            (0.50, 80.0),
        ]
    )
    block = max(size // len(classes), 1)
    class_id = np.arange(size)[:, None] // block + np.arange(size)[None, :] // block
    class_id = class_id % len(classes)
    return xr.Dataset(
        {
            "aot": (("y", "x"), classes[class_id, 0]),
            "rh": (("y", "x"), classes[class_id, 1]),
        }
    )


def main() -> None:
    """Run the same 80x80 cube without, then with, deduplication."""
    wl = np.linspace(0.35, 2.5, 800)
    centres = np.array([0.47, 0.56, 0.65, 0.86, 1.61, 2.20])
    width = 0.03
    weights = np.exp(-(((wl[None, :] - centres[:, None]) / width) ** 2))
    weights /= weights.sum(axis=1, keepdims=True)

    pixels = _pixel_map(80)
    space = pixels.assign(
        wl=("wl", wl),
        srf=(("band", "wl"), weights),
    ).assign_coords(band=centres)
    n_pixels = space.sizes["y"] * space.sizes["x"]
    n_unique = len(set(zip(pixels.aot.values.ravel(), pixels.rh.values.ravel())))

    print("=== What it will cost ===\n")
    print(radiance.explain(space, policy=SweepPolicy(dedup=True)))
    print(f"\npixels: {n_pixels}, unique (aot, rh) pairs: {n_unique}")

    print("\n=== Without deduplication ===\n")
    CALLS["n"] = 0
    start = time.perf_counter()
    plain = radiance(space)
    plain_time = time.perf_counter() - start
    plain_calls = CALLS["n"]
    print(f"{plain_calls} calls in {plain_time:.2f}s")
    record(
        "07_hyperspectral_cube_memory_scaling_no_dedup",
        n_calls=plain_calls,
        wall_time_s=plain_time,
    )

    print("\n=== With deduplication ===\n")
    CALLS["n"] = 0
    start = time.perf_counter()
    deduped = radiance(space, policy=SweepPolicy(dedup=True))
    dedup_time = time.perf_counter() - start
    dedup_calls = CALLS["n"]
    print(f"{dedup_calls} calls in {dedup_time:.2f}s")
    record(
        "07_hyperspectral_cube_memory_scaling_dedup",
        n_calls=dedup_calls,
        wall_time_s=dedup_time,
    )

    print(f"\ncalls divided by {plain_calls / max(dedup_calls, 1):.0f}")
    print(f"wall time divided by {plain_time / dedup_time:.1f}")
    print(
        "bit-identical: "
        f"{np.array_equal(plain.radiance.values, deduped.radiance.values)}"
    )


if __name__ == "__main__":
    main()
