"""
Rayleigh optical depth spectrum
=================================

The Rayleigh optical depth of a clear atmosphere as a function of
wavelength and surface pressure, using the empirical fit of Bodhaine et al.
(1999)::

    tau_R(wl) = 0.002152 * (1.0455996 - 341.29061/wl**2 - 0.90230850*wl**2)
                / (1 + 0.0027059889/wl**2 - 85.968563*wl**2)

valid at sea level (1013.25 hPa) for wavelength ``wl`` in micrometers;
surface pressure enters as the ratio ``pressure / 1013.25`` (Bodhaine, Wood,
Dutton & Slusser, 1999, *On Rayleigh Optical Depth Calculations*, J. Atmos.
Oceanic Technol., 16, 1854-1861).

Demonstrates ``vec`` with the ``@ N`` batch marker: the wavelength axis is
split into chunks of 500 points, and the function is elementwise in
wavelength, so batching never changes the result.

Run with::

    pixi run -e dev python benchmarks/examples/02_rayleigh_optical_depth_spectrum.py
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


@sweep("loop(pressure) vec(wl @ 500) -> tau_r(wl)")
def tau_rayleigh(pressure: float, wl: xr.DataArray) -> xr.DataArray:
    """Bodhaine et al. (1999) Rayleigh optical depth, wl in micrometers."""
    CALLS["n"] += 1
    tau_sea_level = 0.002152 * (1.0455996 - 341.29061 * wl**-2 - 0.90230850 * wl**2)
    tau_sea_level = tau_sea_level / (1 + 0.0027059889 * wl**-2 - 85.968563 * wl**2)
    return tau_sea_level * (pressure / 1013.25)


def main() -> None:
    """Sweep the Rayleigh spectrum over station pressure, batched over wl."""
    space = xr.Dataset(
        {
            "pressure": ("pressure", [700.0, 800.0, 900.0, 1000.0, 1013.25]),
            "wl": ("wl", np.linspace(0.35, 2.5, 2000)),
        }
    )

    print("=== What it will cost ===\n")
    print(tau_rayleigh.explain(space))

    print("\n=== Running ===\n")
    start = time.perf_counter()
    result = tau_rayleigh(space)
    elapsed = time.perf_counter() - start

    print(f"calls made: {CALLS['n']}")
    print(f"result: {dict(result.sizes)}")

    fig, ax = plt.subplots()
    for pressure in (700.0, 1013.25):
        curve = result.tau_r.sel(pressure=pressure)
        ax.plot(result.wl, curve, label=f"{pressure:.0f} hPa")
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("Rayleigh optical depth")
    ax.legend()
    fig.tight_layout()

    bench = record(
        "02_rayleigh_optical_depth_spectrum",
        n_calls=CALLS["n"],
        wall_time_s=elapsed,
    )
    print(f"\nrecorded: {bench.wall_time_s:.4f}s for {bench.n_calls} calls")


if __name__ == "__main__":
    main()
