"""
Beer-Lambert transmission
==========================

The simplest case in this gallery: a scalar function, an in-memory Cartesian
sweep, no store. Transmission follows the Beer-Lambert extinction law with a
plane-parallel (secant) airmass approximation, standard in atmospheric
radiative transfer (see e.g. Liou, *An Introduction to Atmospheric
Radiation*, 2002).

Run with::

    pixi run -e dev python benchmarks/examples/01_beer_lambert_transmission.py
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
from xsweep import sweep  # noqa: E402

CALLS = {"n": 0}


@sweep("loop(aot, sza) -> transmission()")
def transmission(aot: float, sza: float) -> float:
    """Beer-Lambert transmission at a given aerosol optical thickness and SZA."""
    CALLS["n"] += 1
    airmass = 1.0 / np.cos(np.radians(sza))
    return float(np.exp(-aot * airmass))


def main() -> None:
    """Sweep transmission over a small aot x sza grid, in memory."""
    space = xr.Dataset(
        {
            "aot": ("aot", np.linspace(0.01, 1.0, 25)),
            "sza": ("sza", np.linspace(0.0, 80.0, 20)),
        }
    )

    print("=== What it will cost ===\n")
    print(transmission.explain(space))

    print("\n=== Running ===\n")
    start = time.perf_counter()
    result = transmission(space)
    elapsed = time.perf_counter() - start

    print(f"calls made: {CALLS['n']}")
    print(f"result: {dict(result.sizes)}")
    print(
        "transmission range: "
        f"[{float(result.transmission.min()):.4f}, "
        f"{float(result.transmission.max()):.4f}]"
    )

    bench = record(
        "01_beer_lambert_transmission", n_calls=CALLS["n"], wall_time_s=elapsed
    )
    print(f"\nrecorded: {bench.wall_time_s:.4f}s for {bench.n_calls} calls")


if __name__ == "__main__":
    main()
