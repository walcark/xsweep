"""
Henyey-Greenstein phase function
==================================

The single-parameter scattering phase function of Henyey & Greenstein
(1941, *Diffuse radiation in the Galaxy*, ApJ, 93, 70-83), used already
(unlabelled as such) in the two path-radiance examples earlier in this
gallery::

    P(theta; g) = (1 - g**2) / (1 + g**2 - 2*g*cos(theta))**1.5

``g`` is the asymmetry factor (0 = isotropic, near 1 = strongly
forward-scattering, near -1 = strongly backward-scattering); ``theta`` is
the scattering angle in radians. The smallest ``vec`` case in this gallery:
no ``const``, a single loop dimension.

Run with::

    pixi run -e dev python benchmarks/examples/06_henyey_greenstein_phase_function.py
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


@sweep("loop(g) vec(theta @ 250) -> phase(theta)")
def phase(g: float, theta: xr.DataArray) -> xr.DataArray:
    """Henyey-Greenstein phase function, theta in radians."""
    CALLS["n"] += 1
    return (1.0 - g**2) / (1.0 + g**2 - 2.0 * g * np.cos(theta)) ** 1.5


def main() -> None:
    """Sweep the phase function over asymmetry factor, batched over theta."""
    space = xr.Dataset(
        {
            "g": ("g", np.linspace(-0.9, 0.9, 12)),
            "theta": ("theta", np.linspace(0.0, np.pi, 1000)),
        }
    )

    print("=== What it will cost ===\n")
    print(phase.explain(space))

    print("\n=== Running ===\n")
    start = time.perf_counter()
    result = phase(space)
    elapsed = time.perf_counter() - start

    print(f"calls made: {CALLS['n']}")
    print(f"result: {dict(result.sizes)}")

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    for g in (-0.8, 0.0, 0.8):
        curve = result.phase.sel(g=g, method="nearest")
        ax.plot(result.theta, curve, label=f"g={g}")
    ax.set_theta_zero_location("N")
    ax.legend(loc="lower right")

    bench = record(
        "06_henyey_greenstein_phase_function", n_calls=CALLS["n"], wall_time_s=elapsed
    )
    print(f"\nrecorded: {bench.wall_time_s:.4f}s for {bench.n_calls} calls")


if __name__ == "__main__":
    main()
