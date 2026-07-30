"""
A lookup table reused across worker processes
=================================================

``SweepModule`` gives state built once a home (see ``docs/guide.md``,
section 8): here, a coarse Rayleigh optical-depth lookup table (the same
Bodhaine et al. (1999) formula as the spectrum example), built once in
``__init__`` and bilinearly interpolated by every call. Combined with
``executor="process"``, the instance, LUT included, travels into each
worker process intact.

Run with::

    pixi run -e dev python benchmarks/examples/09_sweepmodule_lut_process_executor.py
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
from xsweep import SweepModule, SweepPolicy  # noqa: E402


def _tau_rayleigh(pressure: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Bodhaine et al. (1999) Rayleigh optical depth, wl in um, on a grid."""
    tau = 0.002152 * (1.0455996 - 341.29061 * wl**-2 - 0.90230850 * wl**2)
    tau = tau / (1 + 0.0027059889 * wl**-2 - 85.968563 * wl**2)
    return tau * (pressure / 1013.25)


class RayleighLUT(SweepModule):
    """Interpolate a precomputed Rayleigh optical-depth table."""

    contract = "loop(pressure_q, wl_q) -> tau_r()"

    def __init__(self, policy: SweepPolicy | None = None) -> None:
        super().__init__(policy)
        self._pressures = np.linspace(700.0, 1013.25, 20)
        self._wls = np.linspace(0.35, 2.5, 20)
        pp, ww = np.meshgrid(self._pressures, self._wls, indexing="ij")
        self._lut = _tau_rayleigh(pp, ww)

    def forward(self, pressure_q: float, wl_q: float) -> float:
        """Bilinear interpolation on the lookup table built in __init__."""
        pi = int(np.clip(np.searchsorted(self._pressures, pressure_q) - 1, 0, 18))
        wi = int(np.clip(np.searchsorted(self._wls, wl_q) - 1, 0, 18))
        p0, p1 = self._pressures[pi], self._pressures[pi + 1]
        w0, w1 = self._wls[wi], self._wls[wi + 1]
        tp = (pressure_q - p0) / (p1 - p0)
        tw = (wl_q - w0) / (w1 - w0)
        f00, f10 = self._lut[pi, wi], self._lut[pi + 1, wi]
        f01, f11 = self._lut[pi, wi + 1], self._lut[pi + 1, wi + 1]
        return float(
            f00 * (1 - tp) * (1 - tw)
            + f10 * tp * (1 - tw)
            + f01 * (1 - tp) * tw
            + f11 * tp * tw
        )


def main() -> None:
    """Query the LUT on a 40x40 grid, computed across two worker processes."""
    space = xr.Dataset(
        {
            "pressure_q": ("pressure_q", np.linspace(720.0, 1000.0, 40)),
            "wl_q": ("wl_q", np.linspace(0.4, 2.4, 40)),
        }
    )
    n_points = space.sizes["pressure_q"] * space.sizes["wl_q"]

    module = RayleighLUT(SweepPolicy(executor="process", max_workers=2))

    print("=== What it will cost ===\n")
    print(module.explain(space))

    print("\n=== Running across 2 worker processes ===\n")
    start = time.perf_counter()
    result = module(space)
    elapsed = time.perf_counter() - start

    print(f"points: {n_points}")
    print(f"result: {dict(result.sizes)}")

    # Cross-check a handful of query points against the exact formula, not
    # the LUT: bilinear interpolation is approximate by construction.
    checks = [(850.0, 0.55), (950.0, 1.2), (750.0, 2.0)]
    max_error = 0.0
    for pressure_q, wl_q in checks:
        exact = float(_tau_rayleigh(np.array(pressure_q), np.array(wl_q)))
        interpolated = float(
            result.tau_r.sel(pressure_q=pressure_q, wl_q=wl_q, method="nearest")
        )
        max_error = max(max_error, abs(exact - interpolated))
    print(f"max error vs exact formula at 3 check points: {max_error:.2e}")

    bench = record(
        "09_sweepmodule_lut_process_executor", n_calls=n_points, wall_time_s=elapsed
    )
    print(f"\nrecorded: {bench.wall_time_s:.4f}s for {bench.n_calls} calls")


if __name__ == "__main__":
    main()
