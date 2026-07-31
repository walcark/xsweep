"""
TOA reflectance over a big optical-property grid
===================================================

Single-scattering approximation of top-of-atmosphere reflectance for a
homogeneous layer, swept over optical thickness, single-scattering albedo,
asymmetry factor and solar zenith angle: a large, purely ``loop`` sweep with
no ``vec``/``const`` clause at all, to show xsweep's cost at scale rather
than a new contract feature.

``reflectance`` is the standard single-scattering solution (Chandrasekhar,
1960, *Radiative Transfer*) for a fixed nadir view, using the
Henyey-Greenstein phase function (Henyey & Greenstein, 1941, *Diffuse
radiation in the Galaxy*, ApJ, 93, 70-83) for the asymmetry-factor
dependence::

    P(g, mu0) = (1 - g**2) / (1 + g**2 + 2*g*mu0)**1.5
    reflectance = ssa/(4*(1+mu0)) * P(g, mu0) * (1 - exp(-tau*(1+1/mu0)))

An earlier draft of this case used the Meador & Weaver (1980) two-stream
Eddington closure instead; it was dropped after its ``ssa=0`` limit failed a
basic sanity check (no scattering must mean no reflectance) that this
formula passes, and the exact sign convention could not be pinned down with
confidence from secondary sources. Single scattering only, not the full
two-stream solution.

Run with::

    pixi run -e dev python benchmarks/examples/05_single_scattering_toa_reflectance.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
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


@sweep("loop(tau, ssa, g, mu0) -> reflectance()")
def reflectance(tau: float, ssa: float, g: float, mu0: float) -> float:
    """Single-scattering TOA reflectance, Henyey-Greenstein phase function."""
    CALLS["n"] += 1
    phase = (1.0 - g**2) / (1.0 + g**2 + 2.0 * g * mu0) ** 1.5
    return float(
        (ssa / (4.0 * (1.0 + mu0))) * phase * (1.0 - np.exp(-tau * (1.0 + 1.0 / mu0)))
    )


def main() -> None:
    """Sweep reflectance over a 20x15x15x6 = 27000-point grid, persisted."""
    space = xr.Dataset(
        {
            "tau": ("tau", np.linspace(0.05, 15.0, 20)),
            "ssa": ("ssa", np.linspace(0.3, 1.0, 15)),
            "g": ("g", np.linspace(-0.5, 0.9, 15)),
            "mu0": ("mu0", np.cos(np.radians([0.0, 15.0, 30.0, 45.0, 60.0, 75.0]))),
        }
    )

    print("=== What it will cost ===\n")
    print(reflectance.explain(space))

    store = Path(tempfile.mkdtemp()) / "reflectance.zarr"
    try:
        print("\n=== Running, persisted to a store ===\n")
        start = time.perf_counter()
        result = reflectance(space, policy=SweepPolicy(store=str(store)))
        elapsed = time.perf_counter() - start

        print(f"calls made: {CALLS['n']}")
        print(f"result: {dict(result.sizes)}")
        print(
            "reflectance range: "
            f"[{float(result.reflectance.min()):.4f}, "
            f"{float(result.reflectance.max()):.4f}]"
        )

        bench = record(
            "05_single_scattering_toa_reflectance",
            n_calls=CALLS["n"],
            wall_time_s=elapsed,
        )
        print(f"\nrecorded: {bench.wall_time_s:.4f}s for {bench.n_calls} calls")
    finally:
        shutil.rmtree(store.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
