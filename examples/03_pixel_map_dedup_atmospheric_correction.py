"""
Atmospheric-correction path reflectance on a pixel map, deduplicated
=======================================================================

A satellite scene rarely has as many distinct (AOT, RH) pairs as it has
pixels: water, cloud, and large uniform land patches repeat. ``dedup=True``
computes each unique pair once and copies the result onto every pixel that
shares it, and persisting to a store adds resumability on top for free.

``rho_atm`` is a single-scattering approximation of the atmospheric path
reflectance for a fixed nadir view and 30 degree solar zenith angle, with an
isotropic phase function; it is the standard single-scattering solution
used in atmospheric correction (see e.g. Chandrasekhar, 1960, *Radiative
Transfer*, and the 6S radiative transfer model of Vermote et al., 1997) with
the phase function set to 1. The aerosol single-scattering albedo's rise
with relative humidity is an illustrative hygroscopic-growth trend, not fit
to a specific published aerosol model.

Run with::

    pixi run -e dev python benchmarks/examples/03_pixel_map_dedup_atmospheric_correction.py
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

_MU0 = np.cos(np.radians(30.0))  # fixed 30 degree solar zenith angle
_MU = 1.0  # fixed nadir view


@sweep("loop(aot, rh) -> rho_atm()")
def rho_atm(aot: float, rh: float) -> float:
    """Stand in for a per-pixel atmospheric-correction engine call."""
    CALLS["n"] += 1
    time.sleep(0.001)
    omega0 = min(0.80 + 0.0015 * rh, 0.999)
    return float(
        (omega0 / (4.0 * (_MU + _MU0)))
        * (1.0 - np.exp(-aot * (1.0 / _MU + 1.0 / _MU0)))
    )


def _pixel_map() -> xr.Dataset:
    """Build a 60x60 pixel map covered by only 6 distinct (aot, rh) classes."""
    classes = np.array(
        [
            (0.05, 30.0),
            (0.05, 70.0),
            (0.15, 30.0),
            (0.15, 70.0),
            (0.30, 50.0),
            (0.50, 20.0),
        ]
    )
    size = 60
    class_id = np.arange(size)[:, None] // 10 + np.arange(size)[None, :] // 10
    class_id = class_id % len(classes)
    return xr.Dataset(
        {
            "aot": (("y", "x"), classes[class_id, 0]),
            "rh": (("y", "x"), classes[class_id, 1]),
        }
    )


def main() -> None:
    """Compare with/without dedup, then persist and resume the deduped run."""
    space = _pixel_map()
    n_pixels = space.sizes["y"] * space.sizes["x"]
    n_unique = len(set(zip(space.aot.values.ravel(), space.rh.values.ravel())))

    print("=== What deduplication buys, before running ===\n")
    print(rho_atm.explain(space, policy=SweepPolicy(dedup=True)))
    print(f"\npixels: {n_pixels}, unique (aot, rh) pairs: {n_unique}")

    print("\n=== Without deduplication, in memory ===\n")
    CALLS["n"] = 0
    start = time.perf_counter()
    plain = rho_atm(space)
    plain_time = time.perf_counter() - start
    plain_calls = CALLS["n"]
    print(f"{plain_calls} calls in {plain_time:.2f}s")

    store = Path(tempfile.mkdtemp()) / "rho_atm.zarr"
    try:
        print("\n=== With deduplication, persisted to a store ===\n")
        CALLS["n"] = 0
        start = time.perf_counter()
        deduped = rho_atm(space, policy=SweepPolicy(store=str(store), dedup=True))
        dedup_time = time.perf_counter() - start
        dedup_calls = CALLS["n"]
        print(f"{dedup_calls} calls in {dedup_time:.2f}s")
        print(f"calls divided by {plain_calls / max(dedup_calls, 1):.0f}")
        print(
            f"bit-identical: {np.array_equal(plain.rho_atm.values, deduped.rho_atm.values)}"
        )

        print("\n=== Same store again (resume) ===\n")
        CALLS["n"] = 0
        resumed = rho_atm(space, policy=SweepPolicy(store=str(store), dedup=True))
        print(f"{CALLS['n']} calls")
        print(
            "bit-identical: "
            f"{np.array_equal(deduped.rho_atm.values, resumed.rho_atm.values)}"
        )

        bench = record(
            "03_pixel_map_dedup_atmospheric_correction",
            n_calls=dedup_calls,
            wall_time_s=dedup_time,
        )
        print(f"\nrecorded: {bench.wall_time_s:.4f}s for {bench.n_calls} calls")
    finally:
        shutil.rmtree(store.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
