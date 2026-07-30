"""
Monte-Carlo ensemble over a seed dimension
=============================================

The documented idiom for replication (see ``docs/idioms.md``): a bare
repeat dimension does not work, since a contract loops over variables, not
dims, so the repeat index has to be an actual swept *variable*, ``seed``,
consumed by the callee to build a per-point random generator.

This is a toy Monte-Carlo uncertainty-propagation exercise, not a real
Monte-Carlo radiative-transfer solver: it propagates a synthetic sub-pixel
spread in aerosol optical thickness through the Beer-Lambert transmission
law from the first example in this gallery, and reports the ensemble mean
and spread across seeds at fixed AOT.

Run with::

    pixi run -e dev python benchmarks/examples/08_monte_carlo_seed_ensemble.py
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

_MU0 = np.cos(np.radians(30.0))  # fixed 30 degree solar zenith angle
_N_SAMPLES = 200


@sweep("loop(seed, aot) -> radiance()")
def radiance(seed: int, aot: float) -> float:
    """Monte-Carlo mean transmission under a synthetic sub-pixel AOT spread."""
    CALLS["n"] += 1
    rng = np.random.default_rng(int(seed))
    samples = rng.normal(loc=aot, scale=0.1 * max(aot, 1e-6), size=_N_SAMPLES)
    samples = np.clip(samples, 0.0, None)
    return float(np.mean(np.exp(-samples / _MU0)))


def main() -> None:
    """Sweep a 20-seed ensemble over 15 AOT values, seed used to build the RNG."""
    space = xr.Dataset(
        {
            "seed": ("seed", np.arange(20)),
            "aot": ("aot", np.linspace(0.05, 0.6, 15)),
        }
    )

    print("=== What it will cost ===\n")
    print(radiance.explain(space))

    print("\n=== Running ===\n")
    start = time.perf_counter()
    result = radiance(space)
    elapsed = time.perf_counter() - start

    main_calls = CALLS["n"]
    print(f"calls made: {main_calls}")
    print(f"result: {dict(result.sizes)}")

    ensemble = result.radiance.sel(aot=0.3, method="nearest")
    print(
        f"\nat aot~=0.3: mean={float(ensemble.mean()):.4f}, "
        f"std={float(ensemble.std()):.4f} across {ensemble.sizes['seed']} seeds"
    )

    print("\n=== Same seed, same result ===\n")
    repeat = radiance(space.isel(seed=[0], aot=[0]))
    original = result.isel(seed=[0], aot=[0])
    print(
        "bit-identical: "
        f"{np.array_equal(repeat.radiance.values, original.radiance.values)}"
    )

    bench = record(
        "08_monte_carlo_seed_ensemble", n_calls=main_calls, wall_time_s=elapsed
    )
    print(f"\nrecorded: {bench.wall_time_s:.4f}s for {bench.n_calls} calls")


if __name__ == "__main__":
    main()
