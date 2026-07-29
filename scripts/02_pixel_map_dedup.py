"""Sweep a per-pixel function over a map, and collapse the duplicates.

Run with::

    pixi run -e dev python scripts/02_pixel_map_dedup.py

Two things are worth watching. Variables sharing dims are zipped, so a
1000-pixel map is 1000 points and not a million. And deduplication divides
the CALLS, which is what costs, while the result stays bit-identical.
"""

from __future__ import annotations

import time

import numpy as np
import xarray as xr

from xsweep import SweepPolicy, sweep

CALLS = {"n": 0}


@sweep("loop(aot, rh) -> rho()", version="1")
def rho_atm(aot: float, rh: float) -> float:
    """Stand in for a per-pixel engine call."""
    CALLS["n"] += 1
    time.sleep(0.001)
    return float(aot * 100.0 + rh)


def main() -> None:
    """Compare the same map swept with and without deduplication."""
    rng = np.random.default_rng(0)
    size = 40
    space = xr.Dataset(
        {
            "aot": (("y", "x"), rng.choice([0.05, 0.1, 0.2, 0.3], (size, size))),
            "rh": (("y", "x"), rng.choice([20.0, 40.0, 60.0, 80.0], (size, size))),
        }
    )

    print("=== The plan says what deduplication buys, before running ===\n")
    print(rho_atm.explain(space, policy=SweepPolicy(dedup=True)))

    print("\n=== Without deduplication ===")
    CALLS["n"] = 0
    started = time.monotonic()
    plain = rho_atm(space)
    plain_time = time.monotonic() - started
    plain_calls = CALLS["n"]
    print(f"{plain_calls} calls in {plain_time:.2f} s")

    print("\n=== With deduplication ===")
    CALLS["n"] = 0
    started = time.monotonic()
    deduped = rho_atm(space, policy=SweepPolicy(dedup=True))
    dedup_time = time.monotonic() - started
    print(f"{CALLS['n']} calls in {dedup_time:.2f} s")

    print(f"\ncalls divided by {plain_calls / max(CALLS['n'], 1):.0f}")
    print(f"wall time divided by {plain_time / dedup_time:.1f}")
    print(f"bit-identical: {np.array_equal(plain.rho.values, deduped.rho.values)}")
    print(f"same dims: {plain.rho.dims} == {deduped.rho.dims}")
    print(
        "\nThe two ratios differ because deduplication removes calls, not "
        "writes:\nevery pixel still has to be filled. See "
        "docs/implementation-findings.md."
    )


if __name__ == "__main__":
    main()
