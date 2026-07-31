"""
When points fail: record, retry, resume
=========================================

An engine that runs for hours will eventually not run for hours. A node dies,
a licence server hiccups, one configuration hits a case the solver cannot
handle. The question is not how to avoid that but what the run does about it.

By default xsweep records the failure and carries on: the point gets ``nan``
and a ``failed`` status, the other 4999 points still get computed, and the
store remembers which is which. Then you fix the cause and run again, and
only the failures are retried.

Run with::

    pixi run -e dev python examples/09_when_points_fail.py
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

# sphinx-gallery executes examples without `__file__` set, so the gallery
# directory is located from the already-installed xsweep package instead.
sys.path.insert(0, str(Path(xsweep.__file__).resolve().parents[2] / "examples"))

from _solvers import mc_reflectance  # noqa: E402
from xsweep import SweepPolicy, sweep  # noqa: E402
from xsweep.errors import PointFailed  # noqa: E402

N_PHOTONS = 1500

# The engine is broken to start with, and gets fixed halfway down the page.
STATE = {"calls": 0, "failures": 0, "broken": True, "flaky_until": 0}


@sweep("loop(tau, ssa) -> reflectance()")
def layer(tau: float, ssa: float) -> float:
    """Monte-Carlo reflectance, with an engine that is not always available."""
    STATE["calls"] += 1
    if STATE["broken"] and ssa > 0.97:
        STATE["failures"] += 1
        raise RuntimeError(f"solver did not converge at ssa={ssa:.3f}")
    return mc_reflectance(tau, ssa, g=0.6, mu0=0.5, n_photons=N_PHOTONS, seed=0)[
        "reflectance"
    ]


space = xr.Dataset(
    {
        "tau": ("tau", np.linspace(0.1, 3.0, 20)),
        "ssa": ("ssa", np.linspace(0.85, 1.0, 10)),
    }
)

run_dir = Path(tempfile.mkdtemp())
store = run_dir / "layer.zarr"
policy = SweepPolicy(store=store)


def report(result: xr.Dataset) -> None:
    """Print the count of each status code in a result."""
    labels = result.status.attrs["labels"]
    codes, counts = np.unique(result.status.values, return_counts=True)
    for code, count in zip(codes, counts, strict=True):
        print(f"  {labels[str(code)]:>8}: {count}")


# %%
# The first run, with a broken engine
# ------------------------------------
#
# Two of the ten albedo columns raise. The run does not stop, and it does not
# pretend they worked either.

start = time.perf_counter()
first = layer(space, policy=policy)
print(
    f"{STATE['calls']} calls, {STATE['failures']} raised, "
    f"in {time.perf_counter() - start:.2f} s"
)
report(first)

failed_mask = np.isnan(first.reflectance.values)
print(f"\nnan where it failed: {failed_mask.sum()} of {first.reflectance.size}")
print(
    f"values elsewhere:    {np.nanmin(first.reflectance.values):.4f} "
    f"to {np.nanmax(first.reflectance.values):.4f}"
)

# %%
# The plan knows what is still missing
# -------------------------------------
#
# ``failed`` is not ``ok``, so those points are still to compute. Rerunning
# without touching anything would retry them, and fail again.

print(layer.explain(space, policy=policy))

# %%
# Fixing the engine and resuming
# -------------------------------
#
# Nothing about the sweep changes: same space, same store, same call. Only
# the failures are recomputed, and the 160 points that already succeeded are
# read back rather than run again.

STATE["broken"] = False
STATE["calls"] = 0

start = time.perf_counter()
second = layer(space, policy=policy)
print(f"{STATE['calls']} calls in {time.perf_counter() - start:.2f} s")
report(second)

assert not np.isnan(second.reflectance.values).any()
np.testing.assert_array_equal(
    first.reflectance.values[~failed_mask], second.reflectance.values[~failed_mask]
)
print("\nthe points that had succeeded were not recomputed")

# %%
# Retries, for a failure that is not reproducible
# ------------------------------------------------
#
# A converging solver that fails is a bug to fix. A network share that times
# out is not: it wants another attempt, not a code change. ``retries`` gives
# the call one, before the failure is recorded.

STATE.update(calls=0, failures=0, flaky_until=3)


@sweep("loop(tau) -> reflectance()")
def flaky(tau: float) -> float:
    """An engine that fails the first few times it is ever called."""
    STATE["calls"] += 1
    if STATE["flaky_until"] > 0:
        STATE["flaky_until"] -= 1
        raise RuntimeError("transient: connection reset")
    return mc_reflectance(tau, 0.95, g=0.6, mu0=0.5, n_photons=N_PHOTONS, seed=0)[
        "reflectance"
    ]


small = xr.Dataset({"tau": ("tau", np.linspace(0.2, 2.0, 5))})

no_retry = flaky(small)
print(f"retries=0: {int(np.isnan(no_retry.reflectance.values).sum())} points lost")

STATE["flaky_until"] = 3
with_retry = flaky(small, policy=SweepPolicy(retries=3))
print(f"retries=3: {int(np.isnan(with_retry.reflectance.values).sum())} points lost")

# %%
# Failing fast instead
# ---------------------
#
# Recording failures is right for a long unattended run. During development
# it hides the traceback you actually want, so ``on_error="raise"`` stops at
# the first one and gives it to you with the point that caused it.

STATE.update(broken=True, calls=0, failures=0)

try:
    layer(space, policy=SweepPolicy(on_error="raise", store=run_dir / "strict.zarr"))
except PointFailed as error:
    print(f"PointFailed after {STATE['calls']} calls")
    print(f"  {error}")

# %%
# Both behaviours leave the same artefact behind: whatever had been computed
# before the stop is in the store, so the fail-fast run is resumable too.

shutil.rmtree(run_dir)
