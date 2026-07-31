"""
The result is also the cache: store, status and resume
=======================================================

Point a policy at a path and the sweep gains three things at once, without a
line changing in the function or the contract: results survive the process,
a second run costs nothing, and an interrupted run picks up where it stopped.

That is one mechanism, not three. Results stream into a zarr store one region
per point, alongside a ``status`` variable saying what happened to each
(``pending``, ``ok``, ``failed``, ``skipped``). "Already done" is a fact on
disk, so ``explain`` can report what a run has left to do before it starts.

This page computes half the grid on purpose, then finishes it, which is what
an interruption looks like from the store's point of view.

Run with::

    pixi run -e dev python examples/02_store_cache_and_resume.py
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

N_PHOTONS = 1500
CALLS = {"n": 0}


@sweep("loop(tau, ssa) -> reflectance(), transmittance()")
def layer(tau: float, ssa: float) -> dict[str, float]:
    """Reflectance and transmittance of a scattering layer, by Monte-Carlo."""
    CALLS["n"] += 1
    return mc_reflectance(tau, ssa, g=0.6, mu0=0.5, n_photons=N_PHOTONS, seed=0)


space = xr.Dataset(
    {
        "tau": ("tau", np.linspace(0.1, 3.0, 12)),
        "ssa": ("ssa", np.array([0.85, 0.9, 0.95, 0.99, 1.0])),
    }
)

run_dir = Path(tempfile.mkdtemp())
store = run_dir / "layer.zarr"


def counted() -> int:
    """Return the calls made since the last check, and reset the counter."""
    made, CALLS["n"] = CALLS["n"], 0
    return made


# %%
# First pass: only the thin half of the grid
# -------------------------------------------
#
# ``skip_where`` takes a predicate on a point's loop values and excludes it
# without ever calling the engine. Here it stands in for a run that stopped
# early; it is also how you would deliberately do the cheap corner of a grid
# first.

thin_only = SweepPolicy(store=store, skip_where=lambda point: point["tau"] > 1.5)

start = time.perf_counter()
first = layer(space, policy=thin_only)
print(f"calls made: {counted()} in {time.perf_counter() - start:.2f} s")

# %%
# The status variable says what happened where
# ---------------------------------------------
#
# It rides along with the result, so "what did this run actually do" is
# answerable without re-running anything.

labels = first.status.attrs["labels"]
codes, counts = np.unique(first.status.values, return_counts=True)
for code, count in zip(codes, counts, strict=True):
    print(f"{labels[str(code)]:>8}: {count}")

print(
    f"\nreflectance where it ran:     {float(first.reflectance.isel(tau=0, ssa=0)):.4f}"
)
print(f"reflectance where it did not: {float(first.reflectance.isel(tau=-1, ssa=0))}")

# %%
# The plan now knows what is left
# --------------------------------
#
# Same space, same contract, no predicate this time. Before running, the plan
# reads the store and splits the grid into what is already there and what is
# not. This is the number to multiply by the cost of one call.

resume = SweepPolicy(store=store)
plan = layer.explain(space, policy=resume)
print(plan)

# %%
# Finishing the run
# ------------------

start = time.perf_counter()
second = layer(space, policy=resume)
print(f"calls made: {counted()} in {time.perf_counter() - start:.2f} s")
print(f"planned:    {plan.n_to_compute}")

# %%
# The points computed the first time were not recomputed, so they are
# bit-identical rather than merely close, which for a Monte-Carlo result is
# the difference between a cache and a coincidence.

kept = first.reflectance.isel(tau=slice(0, 6))
np.testing.assert_array_equal(
    kept.values, second.reflectance.isel(tau=slice(0, 6)).values
)
print("first-pass points came back untouched")

# %%
# A third run costs nothing at all
# ---------------------------------
#
# Everything is ``ok``, so there is nothing to compute. This is the property
# that makes a long sweep safe to relaunch: the cost of asking again is the
# cost of reading the store.

start = time.perf_counter()
third = layer(space, policy=resume)
print(f"calls made: {counted()} in {time.perf_counter() - start:.2f} s")
print(f"plan says:  {layer.explain(space, policy=resume).n_to_compute} to compute")

np.testing.assert_array_equal(second.reflectance.values, third.reflectance.values)
print("\nvalues unchanged across the three runs")

# %%
# What is on disk
# ----------------
#
# An ordinary zarr store, readable by anything that reads zarr, with the
# outputs and the status side by side. The cache and the result are the same
# artefact.

reopened = xr.open_zarr(store)
print(reopened)

reopened.close()
shutil.rmtree(run_dir)
