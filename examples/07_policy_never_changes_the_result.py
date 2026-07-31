"""
Policy: the knobs that change what a run costs, never what it returns
=======================================================================

Three things are decided at three different moments, and keeping them apart
is most of what makes a sweep readable a year later:

- the **contract** is physics, fixed when the function is written;
- the **policy** is run configuration, chosen at the decorator, the instance,
  or the call, in that order of increasing precedence;
- the **data** is the space and the statics, given at the call.

Nothing in a policy is allowed to move a value. Deduplication, batch sizes,
executors, the store: they decide what a run costs and how it survives, and
the library's test suite enforces bit-identical results across combinations
of them as a release gate. This page checks it in the open, on an engine
whose answers are made of random numbers.

Run with::

    pixi run -e dev python examples/07_policy_never_changes_the_result.py
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

from _scene import atmosphere_scene  # noqa: E402
from _solvers import mc_reflectance  # noqa: E402
from xsweep import SweepPolicy, sweep  # noqa: E402

N_PHOTONS = 3000


@sweep("loop(tau, ssa) -> reflectance(), transmittance()")
def layer(tau: float, ssa: float) -> dict[str, float]:
    """Reflectance and transmittance of a scattering layer, by Monte-Carlo."""
    return mc_reflectance(tau, ssa, g=0.6, mu0=0.5, n_photons=N_PHOTONS, seed=0)


tau_map, ssa_map = atmosphere_scene(12, block=3)
scene = xr.Dataset({"tau": (("y", "x"), tau_map), "ssa": (("y", "x"), ssa_map)})

print(f"points: {scene.tau.size}")
print(f"cost:   about {N_PHOTONS} photons per call")

# %%
# Every policy, one at a time
# ----------------------------
#
# Same function, same space, five ways of running it. The wall-clock times
# differ by an order of magnitude; the arrays do not differ at all.

run_dir = Path(tempfile.mkdtemp())
policies = {
    "plain": SweepPolicy(),
    "dedup": SweepPolicy(dedup=True),
    "store": SweepPolicy(store=run_dir / "plain.zarr"),
    "dedup + store": SweepPolicy(dedup=True, store=run_dir / "dedup.zarr"),
    "processes": SweepPolicy(executor="process", max_workers=4),
}

results = {}
elapsed = {}
for name, policy in policies.items():
    start = time.perf_counter()
    results[name] = layer(scene, policy=policy)
    elapsed[name] = time.perf_counter() - start
    print(f"{name:>14}: {elapsed[name]:6.2f} s")

# %%
# The check
# ----------
#
# ``assert_array_equal``, not ``allclose``. A Monte-Carlo engine makes this a
# real test: any policy that changed which random draws a point saw, or that
# rounded a value on its way through a store, would show up immediately.

reference = results["plain"]
for name, result in results.items():
    for variable in ("reflectance", "transmittance"):
        np.testing.assert_array_equal(
            reference[variable].values, result[variable].values
        )
    print(f"{name:>14}: identical")

# %%
# Where the time went
# --------------------
#
# Deduplication removes calls, so it wins whenever the scene repeats.
# Processes divide the calls that remain across cores, and cost a fixed
# amount to start, so they pay off once a call is expensive enough. Neither
# is a decision about the answer.

baseline = elapsed["plain"]
for name in ("dedup", "processes", "dedup + store"):
    print(f"{name:>14}: {baseline / elapsed[name]:5.1f}x faster than plain")

# %%
# Where a policy can be set
# --------------------------
#
# A default on the decorator, an override on the call. The call wins. This
# lets a function ship with the configuration it usually wants, without
# freezing it for the one run that needs something else.


@sweep("loop(tau, ssa) -> reflectance(), transmittance()", dedup=True)
def deduped_by_default(tau: float, ssa: float) -> dict[str, float]:
    """Same engine, with deduplication as its shipped default."""
    return mc_reflectance(tau, ssa, g=0.6, mu0=0.5, n_photons=N_PHOTONS, seed=0)


print(f"decorator default: {deduped_by_default.explain(scene).n_calls} calls")
print(
    f"overridden at call: "
    f"{deduped_by_default.explain(scene, policy=SweepPolicy(dedup=False)).n_calls} calls"
)

shutil.rmtree(run_dir)

# %%
# The one thing a policy does change, besides cost, is what survives: a store
# is the difference between a run you can resume and one you cannot. Page 09
# is what happens when a run does not finish.
