"""
Replication: measuring the noise of a stochastic engine
=========================================================

Every Monte-Carlo number on the preceding pages carries an error bar nobody
has drawn yet. Getting one means running the same configuration several times
and looking at the spread, which sounds like it needs a "repeat" axis.

It does not, and the reason is worth understanding: a contract loops over
**variables**, not over dims, so a bare ``rep`` dim has nothing to hand the
function. Worse, if it somehow worked it would be a trap, because twenty
identical calls are exactly what the cache and the deduplicator exist to
collapse, and the standard deviation would come back as zero.

The idiom is a carrier variable that makes each repetition genuinely
different: the seed.

Run with::

    pixi run -e dev python examples/10_replication_and_the_seed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import xsweep

# sphinx-gallery executes examples without `__file__` set, so the gallery
# directory is located from the already-installed xsweep package instead.
sys.path.insert(0, str(Path(xsweep.__file__).resolve().parents[2] / "examples"))

from _solvers import mc_reflectance  # noqa: E402
from xsweep import SweepPolicy, sweep  # noqa: E402
from xsweep.errors import SpaceError  # noqa: E402

# %%
# The spelling that does not work
# --------------------------------
#
# ``rep`` is a dim of the space, not a variable in it, so there is nothing to
# pass to the parameter the contract named.


@sweep("loop(tau, rep) -> reflectance()")
def wrong(tau: float, rep: int) -> float:
    """This contract asks for a variable the space does not have."""
    raise AssertionError("unreachable")


bare_dim = xr.Dataset({"tau": ("tau", [0.5, 1.0])}).expand_dims(rep=20)
try:
    wrong.explain(bare_dim)
except SpaceError as error:
    print(f"SpaceError: {error}")

# %%
# The carrier variable
# ---------------------
#
# ``seed`` is an ordinary swept variable that happens to live on a dim called
# ``rep``. Each repetition is now a distinct point: individually
# reproducible, individually cacheable, and resumable like any other.

CALLS = {"n": 0}


@sweep("loop(tau, seed) -> reflectance()")
def replicated(tau: float, seed: int) -> float:
    """Monte-Carlo reflectance, one independent realisation per seed."""
    CALLS["n"] += 1
    return mc_reflectance(tau, 0.95, g=0.6, mu0=0.5, n_photons=1500, seed=int(seed))[
        "reflectance"
    ]


space = xr.Dataset(
    {
        "tau": ("tau", np.linspace(0.2, 3.0, 8)),
        "seed": ("rep", np.arange(24)),
    }
)

print(replicated.explain(space))

# %%
# The result carries the ``rep`` dim the carrier put it on, so the ensemble
# statistics are one xarray reduction away.

result = replicated(space)
mean = result.reflectance.mean("rep")
spread = result.reflectance.std("rep")

print(f"{CALLS['n']} calls")
print(f"\n{'tau':>5} {'mean':>8} {'std':>8} {'rel':>7}")
for tau in space.tau.values:
    m = float(mean.sel(tau=tau))
    s = float(spread.sel(tau=tau))
    print(f"{tau:5.2f} {m:8.4f} {s:8.4f} {s / m:6.1%}")

# %%
# Why deduplication would have eaten it
# --------------------------------------
#
# The point of the seed is that the rows really are distinct. Turn
# deduplication on and the call count does not move, because there is nothing
# to collapse. Had the repetitions been identical, it would have collapsed
# them to one and the spread above would have been exactly zero.

CALLS["n"] = 0
deduped = replicated(space, policy=SweepPolicy(dedup=True))
print(f"dedup=True: {CALLS['n']} calls, {space.tau.size * space.rep.size} points")
np.testing.assert_array_equal(result.reflectance.values, deduped.reflectance.values)

identical = xr.Dataset(
    {"tau": ("tau", np.linspace(0.2, 3.0, 8)), "seed": ("rep", np.zeros(24, dtype=int))}
)
print(
    f"same seed everywhere: "
    f"{replicated.explain(identical, policy=SweepPolicy(dedup=True)).n_calls} calls "
    f"for {identical.tau.size * identical.rep.size} points, "
    f"and a spread of exactly zero"
)

# %%
# The error bar the rest of the gallery was missing
# --------------------------------------------------
#
# This is the number page 08 needed before it could claim anything about
# interpolation error: at 1500 photons the engine's own one-sigma noise is
# around 0.01 in reflectance, roughly 3% of the value, and it shrinks as the
# square root of the photon count and no faster. Quadrupling the photons
# halves the bar; there is no cheaper way to buy a digit.

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.errorbar(
    space.tau, mean, yerr=spread, marker="o", capsize=3, label="mean $\\pm$ 1$\\sigma$"
)
ax.fill_between(space.tau, mean - spread, mean + spread, alpha=0.2)
ax.set_xlabel(r"optical thickness $\tau$")
ax.set_ylabel("reflectance")
ax.set_title("24 independent realisations per point, 1500 photons each")
ax.legend(fontsize="small")
fig.tight_layout()
plt.show()
