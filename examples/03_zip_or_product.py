"""
Zip or product: the shape of the sweep comes from your data
=============================================================

The contract never says how many points there are, or whether two variables
vary together. That is already written in the dims of the ``Dataset`` you
pass:

- variables sharing a dim **vary together** (a zip): one point per position;
- variables on distinct dims **multiply** (a Cartesian product).

One contract, ``loop(tau, ssa) -> reflectance(), transmittance()``, serves a
parameter study and a satellite scene without a character changing. This page
never calls the engine: it only asks ``explain`` what each space would cost,
which is the cheapest place to catch the mistake this design makes possible.

Run with::

    pixi run -e dev python examples/03_zip_or_product.py
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

from _scene import atmosphere_scene  # noqa: E402
from _solvers import mc_reflectance  # noqa: E402
from xsweep import sweep  # noqa: E402

N_PHOTONS = 1500
CALLS = {"n": 0}


@sweep("loop(tau, ssa) -> reflectance(), transmittance()")
def layer(tau: float, ssa: float) -> dict[str, float]:
    """Reflectance and transmittance of a scattering layer, by Monte-Carlo."""
    CALLS["n"] += 1
    return mc_reflectance(tau, ssa, g=0.6, mu0=0.5, n_photons=N_PHOTONS, seed=0)


# %%
# A parameter study: distinct dims, so a product
# -----------------------------------------------
#
# This is the space from page 01. ``tau`` is on dim ``tau`` and ``ssa`` on
# dim ``ssa``, so every combination is wanted: 12 x 5 = 60 points.

study = xr.Dataset(
    {
        "tau": ("tau", np.linspace(0.1, 3.0, 12)),
        "ssa": ("ssa", np.array([0.85, 0.9, 0.95, 0.99, 1.0])),
    }
)
print(layer.explain(study))

# %%
# A satellite scene: shared dims, so a zip
# -----------------------------------------
#
# Now both variables are 2-D maps over the same ``(y, x)`` dims. They vary
# together, one point per pixel, and 64 x 64 pixels is 4096 points rather
# than 4096 squared.

tau_map, ssa_map = atmosphere_scene(64)
scene = xr.Dataset({"tau": (("y", "x"), tau_map), "ssa": (("y", "x"), ssa_map)})
print(layer.explain(scene))

# %%
# The same contract read both spaces. The plan's ``SPACE`` block is where the
# axes and their origin show up, and where the point count is decided.

# %%
# The mistake this makes possible
# --------------------------------
#
# Two variables that were meant to vary together, declared on different dims
# by accident, silently become every combination of themselves. Nothing about
# the code looks wrong; the point count is the only symptom.

flat_tau = tau_map.ravel()[:200]
flat_ssa = ssa_map.ravel()[:200]

correct = xr.Dataset({"tau": ("pixel", flat_tau), "ssa": ("pixel", flat_ssa)})
by_accident = xr.Dataset({"tau": ("pixel", flat_tau), "ssa": ("point", flat_ssa)})

print(
    f"same dim,      zipped:   {correct.sizes} -> {layer.explain(correct).n_points} points"
)
print(
    f"different dims, product: {by_accident.sizes} -> "
    f"{layer.explain(by_accident).n_points} points"
)
print(
    f"\nratio: {layer.explain(by_accident).n_points // layer.explain(correct).n_points}x more work"
)

# %%
# With an engine costing milliseconds that is an annoyance. At seconds per
# call it is the difference between an afternoon and a fortnight, which is
# why ``explain`` exists and why it makes no calls.

per_call = 4.0
for name, space in (("zipped", correct), ("product", by_accident)):
    n = layer.explain(space).n_points
    print(f"{name:>8}: {n:>6} points x {per_call:.0f} s = {n * per_call / 3600:8.2f} h")

# %%
# Running the zipped scene
# -------------------------
#
# A small corner of the scene, to show the result keeps the map's own dims:
# the output is an image, not a list of points to reshape by hand.

corner = scene.isel(y=slice(0, 16), x=slice(0, 16))
result = layer(corner)

print(f"calls made: {CALLS['n']}")
print(result)

# %%
# 256 pixels, 256 calls. The scene is built from six atmosphere classes and
# this corner holds three of them, so most of those calls recomputed a point
# the engine had already produced. That is the next page.

distinct = np.unique(
    np.stack([corner.tau.values.ravel(), corner.ssa.values.ravel()]).T, axis=0
)
print(f"pixels: {corner.tau.size}, distinct atmospheres: {len(distinct)}")

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
for ax, (data, title) in zip(
    axes,
    ((corner.tau, r"input $\tau$"), (result.reflectance, "output reflectance")),
    strict=True,
):
    image = ax.imshow(data, origin="upper", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax, shrink=0.85)
fig.tight_layout()
plt.show()
