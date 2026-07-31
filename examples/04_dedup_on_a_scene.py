"""
Deduplication: a scene has fewer atmospheres than pixels
==========================================================

A retrieval product does not carry a different atmospheric state in every
pixel. It comes out of a classifier or off a coarse grid, so neighbouring
pixels repeat, and a sweep over the image calls the engine again and again
with arguments it has already seen.

``dedup=True`` collapses identical input rows before calling anything,
computes each distinct one once, and expands the result back onto every pixel
that shares it. It never changes a value or a shape, only how many times the
engine runs to produce them.

Run with::

    pixi run -e dev python examples/04_dedup_on_a_scene.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
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
from xsweep import SweepPolicy, sweep  # noqa: E402

N_PHOTONS = 1500
CALLS = {"n": 0}


@sweep("loop(tau, ssa) -> reflectance(), transmittance()")
def layer(tau: float, ssa: float) -> dict[str, float]:
    """Reflectance and transmittance of a scattering layer, by Monte-Carlo."""
    CALLS["n"] += 1
    return mc_reflectance(tau, ssa, g=0.6, mu0=0.5, n_photons=N_PHOTONS, seed=0)


def counted() -> int:
    """Return the calls made since the last check, and reset the counter."""
    made, CALLS["n"] = CALLS["n"], 0
    return made


tau_map, ssa_map = atmosphere_scene(24, block=4)
scene = xr.Dataset({"tau": (("y", "x"), tau_map), "ssa": (("y", "x"), ssa_map)})

rows = np.stack([tau_map.ravel(), ssa_map.ravel()]).T
print(f"pixels:               {tau_map.size}")
print(f"distinct atmospheres: {len(np.unique(rows, axis=0))}")

# %%
# The plan counts the distinct rows for you
# ------------------------------------------
#
# Turning ``dedup`` on adds a line to the ``SPACE`` block: how many unique
# rows the space holds, and therefore how many calls a run will really make.
# Nothing has been computed yet.

print(layer.explain(scene, policy=SweepPolicy(dedup=True)))

# %%
# Without deduplication
# ----------------------
#
# One call per pixel, most of them recomputing an answer the engine has
# already given.

start = time.perf_counter()
plain = layer(scene)
plain_time = time.perf_counter() - start
plain_calls = counted()
print(f"{plain_calls} calls in {plain_time:.2f} s")

# %%
# With deduplication
# -------------------

start = time.perf_counter()
deduped = layer(scene, policy=SweepPolicy(dedup=True))
dedup_time = time.perf_counter() - start
dedup_calls = counted()
print(f"{dedup_calls} calls in {dedup_time:.2f} s")

print(f"\ncalls:  {plain_calls / dedup_calls:.0f}x fewer")
print(f"time:   {plain_time / dedup_time:.0f}x faster")

# %%
# Same numbers, to the last bit
# ------------------------------
#
# Deduplication is a cost decision, not a numerical one. The engine is
# stochastic and the two runs still agree exactly, because the duplicated
# pixels were copied rather than recomputed.

np.testing.assert_array_equal(plain.reflectance.values, deduped.reflectance.values)
np.testing.assert_array_equal(plain.transmittance.values, deduped.transmittance.values)
print("identical on every pixel")

# %%
# What it does not buy
# ---------------------
#
# The duplicate positions still have to be filled. In memory that is a
# gather; into a store it is a batched write pass. Deduplication removes
# engine calls, which is nearly the whole story when a call is expensive and
# almost none of it when a call is free.

run_dir = Path(tempfile.mkdtemp())
try:
    start = time.perf_counter()
    persisted = layer(
        scene, policy=SweepPolicy(dedup=True, store=run_dir / "scene.zarr")
    )
    print(f"{counted()} calls, persisted, in {time.perf_counter() - start:.2f} s")
    np.testing.assert_array_equal(
        plain.reflectance.values, persisted.reflectance.values
    )
    print("still identical once written to a store")
finally:
    shutil.rmtree(run_dir)

# %%
# The scaling that matters
# -------------------------
#
# The number of distinct rows is a property of the scene, not of its size.
# Doubling the image doubles the pixels and leaves the call count where it
# was, so the saving grows with the image.

print(f"{'pixels':>8}  {'distinct':>9}  {'calls saved':>12}")
for size in (24, 48, 96, 192):
    t_map, s_map = atmosphere_scene(size, block=size // 6)
    unique = len(np.unique(np.stack([t_map.ravel(), s_map.ravel()]).T, axis=0))
    print(f"{t_map.size:>8}  {unique:>9}  {t_map.size - unique:>12}")

# %%
# The result is still an image, with the pixels the duplicates were expanded
# onto in their proper places.

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
for ax, (data, title) in zip(
    axes,
    ((scene.tau, r"input $\tau$"), (deduped.reflectance, "reflectance, deduplicated")),
    strict=True,
):
    image = ax.imshow(data, origin="upper", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax, shrink=0.85)
fig.tight_layout()
plt.show()
