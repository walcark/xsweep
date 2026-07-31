"""
SweepModule: state built once, reused by every call
=====================================================

The standard way to make an expensive solver affordable is to run it on a
coarse grid once and interpolate afterwards. That gives the sweep a piece of
state: a lookup table that costs a lot to build, nothing to use, and must not
be rebuilt per point.

A closure cannot hold it, because the process executor has to pickle the
callable. A module global can, badly. ``SweepModule`` gives it a home,
modelled on the ``torch.nn.Module`` idiom: ``__init__`` builds the state,
``forward`` is the physics, ``__call__`` does the sweep.

Run with::

    pixi run -e dev python examples/08_module_with_state.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import xsweep

# sphinx-gallery executes examples without `__file__` set, so the gallery
# directory is located from the already-installed xsweep package instead.
sys.path.insert(0, str(Path(xsweep.__file__).resolve().parents[2] / "examples"))

from _solvers import mc_reflectance  # noqa: E402
from xsweep import SweepModule, SweepPolicy  # noqa: E402

N_PHOTONS = 4000

# %%
# The module
# -----------
#
# ``contract`` is a class attribute, so a malformed one raises when the
# module is imported rather than after twenty minutes of engine time.
# ``forward`` never mentions the sweep and can be called on its own.


class LayerLUT(SweepModule):
    """Monte-Carlo reflectance, tabulated once and interpolated afterwards."""

    contract = "loop(tau, ssa) -> reflectance()"

    def __init__(
        self,
        n_tau: int = 8,
        n_ssa: int = 6,
        policy: SweepPolicy | None = None,
    ) -> None:
        super().__init__(policy)
        self.tau_nodes = np.linspace(0.05, 3.0, n_tau)
        self.ssa_nodes = np.linspace(0.80, 1.0, n_ssa)
        self.table = np.array(
            [
                [
                    mc_reflectance(
                        float(tau),
                        float(ssa),
                        g=0.6,
                        mu0=0.5,
                        n_photons=N_PHOTONS,
                        seed=0,
                    )["reflectance"]
                    for ssa in self.ssa_nodes
                ]
                for tau in self.tau_nodes
            ]
        )

    def forward(self, tau: float, ssa: float) -> float:
        """Bilinearly interpolate the table at one point."""
        i = int(
            np.clip(
                np.searchsorted(self.tau_nodes, tau) - 1, 0, len(self.tau_nodes) - 2
            )
        )
        j = int(
            np.clip(
                np.searchsorted(self.ssa_nodes, ssa) - 1, 0, len(self.ssa_nodes) - 2
            )
        )
        u = (tau - self.tau_nodes[i]) / (self.tau_nodes[i + 1] - self.tau_nodes[i])
        v = (ssa - self.ssa_nodes[j]) / (self.ssa_nodes[j + 1] - self.ssa_nodes[j])
        return float(
            (1 - u) * (1 - v) * self.table[i, j]
            + u * (1 - v) * self.table[i + 1, j]
            + (1 - u) * v * self.table[i, j + 1]
            + u * v * self.table[i + 1, j + 1]
        )


# %%
# Building the state
# -------------------
#
# 48 Monte-Carlo runs, paid once. Every call afterwards is an interpolation.

start = time.perf_counter()
module = LayerLUT()
build_time = time.perf_counter() - start

print(f"table: {module.table.shape} built in {build_time:.2f} s")
print(f"that is {module.table.size} Monte-Carlo runs at {N_PHOTONS} photons")

# %%
# forward is testable without a sweep
# ------------------------------------
#
# No space, no policy, no store: just the physics, with the arguments the
# contract says it takes.

print(f"forward(1.0, 0.95) = {module.forward(1.0, 0.95):.4f}")

# %%
# The sweep
# ----------
#
# A grid far finer than the table, which is the entire point of building the
# table in the first place.

space = xr.Dataset(
    {
        "tau": ("tau", np.linspace(0.1, 2.9, 60)),
        "ssa": ("ssa", np.linspace(0.82, 0.99, 30)),
    }
)

print(module.explain(space))

# %%

start = time.perf_counter()
result = module(space)
print(
    f"{space.tau.size * space.ssa.size} points in {time.perf_counter() - start:.3f} s"
)

# %%
# What the table costs in accuracy
# ---------------------------------
#
# Interpolating is not free of consequence, and a page that showed only the
# speed-up would be selling something. Against direct Monte-Carlo at points
# the table does not contain, the error runs from about 0.003 to 0.015 in
# reflectance, and it is consistently positive: reflectance is convex in
# both directions here, so a linear interpolation sits above it. The largest
# errors are well past the Monte-Carlo noise at this photon count, which
# means the table spacing is the limit, not the number of photons. Refining
# the table is the fix; more photons would buy nothing.

checks = [(0.4, 0.87), (1.2, 0.93), (2.2, 0.96), (2.7, 0.985)]
print(f"{'tau':>5} {'ssa':>6} {'table':>8} {'direct':>8} {'error':>8}")
for tau, ssa in checks:
    direct = mc_reflectance(tau, ssa, g=0.6, mu0=0.5, n_photons=N_PHOTONS, seed=1)
    interpolated = module.forward(tau, ssa)
    print(
        f"{tau:5.2f} {ssa:6.3f} {interpolated:8.4f} "
        f"{direct['reflectance']:8.4f} {interpolated - direct['reflectance']:+8.4f}"
    )

# %%
# The instance travels into worker processes
# -------------------------------------------
#
# This is the combination the class facade is really for: state built once,
# then handed to several workers. The table goes with the instance, and the
# answers do not change.

parallel = LayerLUT(policy=SweepPolicy(executor="process", max_workers=4))
parallel.table = module.table  # reuse the table rather than rebuild it here

np.testing.assert_array_equal(
    result.reflectance.values, parallel(space).reflectance.values
)
print("four workers, identical result")

# %%
# A bare ``@sweep`` function would have been simpler here, and is the right
# choice whenever there is no per-instance setup. Reach for the class when
# ``__init__`` earns its keep, which is exactly when something expensive must
# be built before the first point and kept after it.

fig, ax = plt.subplots(figsize=(6.5, 4.0))
image = ax.pcolormesh(
    result.ssa, result.tau, result.reflectance, cmap="viridis", shading="auto"
)
ax.plot(
    *np.meshgrid(module.ssa_nodes, module.tau_nodes),
    "w.",
    markersize=3,
    label="table nodes",
)
ax.set_xlabel(r"single-scattering albedo $\omega$")
ax.set_ylabel(r"optical thickness $\tau$")
ax.set_title("1800 interpolated points over a 48-node table")
ax.legend(fontsize="small", loc="upper left")
fig.colorbar(image, ax=ax, label="reflectance")
fig.tight_layout()
plt.show()
