"""A synthetic classified scene, shared by the pixel-map examples.

Real satellite imagery does not carry a different atmospheric state in every
pixel. Retrieval products come out of a classifier or off a coarse grid, so
neighbouring pixels repeat: open water, one aerosol regime over a city, a
different one over farmland. That repetition is what deduplication exists to
exploit, and it is the only reason this scene is built from a small number of
classes rather than from smooth noise.

Nothing here is measured data. It is a tiled pattern chosen to have a known
number of distinct rows, so that the examples can state exactly how much work
deduplication removes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["atmosphere_scene"]

# (optical thickness, single-scattering albedo) per surface class.
_CLASSES = (
    (0.08, 0.99),  # clear maritime air
    (0.15, 0.97),  # light continental haze
    (0.30, 0.95),  # moderate continental aerosol
    (0.55, 0.92),  # urban plume
    (0.90, 0.88),  # heavy smoke
    (1.40, 0.85),  # dense smoke core
)


def atmosphere_scene(size: int, block: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return an optical thickness map and an albedo map of shape ``(size, size)``.

    The scene is tiled in blocks of ``block`` pixels, cycling through a fixed
    set of classes, so the number of distinct ``(tau, ssa)`` rows never
    exceeds the number of classes however large the image gets.

    Parameters
    ----------
    size
        Side of the square image, in pixels.
    block
        Side of a uniform block, in pixels.

    Returns
    -------
    tuple of ndarray
        The optical thickness map and the single-scattering albedo map.
    """
    rows = np.arange(size)[:, None] // block
    cols = np.arange(size)[None, :] // block
    class_id = (rows + cols) % len(_CLASSES)

    tau = np.array([c[0] for c in _CLASSES])[class_id]
    ssa = np.array([c[1] for c in _CLASSES])[class_id]
    return tau, ssa
