"""Deduplication: fewer calls, identical results.

A map sweep usually repeats itself. Collapsing identical parameter rows turns
O(pixels) engine calls into O(unique rows), which is the difference between
a feasible and an infeasible run.

Uniqueness is computed on integer codes, one column per loop variable,
obtained by factorising each column in its own dtype. That is what makes
mixed dtypes work: floats, strings and datetimes are never compared to each
other, only their codes are.
"""

from __future__ import annotations

import numpy as np

from .space import LoopGrid

__all__ = ["unique_map"]


def unique_map(
    grid: LoopGrid, dims: tuple[str, ...] | None
) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.int64]], int]:
    """Map every loop point to its unique representative.

    Parameters
    ----------
    grid
        The enumerated loop grid.
    dims
        Dims to deduplicate over, or ``None`` for all of them. Points that
        differ along a dim outside this set stay distinct, which is what lets
        a user collapse space while keeping time fully swept.

    Returns
    -------
    tuple
        An array over the loop grid holding each point's representative
        index, and the number of unique points.
    """
    if not grid.axes:
        return np.zeros((), dtype=np.int64), 1

    columns: list[np.ndarray[tuple[int, ...], np.dtype[np.int64]]] = []

    for name in grid.variables:
        values = _broadcast(grid, name)
        _, codes = np.unique(values.ravel(), return_inverse=True)
        columns.append(codes.astype(np.int64))

    # Dims left out of the deduplication must keep points apart, so their
    # position enters the comparison as an extra column.
    target = set(grid.dims) if dims is None else set(dims)
    for axis_pos, axis in enumerate(grid.axes):
        if axis.name in target:
            continue
        spread = np.arange(axis.size)
        shaped = spread.reshape(
            tuple(-1 if i == axis_pos else 1 for i in range(len(grid.axes)))
        )
        columns.append(np.broadcast_to(shaped, grid.shape).ravel().astype(np.int64))

    matrix = np.stack(columns, axis=1)
    _, inverse = np.unique(matrix, axis=0, return_inverse=True)
    inverse = np.asarray(inverse, dtype=np.int64).reshape(grid.shape)
    return inverse, int(inverse.max()) + 1


def _broadcast(
    grid: LoopGrid, name: str
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Spread one loop variable over the whole grid.

    Coordinate labels are deliberately absent here: this works on positions,
    because aligning by label when two variables carry different labels on
    the same dim is exactly the trap that produces silent misalignment.
    """
    values = grid._arrays[name]
    var_dims = grid._var_dims[name]
    shape = tuple(
        grid.shape[grid.dims.index(d)] if d in var_dims else 1 for d in grid.dims
    )
    ordered = np.transpose(
        values, [var_dims.index(d) for d in grid.dims if d in var_dims]
    )
    return np.broadcast_to(ordered.reshape(shape), grid.shape)
