"""Sweep-space validation and loop-point enumeration.

Semantics come from xarray, not from a second description: variables sharing
a dim vary together (zip), variables on distinct dims multiply (Cartesian
product), and the sweep space is the union of the loop variables' dims.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import xarray as xr

from .contract import Contract
from .errors import SpaceError

__all__ = ["LoopAxis", "LoopGrid", "build_grid", "validate_space"]

#: dtype kinds a sweep coordinate may take. Objects break fingerprinting,
#: store serialisation and unique-row extraction alike, so they are refused
#: with a message pointing at the label idiom.
_PRIMITIVE_KINDS = frozenset("fiubSUM")


@dataclass(frozen=True)
class LoopAxis:
    """One axis of the result, with what produced it.

    Parameters
    ----------
    name
        Dim name in the space.
    size
        Number of positions along it.
    carriers
        Loop variables that carry this dim. More than one means those
        variables are zipped along it.
    """

    name: str
    size: int
    carriers: tuple[str, ...]

    @property
    def is_zipped(self) -> bool:
        """Return whether several loop variables share this axis."""
        return len(self.carriers) > 1


@dataclass(frozen=True)
class LoopGrid:
    """The enumerable grid of loop points.

    Values are read on demand through :meth:`values_at`; the full Cartesian
    table is never materialised.
    """

    axes: tuple[LoopAxis, ...]
    variables: tuple[str, ...]
    _arrays: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]
    _var_dims: dict[str, tuple[str, ...]]

    @property
    def dims(self) -> tuple[str, ...]:
        """Return the loop dim names, in result order."""
        return tuple(a.name for a in self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the loop grid shape."""
        return tuple(a.size for a in self.axes)

    @property
    def n_points(self) -> int:
        """Return the number of loop points."""
        return int(np.prod(self.shape, dtype=np.int64)) if self.axes else 1

    def values_at(self, index: tuple[int, ...]) -> dict[str, object]:
        """Return the loop values at one grid position.

        Parameters
        ----------
        index
            Position along each loop dim, aligned with :attr:`dims`.

        Returns
        -------
        dict
            Native Python values, one per loop variable.
        """
        out: dict[str, object] = {}
        for name in self.variables:
            var_dims = self._var_dims[name]
            sub = tuple(index[self.dims.index(d)] for d in var_dims)
            value = self._arrays[name][sub] if sub else self._arrays[name][()]
            out[name] = value.item()
        return out

    def indices(self) -> list[tuple[int, ...]]:
        """Return every grid position, in C order."""
        if not self.axes:
            return [()]
        return [tuple(int(i) for i in idx) for idx in np.ndindex(*self.shape)]


def validate_space(space: xr.Dataset, contract: Contract) -> None:
    """Check a space can serve a contract.

    Parameters
    ----------
    space
        The Dataset describing the whole sweep.
    contract
        The contract whose inputs must be present and usable.

    Raises
    ------
    SpaceError
        If an input is missing or carries a non-primitive dtype.

    Notes
    -----
    Alignment on shared dims is exact by construction here: a Dataset holds
    one coordinate per dim, so two variables sharing a dim necessarily share
    its labels. The case the specification worries about, near-identical
    coordinates silently inner-joined, arises while BUILDING the space, hence
    upstream of this call. The equivalent guard on the sweeper's own side is
    the store signature check performed on resume.
    """
    missing = [name for name in contract.inputs if name not in space.variables]
    if missing:
        available = ", ".join(sorted(map(str, space.variables))) or "none"
        raise SpaceError(
            f"space is missing {missing!r}, required by the contract; "
            f"available variables: {available}"
        )

    for name in contract.inputs:
        var = space[name]
        if var.dtype.kind not in _PRIMITIVE_KINDS:
            raise SpaceError(
                f"variable {name!r} has dtype {var.dtype!r}; sweep values must "
                "be primitives (float, int, str, bool, datetime64). For "
                "object-valued parameters, sweep a label string and pass the "
                "mapping as a static exposing __cache_token__"
            )


def build_grid(space: xr.Dataset, contract: Contract) -> LoopGrid:
    """Enumerate the loop grid implied by a contract and a space.

    Parameters
    ----------
    space
        The validated sweep space.
    contract
        The contract naming the loop variables.

    Returns
    -------
    LoopGrid
        The grid, from which points are read on demand.
    """
    names = tuple(v.name for v in contract.loop)

    dims: list[str] = []
    carriers: dict[str, list[str]] = {}
    for name in names:
        for dim in map(str, space[name].dims):
            if dim not in carriers:
                dims.append(dim)
                carriers[dim] = []
            carriers[dim].append(name)

    arrays: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]] = {}
    var_dims: dict[str, tuple[str, ...]] = {}
    for name in names:
        var = space[name]
        arrays[name] = _materialise(var, name)
        var_dims[name] = tuple(map(str, var.dims))

    axes = tuple(
        LoopAxis(dim, int(space.sizes[dim]), tuple(carriers[dim])) for dim in dims
    )
    return LoopGrid(axes, names, arrays, var_dims)


def _materialise(
    var: xr.DataArray, name: str
) -> np.ndarray[tuple[int, ...], np.dtype[np.generic]]:
    """Return a loop variable's values in memory, warning if it was lazy.

    In v0 loop variables are assumed to fit in memory. A lazily-backed one is
    computed rather than streamed, but never silently: uncontrolled
    materialisation is exactly the surprise the warning exists to prevent.
    """
    if not isinstance(var.variable.data, np.ndarray):
        warnings.warn(
            f"loop variable {name!r} is lazily backed and is being computed "
            "into memory; v0 assumes loop variables fit in memory",
            stacklevel=3,
        )
        return np.asarray(var.compute().values)
    return np.asarray(var.values)
