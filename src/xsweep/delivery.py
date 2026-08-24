"""What one call receives, and what it is allowed to return.

Arguments are assembled from the contract, never from the callable's
signature: introspection is what makes a typo silently drop an argument, and
the contract already names everything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import xarray as xr

from .contract import Contract
from .errors import ContractError

__all__ = [
    "GROUP_DIM",
    "assemble_args",
    "assemble_group_args",
    "normalise_return",
    "split_group_return",
]

#: Dim a batched call stacks its points along, in both directions: the
#: arguments arrive over it and the return is cut back along it.  Reserved
#: rather than configurable, so that reading a batched callable never
#: requires looking up which name this sweep happened to choose.
GROUP_DIM = "point"


def assemble_args(
    contract: Contract,
    *,
    loop_values: Mapping[str, Any],
    arrays: Mapping[str, xr.DataArray],
    statics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the keyword arguments for one call.

    Parameters
    ----------
    contract
        The contract naming every argument.
    loop_values
        Native Python values for the loop variables at this point.
    arrays
        Vec variables sliced to the current batch, and const variables whole
        except on any dim they share with an active batch and do not
        protect.
    statics
        Configuration forwarded verbatim to every call.

    Returns
    -------
    dict
        Keyword arguments, ready to splat into the wrapped callable.
    """
    args: dict[str, Any] = {}
    for loop_var in contract.loop:
        value = loop_values[loop_var.name]
        args[loop_var.name] = (
            xr.DataArray(value) if loop_var.deliver == "array" else value
        )
    for vec_var in contract.vec:
        args[vec_var.name] = arrays[vec_var.name]
    for const_var in contract.const:
        args[const_var.name] = arrays[const_var.name]
    args.update(statics)
    return args


def normalise_return(result: Any, contract: Contract) -> xr.Dataset:
    """Turn whatever the callable returned into a named Dataset.

    Parameters
    ----------
    result
        A bare array for a single-output contract, a dict or an explicit
        Dataset for any contract.
    contract
        The contract declaring the expected output names.

    Returns
    -------
    xr.Dataset
        One variable per declared output, named as declared.

    Raises
    ------
    ContractError
        If the shape of the return does not match the declaration, a dict or
        a Dataset omits a declared output, or a multi-output contract is
        handed a tuple: a tuple is matched by position only, so a swapped
        pair of return values would go undetected, which is exactly the
        failure a name-checked return exists to prevent.
    """
    names = contract.outputs

    if isinstance(result, xr.Dataset):
        missing = [n for n in names if n not in result.data_vars]
        if missing:
            got = ", ".join(map(str, result.data_vars)) or "none"
            raise ContractError(
                f"returned Dataset is missing {missing!r}; the contract "
                f"declares outputs {list(names)!r} and the Dataset holds: {got}"
            )
        return result[list(names)]

    if isinstance(result, dict):
        missing = [n for n in names if n not in result]
        if missing:
            got = ", ".join(map(str, result)) or "none"
            raise ContractError(
                f"returned dict is missing {missing!r}; the contract "
                f"declares outputs {list(names)!r} and the dict holds: {got}"
            )
        return xr.Dataset({name: _as_dataarray(result[name], name) for name in names})

    if len(names) == 1:
        if isinstance(result, tuple):
            raise ContractError(
                f"contract declares one output {names[0]!r} but the function "
                f"returned a tuple of {len(result)}"
            )
        return xr.Dataset({names[0]: _as_dataarray(result, names[0])})

    example = ", ".join(f"{n!r}: ..." for n in names)
    raise ContractError(
        f"contract declares {len(names)} outputs {list(names)!r}; return a "
        f"dict ({{{example}}}) or an xr.Dataset, not a tuple: a tuple is "
        "matched by position only, so swapping two return values would go "
        "undetected"
    )


def _as_dataarray(value: Any, name: str) -> xr.DataArray:
    """Coerce one returned value to a named DataArray."""
    array = value if isinstance(value, xr.DataArray) else xr.DataArray(value)
    return array.rename(name)


def assemble_group_args(
    contract: Contract,
    *,
    group_values: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, xr.DataArray],
    statics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the keyword arguments for one batched call.

    Batched loop variables arrive as 1-D arrays over :data:`GROUP_DIM`, one
    entry per point of the group and aligned across variables, so that
    position ``i`` of every batched argument describes the same point.
    Scalar-delivered loop variables are shared by the whole group and so
    arrive as they always do.

    Parameters
    ----------
    contract
        The contract naming every argument.
    group_values
        Loop values for each point of the group, in order.
    arrays
        Vec and const variables, as for a single call.
    statics
        Configuration forwarded verbatim to every call.

    Returns
    -------
    dict
        Keyword arguments, ready to splat into the wrapped callable.

    Raises
    ------
    ContractError
        If a scalar-delivered loop variable does not hold one value across
        the group, which would make the group ambiguous.
    """
    args: dict[str, Any] = {}
    for loop_var in contract.loop:
        values = [point[loop_var.name] for point in group_values]
        if loop_var.deliver == "batch":
            args[loop_var.name] = xr.DataArray(np.asarray(values), dims=[GROUP_DIM])
            continue
        distinct = set(values)
        if len(distinct) > 1:
            raise ContractError(
                f"loop variable {loop_var.name!r} takes {len(distinct)} "
                "values across one batched group; group only points that "
                "share it, or declare it in the batch clause too"
            )
        value = values[0]
        args[loop_var.name] = (
            xr.DataArray(value) if loop_var.deliver == "array" else value
        )
    for vec_var in contract.vec:
        args[vec_var.name] = arrays[vec_var.name]
    for const_var in contract.const:
        args[const_var.name] = arrays[const_var.name]
    args.update(statics)
    return args


def split_group_return(result: xr.Dataset, size: int) -> list[xr.Dataset]:
    """Cut one batched call's return into one Dataset per point.

    Parameters
    ----------
    result
        What the callable returned, already normalised, carrying
        :data:`GROUP_DIM` on every output.
    size
        Number of points the group held.

    Returns
    -------
    list[xr.Dataset]
        One Dataset per point, in group order, with the group dim dropped
        so that each looks exactly like an unbatched call's return.

    Raises
    ------
    ContractError
        If an output lacks the group dim or disagrees with the group size.
    """
    for name, var in result.data_vars.items():
        if GROUP_DIM not in var.dims:
            dims = ", ".join(map(str, var.dims)) or "none"
            raise ContractError(
                f"batched call returned {name!r} without a {GROUP_DIM!r} dim "
                f"(dims: {dims}); a batched callable stacks its results along "
                f"{GROUP_DIM!r}, one entry per point it was handed"
            )
        if var.sizes[GROUP_DIM] != size:
            raise ContractError(
                f"batched call was handed {size} point(s) but returned "
                f"{var.sizes[GROUP_DIM]} along {GROUP_DIM!r} for {name!r}"
            )
    return [result.isel({GROUP_DIM: i}, drop=True) for i in range(size)]
