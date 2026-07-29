"""What one call receives, and what it is allowed to return.

Arguments are assembled from the contract, never from the callable's
signature: introspection is what makes a typo silently drop an argument, and
the contract already names everything.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

from .contract import Contract
from .errors import ContractError

__all__ = ["assemble_args", "normalise_return"]


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
        Vec variables sliced to the current batch, and const variables whole.
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
    for name in contract.const:
        args[name] = arrays[name]
    args.update(statics)
    return args


def normalise_return(result: Any, contract: Contract) -> xr.Dataset:
    """Turn whatever the callable returned into a named Dataset.

    Parameters
    ----------
    result
        A bare array for a single-output contract, an ordered tuple for a
        multi-output one, or an explicit Dataset.
    contract
        The contract declaring the expected output names.

    Returns
    -------
    xr.Dataset
        One variable per declared output, named as declared.

    Raises
    ------
    ContractError
        If the shape of the return does not match the declaration, or a
        Dataset omits a declared output.
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

    if len(names) == 1:
        if isinstance(result, tuple):
            raise ContractError(
                f"contract declares one output {names[0]!r} but the function "
                f"returned a tuple of {len(result)}"
            )
        return xr.Dataset({names[0]: _as_dataarray(result, names[0])})

    if not isinstance(result, tuple):
        raise ContractError(
            f"contract declares {len(names)} outputs {list(names)!r} so the "
            f"function must return a tuple in that order, got "
            f"{type(result).__name__}"
        )
    if len(result) != len(names):
        raise ContractError(
            f"contract declares {len(names)} outputs {list(names)!r} but the "
            f"function returned {len(result)} values"
        )
    return xr.Dataset(
        {
            name: _as_dataarray(value, name)
            for name, value in zip(names, result, strict=True)
        }
    )


def _as_dataarray(value: Any, name: str) -> xr.DataArray:
    """Coerce one returned value to a named DataArray."""
    array = value if isinstance(value, xr.DataArray) else xr.DataArray(value)
    return array.rename(name)
