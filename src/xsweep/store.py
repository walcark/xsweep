"""The store: cache, output and resume are the same artefact.

Results stream into a pre-allocated zarr store, one region per work item.
That single mechanism is what makes a sweep resumable, memory-bounded and
safely writable in parallel: the chunk grid is one along every loop dim, so
two writers never touch the same chunk because they never share a point.

The cache is not a hash-keyed side table. It is this store, identified by a
fingerprint over the contract and the statics, and addressed by coordinates.
A point is already computed when its status reads ``ok`` at its position.

Allocation goes through the zarr API rather than ``Dataset.to_zarr(
compute=False)``, because the xarray path returns a delayed object and
therefore imports dask, which must stay optional.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr

from .contract import Contract
from .errors import StoreError
from .plan import Plan, VarSpec

__all__ = ["Store", "fingerprint"]

#: Status codes, carried in the sidecar variable's attributes.
PENDING, OK, FAILED, SKIPPED = 0, 1, 2, 3
STATUS_LABELS = {PENDING: "pending", OK: "ok", FAILED: "failed", SKIPPED: "skipped"}

_META_KEY = "xsweep_meta"


def fingerprint(contract: Contract, statics: Mapping[str, Any]) -> str:
    """Compute the identity of a sweep configuration.

    Parameters
    ----------
    contract
        The contract, whose rendering and version both enter the digest.
    statics
        Configuration forwarded to every call. Statics change results, so
        they are part of the identity, not of the policy.

    Returns
    -------
    str
        A hex digest.

    Raises
    ------
    StoreError
        If a static can be neither serialised nor tokenised, since a cache
        that cannot see a change is worse than no cache.
    """
    payload = json.dumps(
        {
            "contract": contract.render(),
            "version": contract.version,
            "statics": _canonical(statics),
        },
        sort_keys=True,
    )
    return blake2b(payload.encode(), digest_size=16).hexdigest()


def _canonical(statics: Mapping[str, Any]) -> Any:
    """Render statics as canonical JSON-compatible data."""

    def encode(obj: Any) -> Any:
        token = getattr(obj, "__cache_token__", None)
        if callable(token):
            return token()
        raise StoreError(
            f"static of type {type(obj).__name__!r} is not JSON-serialisable "
            "and exposes no __cache_token__(); the cache could not tell when "
            "it changes. Pass a serialisable value, or add a "
            "__cache_token__() method returning one"
        )

    return json.loads(json.dumps(statics, sort_keys=True, default=encode))


def space_signature(plan: Plan) -> dict[str, list[Any]]:
    """Describe the coordinates a store was written against.

    Comparing this on resume is what turns "the space changed" from a silent
    misalignment into a refusal.
    """
    signature: dict[str, list[Any]] = {}
    for var in plan.result:
        for dim, size in zip(var.dims, var.sizes, strict=True):
            if dim in signature or size is None:
                continue
            if dim in plan.space.coords:
                signature[dim] = [
                    _jsonable(v) for v in plan.space.coords[dim].values.tolist()
                ]
            else:
                signature[dim] = [size]
    return signature


def _jsonable(value: Any) -> Any:
    """Coerce a coordinate value to something JSON can hold."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.datetime64,)):
        return str(value)
    return value


@dataclass
class Store:
    """A pre-allocated zarr store, written one region at a time."""

    group: zarr.Group
    plan: Plan
    loop_dims: tuple[str, ...]

    @classmethod
    def open_or_create(cls, plan: Plan, *, sizes: Mapping[str, int]) -> Store:
        """Open an existing store, or allocate a new one.

        Parameters
        ----------
        plan
            The plan being executed.
        sizes
            Sizes for dims the plan could not determine, discovered by the
            probe call.

        Returns
        -------
        Store
            The ready-to-write store.

        Raises
        ------
        StoreError
            If an existing store was written under a different fingerprint or
            a different space.
        """
        target: Any
        if plan.policy.store is None:
            target = zarr.storage.MemoryStore()
        else:
            target = str(Path(plan.policy.store))

        digest = fingerprint(plan.contract, plan.statics)
        signature = space_signature(plan)

        existing = cls._try_open(target)
        if existing is not None:
            cls._check_compatible(existing, digest, signature, plan)
            return cls(existing, plan, plan.loop_dims)

        group = zarr.open_group(target, mode="w")
        cls._allocate(group, plan, sizes)
        group.attrs[_META_KEY] = {
            "fingerprint": digest,
            "contract": plan.contract.render(),
            "version": plan.contract.version,
            "space_signature": signature,
            "chunks": dict(plan.store.chunks) if plan.store else {},
        }
        zarr.consolidate_metadata(group.store)
        return cls(zarr.open_group(target, mode="r+"), plan, plan.loop_dims)

    @staticmethod
    def _try_open(target: Any) -> zarr.Group | None:
        """Return the store at ``target`` if it already holds a sweep."""
        if not isinstance(target, str):
            return None
        # Presence of the group metadata, not of the directory: the write
        # lock creates the directory before anything is allocated in it.
        if not (Path(target) / "zarr.json").exists():
            return None
        try:
            group = zarr.open_group(target, mode="r+")
        except Exception as exc:  # noqa: BLE001 - surfaced with context
            raise StoreError(f"cannot open store at {target!r}: {exc}") from exc
        if _META_KEY not in group.attrs:
            raise StoreError(
                f"{target!r} exists but was not written by xsweep; point the "
                "policy at a fresh path"
            )
        return group

    @staticmethod
    def _check_compatible(
        group: zarr.Group,
        digest: str,
        signature: Mapping[str, list[Any]],
        plan: Plan,
    ) -> None:
        """Refuse a store that does not belong to this sweep configuration."""
        raw: Any = group.attrs[_META_KEY]
        meta: dict[str, Any] = dict(raw)
        if meta.get("fingerprint") != digest:
            raise StoreError(
                f"store holds results for a different configuration "
                f"(contract {meta.get('contract')!r}, version "
                f"{meta.get('version')!r}); bump the store path, or restore "
                "the contract, version and statics it was written with"
            )
        stored: dict[str, list[Any]] = dict(meta.get("space_signature", {}))
        for dim, values in signature.items():
            if dim in stored and stored[dim] != values:
                raise StoreError(
                    f"store was written with a different {dim!r} axis "
                    f"({len(stored[dim])} values, now {len(values)}); resuming "
                    "would misalign results. Use a fresh store path"
                )

    @staticmethod
    def _allocate(group: zarr.Group, plan: Plan, sizes: Mapping[str, int]) -> None:
        """Create every array, metadata only, with the planned chunk grid."""
        resolved = _resolved_specs(plan.result, sizes)
        chunk_of = dict(plan.store.chunks) if plan.store else {}

        written: set[str] = set()
        for var in resolved:
            for dim, size in zip(var.dims, var.sizes, strict=True):
                assert size is not None
                if dim in written:
                    continue
                written.add(dim)
                coord = _coord_values(plan, dim, size)
                arr = group.create_array(
                    dim,
                    shape=(size,),
                    dtype=coord.dtype,
                    chunks=(size,),
                    dimension_names=(dim,),
                )
                arr[:] = coord

        for var in resolved:
            shape = tuple(int(s) for s in var.sizes if s is not None)
            group.create_array(
                var.name,
                shape=shape,
                dtype=var.dtype,
                chunks=tuple(
                    chunk_of.get(d, s) for d, s in zip(var.dims, shape, strict=True)
                ),
                dimension_names=var.dims,
                fill_value=np.nan,
            )

        loop_shape = tuple(a.size for a in plan.axes)
        status = group.create_array(
            "status",
            shape=loop_shape,
            dtype="uint8",
            chunks=tuple(1 for _ in loop_shape),
            dimension_names=plan.loop_dims,
            fill_value=PENDING,
        )
        status.attrs["labels"] = {str(k): v for k, v in STATUS_LABELS.items()}

    def write(self, region: Mapping[str, slice], data: xr.Dataset) -> None:
        """Write one region of the result.

        Parameters
        ----------
        region
            Slice per dim, covering exactly one work item.
        data
            The normalised call output, already carrying the loop dims.
        """
        try:
            data.to_zarr(self.group.store, region=dict(region))
        except Exception as exc:  # noqa: BLE001 - surfaced with context
            raise StoreError(
                f"cannot write region {dict(region)!r}: {exc}. This usually "
                "means a call returned a shape incompatible with the "
                "allocated store, which the contract must declare"
            ) from exc

    def set_status(self, index: tuple[int, ...], code: int) -> None:
        """Record the outcome of one loop point."""
        array = self.group["status"]
        assert isinstance(array, zarr.Array)
        array[index if index else ()] = np.uint8(code)

    def read_status(self) -> np.ndarray[tuple[int, ...], np.dtype[np.uint8]]:
        """Return the status array, used for resume and for progress."""
        array = self.group["status"]
        assert isinstance(array, zarr.Array)
        return np.asarray(array[...], dtype=np.uint8)

    def finalise(self) -> None:
        """Consolidate metadata so later opens are fast and quiet."""
        zarr.consolidate_metadata(self.group.store)

    def result(self, *, load: bool) -> xr.Dataset:
        """Return the sweep result.

        Parameters
        ----------
        load
            Materialise instead of returning a lazily-indexed handle.

        Returns
        -------
        xr.Dataset
            The outputs plus the status variable. ``chunks=None`` keeps dask
            out of the picture: laziness here is xarray's own indexing.
        """
        ds: xr.Dataset = xr.open_zarr(self.group.store, chunks=None, consolidated=True)
        return ds.load() if load else ds


def _resolved_specs(
    specs: tuple[VarSpec, ...], sizes: Mapping[str, int]
) -> tuple[VarSpec, ...]:
    """Fill undetermined sizes with what the probe call discovered."""
    out: list[VarSpec] = []
    for var in specs:
        filled = tuple(
            size if size is not None else sizes.get(dim)
            for dim, size in zip(var.dims, var.sizes, strict=True)
        )
        missing = [d for d, s in zip(var.dims, filled, strict=True) if s is None]
        if missing:
            raise StoreError(
                f"cannot allocate {var.name!r}: sizes for {missing!r} are still "
                "unknown. Declare them in the contract, or let the probe call "
                "discover them"
            )
        out.append(VarSpec(var.name, var.dims, filled, var.dtype))
    return tuple(out)


def _coord_values(
    plan: Plan, dim: str, size: int
) -> np.ndarray[tuple[int, ...], np.dtype[Any]]:
    """Return the coordinate for one dim, falling back to a range."""
    if dim in plan.space.coords:
        return np.asarray(plan.space.coords[dim].values)
    if dim in plan.space.variables and plan.space[dim].ndim == 1:
        return np.asarray(plan.space[dim].values)
    return np.arange(size)
