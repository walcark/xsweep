"""The sweeper: plan, then execute the plan.

The two phases are separate on purpose. Planning decides everything and calls
nothing; execution consumes the plan and calls the function. That is what
lets a user inspect exactly what will run, and guarantees the report cannot
diverge from the run.
"""

from __future__ import annotations

import functools
import logging
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from . import policy as policy_mod
from .contract import Contract, coerce
from .dedup import unique_map
from .delivery import (
    assemble_args,
    assemble_group_args,
    normalise_return,
    split_group_return,
)
from .errors import ContractError, PointFailed, PolicyError, StoreError
from .executors import build_executor
from .lock import StoreLock
from .plan import Plan, WorkItem, build_plan
from .policy import ResolvedPolicy, SweepPolicy
from .space import build_grid, validate_space
from .store import FAILED, OK, SKIPPED, Store

__all__ = ["Sweeper", "sweep"]

logger = logging.getLogger("xsweep")


@dataclass
class Outcome:
    """What one call produced, or why it did not."""

    item: WorkItem
    data: xr.Dataset | None
    error: BaseException | None = None


class Sweeper:
    """Bind a contract and a default policy around a callable.

    Parameters
    ----------
    contract
        The contract describing one call, as a string or an object.
    func
        The callable to lift.
    policy
        Decorator-level defaults, overridable per instance and per call.
    version
        Physics revision, when the contract is given as a string.
    """

    def __init__(
        self,
        contract: str | Contract,
        func: Callable[..., Any],
        policy: SweepPolicy | None = None,
        *,
        version: str = "0",
        name: str | None = None,
    ) -> None:
        self.contract = coerce(contract, version=version)
        self.func = func
        self.policy = policy
        self.__doc__ = func.__doc__
        self.__name__ = name if name is not None else getattr(func, "__name__", "sweep")
        self.__module__ = getattr(func, "__module__", __name__)
        self.__qualname__ = getattr(func, "__qualname__", self.__name__)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle by reference for a function, by value for a bound method.

        The decorator rebinds a plain function's module-level name to this
        Sweeper, so the wrapped function is no longer reachable by its own
        qualified name; sending the sweeper by reference lets a worker
        re-import the module and find the very same object, function
        included.

        A bound method (``SweepModule.forward``) has no such problem, and a
        by-reference lookup would in fact land on the unbound function on
        the class, losing exactly the instance state built in ``__init__``
        that is the reason to use the class facade. A bound method already
        pickles correctly through its instance, so this reconstructs
        through the constructor instead.
        """
        if hasattr(self.func, "__self__"):
            return (
                _rebuild,
                (
                    self.contract,
                    self.func,
                    self.policy,
                    self.contract.version,
                    self.__name__,
                ),
            )
        return _lookup, (self.__module__, self.__qualname__)

    def explain(
        self,
        space: xr.Dataset,
        /,
        *,
        policy: SweepPolicy | None = None,
        **statics: Any,
    ) -> Plan:
        """Resolve the sweep and stop, without calling the function once.

        Parameters
        ----------
        space
            The Dataset describing the whole sweep.
        policy
            Call-level policy overrides.
        **statics
            Configuration forwarded to every call.

        Returns
        -------
        Plan
            The resolved plan, ready to inspect.
        """
        return self._plan(space, policy=policy, statics=statics)

    def __call__(
        self,
        space: xr.Dataset,
        /,
        *,
        policy: SweepPolicy | None = None,
        **statics: Any,
    ) -> xr.Dataset:
        """Plan the sweep, then execute it.

        Parameters
        ----------
        space
            The Dataset describing the whole sweep.
        policy
            Call-level policy overrides.
        **statics
            Configuration forwarded to every call.

        Returns
        -------
        xr.Dataset
            The declared outputs plus the ``status`` sidecar variable.
        """
        plan = self._plan(space, policy=policy, statics=statics)
        return _execute(plan, self)

    def _plan(
        self,
        space: xr.Dataset,
        *,
        policy: SweepPolicy | None,
        statics: Mapping[str, Any],
    ) -> Plan:
        """Validate everything, then resolve the plan."""
        policy_mod.check_static_names(statics)
        resolved = policy_mod.resolve(self.policy, policy)
        validate_space(space, self.contract)
        policy_mod.validate(resolved, available_dims=tuple(map(str, space.sizes)))
        grid = build_grid(space, self.contract)
        done_mask = _read_done_mask(self.contract, resolved, grid.shape, statics)

        unique_of = None
        n_unique = None
        if resolved.dedup:
            dims = resolved.dedup if isinstance(resolved.dedup, tuple) else None
            unique_of, n_unique = unique_map(grid, dims)

        return build_plan(
            name=self.__name__,
            contract=self.contract,
            policy=resolved,
            space=space,
            grid=grid,
            statics=statics,
            done_mask=done_mask,
            unique_of=unique_of,
            n_unique=n_unique,
        )


def _lookup(module: str, qualname: str) -> Sweeper:
    """Re-import a sweeper by name, for unpickling in a worker process."""
    import importlib

    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, Sweeper):
        raise PolicyError(
            f"{module}.{qualname} is not a sweeper; the process executor "
            "needs the decorated function to be reachable at module level"
        )
    return obj


def _rebuild(
    contract: Contract,
    func: Callable[..., Any],
    policy: SweepPolicy | None,
    version: str,
    name: str,
) -> Sweeper:
    """Reconstruct a sweeper wrapping a bound method, for unpickling.

    ``func`` here is a bound method, already unpickled with its own instance
    state intact by the time this runs; wiring it back into a fresh Sweeper
    is all that is left to do.
    """
    return Sweeper(contract, func, policy, version=version, name=name)


def sweep(
    contract: str | Contract,
    *,
    version: str = "0",
    **policy_fields: Any,
) -> Callable[[Callable[..., Any]], Sweeper]:
    """Lift a point function into a sweep.

    Parameters
    ----------
    contract
        The contract describing one call. Parsed immediately, so a malformed
        contract raises at import rather than after minutes of runs.
    version
        Physics revision; bump it whenever the computation changes for
        reasons xsweep cannot observe.
    **policy_fields
        Decorator-level policy defaults.

    Returns
    -------
    callable
        A decorator producing a :class:`Sweeper`.
    """
    defaults = SweepPolicy(**policy_fields) if policy_fields else None

    def decorate(func: Callable[..., Any]) -> Sweeper:
        return Sweeper(contract, func, defaults, version=version)

    return decorate


def _read_done_mask(
    contract: Contract,
    policy: ResolvedPolicy,
    shape: tuple[int, ...],
    statics: Mapping[str, Any],
) -> np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None:
    """Return which points a previous run already completed."""
    from pathlib import Path

    if policy.store is None or not (Path(str(policy.store)) / "zarr.json").exists():
        return None
    import zarr

    group = zarr.open_group(str(policy.store), mode="r")
    if "status" not in group:
        return None
    array = group["status"]
    assert isinstance(array, zarr.Array)
    status = np.asarray(array[...], dtype=np.uint8)
    if status.shape != shape:
        return None
    return np.equal(status, OK)


def _execute(plan: Plan, target: Sweeper) -> xr.Dataset:
    """Run a plan and return its result, holding the store lock throughout."""
    if plan.policy.store is None:
        return _run(plan, target)
    with StoreLock(str(plan.policy.store), force=plan.policy.force_unlock):
        return _run(plan, target)


def _run(plan: Plan, target: Sweeper) -> xr.Dataset:
    """Execute a plan against an already-locked store."""
    started = time.monotonic()
    runnable = [item for item in plan.work_items if item.runnable]

    sizes: dict[str, int] = {}
    probe: list[Outcome] | None = None
    if not plan.determined and runnable:
        # The store cannot be allocated before the output shape is known, and
        # the callee costs minutes, so the probe result is kept, not discarded.
        # A batched contract is probed with a whole group, and the sizes are
        # read after the split, so the group dim never reaches the store.
        probe = _call_group(plan, target, _groups(plan, runnable)[0])
        first = probe[0].data
        if first is None:
            # Without the probe there is no output shape, so the store
            # cannot be allocated and the run cannot continue whatever
            # `on_error` says.  Raising here, with the cause attached,
            # keeps the failure where it happened: reported from the
            # allocation instead, it reads as a contract problem and
            # sends the reader to fix a declaration that was never wrong.
            raise StoreError(
                "the probe call failed, so the output shape is unknown: "
                f"{probe[0].error}"
            ) from probe[0].error
        sizes.update({str(d): int(n) for d, n in first.sizes.items()})

    store = Store.open_or_create(plan, sizes=sizes)
    if plan.policy.store is None:
        logger.info(
            "sweep runs in memory: results are not persisted and no cache was consulted"
        )
    logger.info(
        "sweep start points=%d calls=%d cached=%d skipped=%d",
        plan.n_points,
        plan.n_calls,
        plan.n_cached,
        plan.n_skipped,
    )

    _mark_skipped(plan, store)

    n_ok = n_failed = 0
    buffers = _WriteBuffers(plan, store, runnable)
    try:
        for outcome in _stream(plan, target, runnable, probe):
            if outcome.data is None:
                n_failed += 1
                buffers.place(outcome, FAILED)
                logger.debug(
                    "point failed index=%s error=%s",
                    outcome.item.point_index,
                    outcome.error,
                )
                continue
            buffers.place(outcome, OK)
            n_ok += 1
            logger.debug("point ok index=%s", outcome.item.point_index)
    finally:
        # Also reached on a raise: whatever is buffered was already computed,
        # so it is flushed before the exception propagates, not discarded.
        buffers.flush_all()

    # Unconditional: a run that resumes with every representative already
    # computed still has duplicated points to fill, and skipping the pass
    # would hand back a result full of holes.
    if plan.source_of is not None:
        store.expand(plan.source_of)
        store.expand_status(plan.source_of)

    store.finalise()
    logger.info(
        "sweep done ok=%d failed=%d skipped=%d cached=%d elapsed=%.1fs",
        n_ok,
        n_failed,
        plan.n_skipped,
        plan.n_cached,
        time.monotonic() - started,
    )
    return store.result(load=plan.policy.load or plan.policy.store is None)


def _stream(
    plan: Plan,
    target: Sweeper,
    runnable: list[WorkItem],
    probe: list[Outcome] | None,
) -> Iterator[Outcome]:
    """Dispatch the runnable items, reusing the probe result if there was one.

    A generator, not a list: yielding as each outcome completes is what lets
    `_run` write (or buffer) it immediately, so a later item's failure can
    never discard an earlier item's already-computed result.
    """
    groups = _groups(plan, runnable)
    if probe is not None:
        yield from probe
        rest = groups[1:]
    else:
        rest = groups

    executor = build_executor(plan.policy.executor, max_workers=plan.policy.max_workers)
    # A partial rather than a closure: a lambda cannot be pickled, so the
    # process executor would reject every sweep before running a single one.
    call = functools.partial(_call_group, plan, target)
    for _, outcomes in executor.map_unordered(call, rest):
        yield from outcomes


def _groups(plan: Plan, items: list[WorkItem]) -> list[list[WorkItem]]:
    """Cut the runnable items into the groups one batched call receives.

    Items are grouped only among those sharing the same vec batch slices:
    one call carries one vec batch, so points wanting different slices
    cannot travel together.  Ungrouped contracts yield one item per group,
    which keeps the execution path identical for both.
    """
    if not plan.contract.is_batched:
        return [[item] for item in items]

    size = max(1, plan.policy.batch_size)
    by_slices: dict[tuple[tuple[str, int, int], ...], list[WorkItem]] = {}
    for item in items:
        key = tuple(sorted((d, sl.start, sl.stop) for d, sl in item.slices.items()))
        by_slices.setdefault(key, []).append(item)

    groups: list[list[WorkItem]] = []
    for bucket in by_slices.values():
        groups += [bucket[i : i + size] for i in range(0, len(bucket), size)]
    return groups


def _call_group(plan: Plan, target: Sweeper, group: list[WorkItem]) -> list[Outcome]:
    """Invoke the wrapped function once for a whole group of points.

    Returns one :class:`Outcome` per point, so everything downstream —
    the write buffers, status, dedup expansion — never learns that the
    points travelled together.  A failure is attributed to the whole
    group: the callee produced nothing, and guessing which point caused
    it would be a policy of its own.
    """
    if not plan.contract.is_batched:
        return [_call(plan, target, group[0])]

    args = assemble_group_args(
        plan.contract,
        group_values=[plan.grid.values_at(item.point_index) for item in group],
        arrays=_arrays_for(plan, group[0]),
        statics=plan.statics,
    )
    attempts = plan.policy.retries + 1
    last: BaseException | None = None
    for _ in range(attempts):
        try:
            data = normalise_return(target.func(**args), plan.contract)
        except ContractError:
            # The callable does not honour its own contract. Retrying would
            # fail identically and recording nan would hide the bug, so this
            # one propagates whatever the error policy says.
            raise
        except Exception as exc:  # noqa: BLE001 - policy decides what happens
            last = exc
            continue
        parts = split_group_return(data, len(group))
        return [Outcome(item, part) for item, part in zip(group, parts, strict=True)]
    if plan.policy.on_error == "raise":
        raise PointFailed(
            f"batched call failed over {len(group)} point(s) starting at "
            f"{group[0].point_index} after {attempts} attempt(s): {last}"
        ) from last
    return [Outcome(item, None, last) for item in group]


def _call(plan: Plan, target: Sweeper, item: WorkItem) -> Outcome:
    """Invoke the wrapped function for one work item, honouring the policy."""
    args = assemble_args(
        plan.contract,
        loop_values=plan.grid.values_at(item.point_index),
        arrays=_arrays_for(plan, item),
        statics=plan.statics,
    )
    attempts = plan.policy.retries + 1
    last: BaseException | None = None
    for _ in range(attempts):
        try:
            data = normalise_return(target.func(**args), plan.contract)
            return Outcome(item, data)
        except Exception as exc:  # noqa: BLE001 - policy decides what happens
            last = exc
    if plan.policy.on_error == "raise":
        raise PointFailed(
            f"call failed at point {item.point_index} after {attempts} "
            f"attempt(s): {last}"
        ) from last
    return Outcome(item, None, last)


def _arrays_for(plan: Plan, item: WorkItem) -> dict[str, xr.DataArray]:
    """Slice vec variables, and const variables on any unprotected shared dim."""
    arrays: dict[str, xr.DataArray] = {}
    for var in plan.contract.vec:
        array = plan.space[var.name]
        selection = {dim: sl for dim, sl in item.slices.items() if dim in array.dims}
        arrays[var.name] = array.isel(selection) if selection else array
    for const_var in plan.contract.const:
        array = plan.space[const_var.name]
        selection = {
            dim: sl
            for dim, sl in item.slices.items()
            if dim in array.dims and dim not in const_var.protected
        }
        arrays[const_var.name] = array.isel(selection) if selection else array
    return arrays


class _WriteBuffers:
    """Accumulate outcomes per store chunk, flushing whole chunks at once.

    The loop-dim chunk grid is sized in bytes, not points (Plan.store.chunks),
    so several work items usually share one chunk. Writing each of them
    straight into the store would read-modify-write that chunk once per
    point, and status is no exception: a zarr write's cost is dominated by
    its fixed per-call overhead, not by the one byte a status code actually
    carries. Both are buffered the same way: read the chunk once, place
    every point it produces, write it back once, when every runnable point
    that chunk owns has an outcome.
    """

    def __init__(self, plan: Plan, store: Store, runnable: list[WorkItem]) -> None:
        """Count, per chunk, how many runnable points it still owes."""
        self.plan = plan
        self.store = store
        self.remaining: Counter[tuple[Any, ...]] = Counter(
            _chunk_key(plan, item) for item in runnable
        )
        self.buffers: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        self.status_buffers: dict[tuple[Any, ...], np.ndarray] = {}
        self.direct = _is_direct(plan)

    def place(self, outcome: Outcome, code: int) -> None:
        """Place an outcome's data (if any) and status, buffered per chunk."""
        item = outcome.item
        key = _chunk_key(self.plan, item)
        if self.direct:
            if outcome.data is not None:
                self.store.write(_region(self.plan, item), _placed(self.plan, outcome))
            self.store.set_status(item.point_index, code)
            self._resolve(key)
            return

        region = _chunk_region(self.plan, item)
        local = _local_region(self.plan, item, region)
        if outcome.data is not None:
            if key not in self.buffers:
                self.buffers[key] = self.store.read_chunk(region)
            self.store.place_in_chunk(
                self.buffers[key], local, _placed(self.plan, outcome)
            )
        if key not in self.status_buffers:
            self.status_buffers[key] = self.store.read_status_chunk(region)
        index = tuple(local[dim].start for dim in self.plan.loop_dims)
        self.status_buffers[key][index] = code
        self._resolve(key)

    def _resolve(self, key: tuple[Any, ...]) -> None:
        self.remaining[key] -= 1
        if self.remaining[key] <= 0:
            self._flush(key)

    def _flush(self, key: tuple[Any, ...]) -> None:
        self.remaining.pop(key, None)
        buffer = self.buffers.pop(key, None)
        status_buffer = self.status_buffers.pop(key, None)
        if buffer is None and status_buffer is None:
            return
        region = _chunk_region_for_key(self.plan, key)
        if buffer is not None:
            self.store.flush_chunk(region, buffer)
        if status_buffer is not None:
            self.store.flush_status_chunk(region, status_buffer)

    def flush_all(self) -> None:
        """Write back every chunk still buffered, complete or not."""
        for key in list(self.remaining):
            self._flush(key)


def _is_direct(plan: Plan) -> bool:
    """Return whether every loop chunk holds exactly one point.

    A one-point chunk has nothing else to preserve, so reading it before
    writing back would only add a read no direct write ever paid for.
    """
    widths = plan.store.chunks if plan.store else {}
    return all(widths.get(dim, 1) <= 1 for dim in plan.loop_dims)


def _mark_skipped(plan: Plan, store: Store) -> None:
    """Record every skipped point's status, one write per chunk it touches.

    Skipped points are known entirely from the plan, before any call runs,
    so they are grouped by chunk directly rather than through the streaming
    write buffer, which exists to accumulate outcomes as they arrive.
    """
    direct = _is_direct(plan)
    by_chunk: dict[tuple[Any, ...], list[WorkItem]] = {}
    for item in plan.work_items:
        if not item.skipped:
            continue
        if direct:
            store.set_status(item.point_index, SKIPPED)
            continue
        by_chunk.setdefault(_chunk_key(plan, item), []).append(item)

    for items in by_chunk.values():
        region = _chunk_region(plan, items[0])
        buffer = store.read_status_chunk(region)
        for item in items:
            local = _local_region(plan, item, region)
            index = tuple(local[dim].start for dim in plan.loop_dims)
            buffer[index] = SKIPPED
        store.flush_status_chunk(region, buffer)


def _chunk_key(plan: Plan, item: WorkItem) -> tuple[Any, ...]:
    """Identify the store chunk a work item's region falls into.

    A chunk is a loop-dim chunk of the grid combined with the item's own
    batch slices: a batched call dim already writes exactly one chunk per
    item (FR-010's "batch size is a chunk size"), so items with different
    batches never share a chunk even at the same loop position.
    """
    widths = plan.store.chunks if plan.store else {}
    loop_part = tuple(
        pos // widths.get(dim, 1)
        for dim, pos in zip(plan.loop_dims, item.point_index, strict=True)
    )
    batch_part = tuple(sorted((d, sl.start, sl.stop) for d, sl in item.slices.items()))
    return (loop_part, batch_part)


def _chunk_region(plan: Plan, item: WorkItem) -> dict[str, slice]:
    """Return the global region the whole chunk owning this item covers."""
    widths = plan.store.chunks if plan.store else {}
    region: dict[str, slice] = {}
    for dim, pos, axis in zip(plan.loop_dims, item.point_index, plan.axes, strict=True):
        width = widths.get(dim, 1)
        start = (pos // width) * width
        region[dim] = slice(start, min(start + width, axis.size))
    region.update(item.slices)
    return region


def _chunk_region_for_key(plan: Plan, key: tuple[Any, ...]) -> dict[str, slice]:
    """Rebuild a chunk's global region from its key, to flush without an item."""
    widths = plan.store.chunks if plan.store else {}
    loop_part, batch_part = key
    region: dict[str, slice] = {}
    for dim, cid, axis in zip(plan.loop_dims, loop_part, plan.axes, strict=True):
        width = widths.get(dim, 1)
        start = cid * width
        region[dim] = slice(start, min(start + width, axis.size))
    for dim, start, stop in batch_part:
        region[dim] = slice(start, stop)
    return region


def _local_region(
    plan: Plan, item: WorkItem, chunk_region: Mapping[str, slice]
) -> dict[str, slice]:
    """Translate a work item's global region into offsets local to its chunk."""
    local: dict[str, slice] = {}
    for dim, pos in zip(plan.loop_dims, item.point_index, strict=True):
        base = chunk_region[dim].start
        local[dim] = slice(pos - base, pos - base + 1)
    for dim, sl in item.slices.items():
        base = chunk_region[dim].start
        local[dim] = slice(sl.start - base, sl.stop - base)
    return local


def _region(plan: Plan, item: WorkItem) -> dict[str, slice]:
    """Return the store region one work item owns."""
    return _region_at(plan, item.point_index, item)


def _region_at(plan: Plan, index: tuple[int, ...], item: WorkItem) -> dict[str, slice]:
    """Return the store region at one grid position, with the item's batch."""
    region = {
        dim: slice(pos, pos + 1) for dim, pos in zip(plan.loop_dims, index, strict=True)
    }
    region.update(item.slices)
    return region


def _placed(plan: Plan, outcome: Outcome) -> xr.Dataset:
    """Return the call output as the store expects it.

    The loop dims are NOT added here: the store places the values itself by
    reshaping into its own selection, which avoids an xarray round trip on
    every point. Coordinates are dropped because the store already holds
    them from allocation.
    """
    assert outcome.data is not None
    return outcome.data.drop_vars(list(outcome.data.coords), errors="ignore")
