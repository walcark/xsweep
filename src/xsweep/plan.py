"""Planning: everything decided before the first expensive call.

Planning resolves the loop points, deduplication, batching, the result and
store shapes, and the resume state, and it MUST do so without calling the
wrapped function even once.

Its output is a single stream of work items. Point enumeration, dedup,
batching and resume are four transformations of the same stream; resolving
them here rather than combining them at execution time is what keeps the
execution loop free of conditional branches, and keeps their interactions
decided in one place.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr

from .contract import Contract
from .errors import ContractError, PolicyError
from .policy import ResolvedPolicy
from .space import LoopAxis, LoopGrid

__all__ = ["ArgSpec", "Plan", "VarSpec", "WorkItem"]


@dataclass(frozen=True)
class WorkItem:
    """One unit of work: a point, possibly a batch of it.

    Parameters
    ----------
    point_index
        Position on the loop grid, aligned with the plan's loop dims.
    unique_index
        Position among the deduplicated points, or ``None`` without dedup.
    slices
        Batch slice per vec dim; empty when nothing is batched.
    done
        Already satisfied by the store, so execution skips it silently.
    skipped
        Excluded by the skip predicate, so it is never called.
    mirrors
        Other grid positions sharing this representative under dedup. Their
        values are copies of this one, so execution writes them without
        calling the function again.
    """

    point_index: tuple[int, ...]
    unique_index: int | None = None
    slices: Mapping[str, slice] = field(default_factory=dict)
    done: bool = False
    skipped: bool = False
    mirrors: tuple[tuple[int, ...], ...] = ()

    @property
    def runnable(self) -> bool:
        """Return whether execution must actually call the function."""
        return not (self.done or self.skipped)


@dataclass(frozen=True)
class ArgSpec:
    """One argument as a call receives it."""

    name: str
    kind: str
    dtype: str
    shape: tuple[int, ...] | None


@dataclass(frozen=True)
class VarSpec:
    """One result variable, with the sizes known so far."""

    name: str
    dims: tuple[str, ...]
    sizes: tuple[int | None, ...]
    dtype: str = "float64"

    @property
    def determined(self) -> bool:
        """Return whether every size is known without a probe call."""
        return all(s is not None for s in self.sizes)


@dataclass(frozen=True)
class StoreSpec:
    """Where and how results are laid out."""

    path: str | None
    chunks: Mapping[str, int]
    n_regions: int


@dataclass(frozen=True)
class Plan:
    """The resolved description of a sweep, inspectable before it runs.

    Execution consumes this object, which is what guarantees the report and
    the run cannot diverge.
    """

    contract: Contract
    policy: ResolvedPolicy
    axes: tuple[LoopAxis, ...]
    n_points: int
    n_unique: int | None
    batches: Mapping[str, tuple[slice, ...]]
    work_items: tuple[WorkItem, ...]
    call_signature: tuple[ArgSpec, ...]
    result: tuple[VarSpec, ...]
    store: StoreSpec | None
    executor: str
    grid: LoopGrid
    space: xr.Dataset
    statics: Mapping[str, Any]

    @property
    def loop_dims(self) -> tuple[str, ...]:
        """Return the loop dims, which are also the result's leading dims."""
        return tuple(a.name for a in self.axes)

    @property
    def n_calls(self) -> int:
        """Return how many calls a full run would make."""
        return sum(1 for item in self.work_items if not item.skipped)

    @property
    def n_cached(self) -> int:
        """Return how many work items the store already satisfies."""
        return sum(1 for item in self.work_items if item.done)

    @property
    def n_to_compute(self) -> int:
        """Return how many calls this run will actually make."""
        return sum(1 for item in self.work_items if item.runnable)

    @property
    def n_skipped(self) -> int:
        """Return how many points the skip predicate excluded."""
        return sum(1 for item in self.work_items if item.skipped)

    @property
    def determined(self) -> bool:
        """Return whether every result size is known without a probe call."""
        return all(v.determined for v in self.result)

    def __repr__(self) -> str:
        """Render the human-readable plan report."""
        return render(self)


def build_plan(
    *,
    contract: Contract,
    policy: ResolvedPolicy,
    space: xr.Dataset,
    grid: LoopGrid,
    statics: Mapping[str, Any],
    done_mask: np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None = None,
    unique_of: np.ndarray[tuple[int, ...], np.dtype[np.int64]] | None = None,
    n_unique: int | None = None,
) -> Plan:
    """Resolve a sweep into a plan.

    Parameters
    ----------
    contract
        The contract describing one call.
    policy
        The fully resolved run policy.
    space
        The validated sweep space.
    grid
        The enumerated loop grid.
    statics
        Configuration forwarded to every call.
    done_mask
        Boolean array over the loop grid, ``True`` where the store already
        holds a successful result. ``None`` means nothing is cached.
    unique_of
        Integer array over the loop grid mapping each point to its unique
        representative, or ``None`` when dedup is off.
    n_unique
        Number of unique points, or ``None`` when dedup is off.

    Returns
    -------
    Plan
        The plan, ready to inspect or to execute.
    """
    batches = _batch_slices(contract, policy, space)
    items = _work_items(
        contract=contract,
        policy=policy,
        grid=grid,
        batches=batches,
        done_mask=done_mask,
        unique_of=unique_of,
    )
    result = _result_spec(contract, space, grid)
    store = _store_spec(policy, result, grid.dims, batches, items)
    return Plan(
        contract=contract,
        policy=policy,
        axes=grid.axes,
        n_points=grid.n_points,
        n_unique=n_unique,
        batches=batches,
        work_items=items,
        call_signature=_call_signature(contract, space, batches, statics),
        result=result,
        store=store,
        executor=_render_executor(policy),
        grid=grid,
        space=space,
        statics=dict(statics),
    )


def _batch_slices(
    contract: Contract, policy: ResolvedPolicy, space: xr.Dataset
) -> dict[str, tuple[slice, ...]]:
    """Return the slice list for each batched vec dim.

    A vec variable may carry any number of dims. Batch sizes come from the
    policy, keyed by DIM; the contract's ``@ N`` is a one-dimensional
    shorthand and is rejected on anything else, since it could not say which
    dim it meant.
    """
    sizes: dict[str, int] = {}

    for var in contract.vec:
        if var.max_batch is None:
            continue
        dims = tuple(map(str, space[var.name].dims)) if var.name in space else ()
        if len(dims) != 1:
            hint = (
                f"Declare batch sizes per dim in the policy, e.g. "
                f"chunks={{{dims[0]!r}: {var.max_batch}}}"
                if dims
                else "The variable carries no dim in the space"
            )
            raise ContractError(
                f"'@ {var.max_batch}' on variable {var.name!r}, which has dims "
                f"{list(dims)!r}; the marker is a shorthand for 1-D variables "
                f"only, where variable and dim names coincide. {hint}"
            )
        sizes[dims[0]] = var.max_batch

    vec_dims = {
        dim
        for var in contract.vec
        if var.name in space
        for dim in map(str, space[var.name].dims)
    }
    for dim, size in policy.chunks.items():
        if dim not in vec_dims:
            listed = ", ".join(sorted(vec_dims)) or "none"
            raise PolicyError(
                f"chunks names dim {dim!r}, which no vec variable carries; "
                f"only vec dims can be batched. Batchable dims: {listed}"
            )
        sizes[dim] = size

    out: dict[str, tuple[slice, ...]] = {}
    for dim, size in sizes.items():
        total = int(space.sizes[dim])
        out[dim] = tuple(
            slice(start, min(start + size, total)) for start in range(0, total, size)
        )
    return out


def _work_items(
    *,
    contract: Contract,
    policy: ResolvedPolicy,
    grid: LoopGrid,
    batches: Mapping[str, tuple[slice, ...]],
    done_mask: np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None,
    unique_of: np.ndarray[tuple[int, ...], np.dtype[np.int64]] | None,
) -> tuple[WorkItem, ...]:
    """Expand the loop grid into the single stream execution consumes."""
    dims = tuple(batches)
    combos: list[dict[str, slice]] = [{}]
    for dim in dims:
        combos = [{**combo, dim: sl} for combo in combos for sl in batches[dim]]

    mirrors_of: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    representative: dict[int, tuple[int, ...]] = {}
    if unique_of is not None:
        for index in grid.indices():
            code = int(unique_of[index]) if index else int(unique_of[()])
            if code not in representative:
                representative[code] = index
                mirrors_of[index] = []
            elif representative[code] != index:
                mirrors_of[representative[code]].append(index)

    items: list[WorkItem] = []
    for index in grid.indices():
        if unique_of is not None and index not in mirrors_of:
            continue

        skipped = False
        if policy.skip_where is not None:
            skipped = bool(policy.skip_where(grid.values_at(index)))

        unique_index: int | None = None
        if unique_of is not None:
            unique_index = int(unique_of[index]) if index else int(unique_of[()])

        done = bool(done_mask[index]) if done_mask is not None else False
        for combo in combos:
            items.append(
                WorkItem(
                    point_index=index,
                    unique_index=unique_index,
                    slices=dict(combo),
                    done=done,
                    skipped=skipped,
                    mirrors=tuple(mirrors_of.get(index, ())),
                )
            )
    return tuple(items)


def _result_spec(
    contract: Contract, space: xr.Dataset, grid: LoopGrid
) -> tuple[VarSpec, ...]:
    """Describe each result variable, leaving unknown sizes as ``None``."""
    loop_dims = grid.dims
    loop_sizes = tuple(axis.size for axis in grid.axes)
    specs: list[VarSpec] = []
    for out in contract.out:
        call_sizes = tuple(
            int(space.sizes[d]) if d in space.sizes else None for d in out.dims
        )
        specs.append(
            VarSpec(
                name=out.name,
                dims=loop_dims + out.dims,
                sizes=loop_sizes + call_sizes,
            )
        )
    return tuple(specs)


def _store_spec(
    policy: ResolvedPolicy,
    result: Sequence[VarSpec],
    loop_dims: tuple[str, ...],
    batches: Mapping[str, tuple[slice, ...]],
    items: Sequence[WorkItem],
) -> StoreSpec:
    """Describe the store layout.

    The chunk grid is what makes concurrent region writes safe without any
    coordination: one along every loop dim, so two writers can never touch
    the same chunk because they never share a loop point. Along a call dim it
    is the batch size when batched, the full extent otherwise.
    """
    chunks: dict[str, int] = {dim: 1 for dim in loop_dims}
    for var in result:
        for dim, size in zip(var.dims, var.sizes, strict=True):
            if dim in loop_dims:
                continue
            if dim in batches:
                chunks[dim] = max(sl.stop - sl.start for sl in batches[dim])
            elif size is not None:
                chunks[dim] = size
    return StoreSpec(
        path=None if policy.store is None else str(policy.store),
        chunks=chunks,
        n_regions=len(items),
    )


def _call_signature(
    contract: Contract,
    space: xr.Dataset,
    batches: Mapping[str, tuple[slice, ...]],
    statics: Mapping[str, Any],
) -> tuple[ArgSpec, ...]:
    """Describe every argument one call receives."""
    specs: list[ArgSpec] = []
    for loop_var in contract.loop:
        dtype = str(space[loop_var.name].dtype) if loop_var.name in space else "?"
        specs.append(ArgSpec(loop_var.name, "loop", dtype, None))
    for vec_var in contract.vec:
        if vec_var.name not in space:
            specs.append(ArgSpec(vec_var.name, "vec", "?", None))
            continue
        var = space[vec_var.name]
        shape = tuple(
            (batches[d][0].stop - batches[d][0].start)
            if d in batches
            else int(var.sizes[d])
            for d in map(str, var.dims)
        )
        specs.append(ArgSpec(vec_var.name, "vec", str(var.dtype), shape))
    for name in contract.const:
        var = space[name]
        specs.append(ArgSpec(name, "const", str(var.dtype), tuple(var.shape)))
    for name, value in statics.items():
        specs.append(ArgSpec(name, "static", type(value).__name__, None))
    return tuple(specs)


def _render_executor(policy: ResolvedPolicy) -> str:
    """Render the executor as a short human-readable string."""
    if isinstance(policy.executor, str):
        if policy.executor == "process" and policy.max_workers is not None:
            return f"process(max_workers={policy.max_workers})"
        return policy.executor
    return type(policy.executor).__name__


def render(plan: Plan) -> str:
    """Render a plan as the textual report.

    Parameters
    ----------
    plan
        The plan to describe.

    Returns
    -------
    str
        A multi-line report: contract, space, counts, per-call arguments,
        result and store layout.
    """
    lines: list[str] = []
    lines.append(f"Sweep plan{' ' * 21}contract version {plan.contract.version!r}")
    lines.append(f"  {plan.contract.render()}")
    lines.append("")

    origin = " x ".join(
        f"{a.name}({'zip' if a.is_zipped else 'axis'}: {a.size})" for a in plan.axes
    )
    lines.append(f"Loop space     {origin or 'none'}")
    lines.append(f"  points       {plan.n_points}")
    if plan.n_unique is not None:
        lines.append(f"  unique       {plan.n_unique}   (dedup enabled)")
    else:
        lines.append("  dedup        disabled")

    for dim, slices in plan.batches.items():
        widths = sorted({sl.stop - sl.start for sl in slices}, reverse=True)
        detail = f"of <= {widths[0]}"
        if len(widths) > 1:
            detail += f" (last: {widths[-1]})"
        lines.append(f"Batches        {dim}: {len(slices)} batches {detail}")

    lines.append(f"Calls          {plan.n_calls}")
    lines.append(f"  cached       {plan.n_cached}")
    lines.append(f"  skipped      {plan.n_skipped}")
    lines.append(f"  to compute   {plan.n_to_compute}")
    lines.append("")

    lines.append("Each call receives")
    for arg in plan.call_signature:
        shape = "" if arg.shape is None else f" {list(arg.shape)}"
        lines.append(f"  {arg.name:<12} {arg.kind:<7} {arg.dtype}{shape}")

    lines.append("")
    for var in plan.result:
        sizes = ", ".join(
            f"{d}: {s if s is not None else '?'}"
            for d, s in zip(var.dims, var.sizes, strict=True)
        )
        lines.append(f"Result         {var.name} ({sizes}) {var.dtype}")
    if not plan.determined:
        lines.append("               sizes marked '?' are undetermined and will")
        lines.append("               be discovered by a probe call at execution")

    if plan.store is not None and plan.store.path is not None:
        rendered = ", ".join(f"{d}: {n}" for d, n in plan.store.chunks.items())
        chunks = rendered or "1 per point"
        lines.append(f"Store          {plan.store.path}")
        lines.append(f"  chunks       {chunks}")
        lines.append(f"  regions      {plan.store.n_regions}")
    else:
        lines.append("Store          none (in memory, no cache and no resume)")
    lines.append(f"Executor       {plan.executor}")
    return "\n".join(lines)
