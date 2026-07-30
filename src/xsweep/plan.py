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

    name: str
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
    source_of: np.ndarray[tuple[int, ...], np.dtype[np.int64]] | None = None

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
    name: str = "sweep",
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
    name
        The wrapped callable's name, used only to label the report.
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
    store = _store_spec(policy, result, grid.axes, batches, items)
    source_of = _source_map(grid, items) if unique_of is not None else None
    return Plan(
        name=name,
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
        source_of=source_of,
    )


def _source_map(
    grid: LoopGrid, items: Sequence[WorkItem]
) -> np.ndarray[tuple[int, ...], np.dtype[np.int64]]:
    """Map every loop point to the flat index of the point standing for it.

    Duplicated points are not computed and not written during the sweep; this
    is what the final expansion pass reads to fill them in one go.
    """
    flat = np.arange(grid.n_points, dtype=np.int64).reshape(grid.shape)
    source = flat.copy()
    for item in items:
        origin = int(flat[item.point_index]) if item.point_index else 0
        for mirror in item.mirrors:
            source[mirror] = origin
    return source.reshape(-1)


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
    out_dims = set(contract.out_dims)
    for dim, size in policy.chunks.items():
        if dim not in vec_dims:
            listed = ", ".join(sorted(vec_dims)) or "none"
            raise PolicyError(
                f"chunks names dim {dim!r}, which no vec variable carries; "
                f"only vec dims can be batched. Batchable dims: {listed}"
            )
        if dim not in out_dims:
            raise PolicyError(
                f"chunks batches dim {dim!r}, which is absent from every "
                f"declared output {sorted(out_dims)!r}; the function reduces "
                "over it, so batching would corrupt the result. Remove it "
                "from chunks to pass the whole axis in one call"
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


#: Memory ceiling for one loop-dim chunk's worth of buffered results, same
#: role as store.py's _EXPAND_BUDGET: not a policy field, since it never
#: changes a result, only the cost of writing it.
_LOOP_CHUNK_BUDGET = 64 * 1024 * 1024


def _store_spec(
    policy: ResolvedPolicy,
    result: Sequence[VarSpec],
    axes: tuple[LoopAxis, ...],
    batches: Mapping[str, tuple[slice, ...]],
    items: Sequence[WorkItem],
) -> StoreSpec:
    """Describe the store layout.

    Loop dims get a chunk grid sized from a memory budget, or from
    ``policy.loop_chunks``: a size-1 grid means every point is its own zarr
    chunk, so a write becomes a read-modify-write of that whole chunk on
    every single point, which is what execution's write buffer exists to
    amortise into one read and one write per chunk instead. Along a call
    dim, the chunk is the batch size when batched (a batch already fills one
    chunk exactly) or the full extent otherwise.
    """
    chunks: dict[str, int] = _loop_chunk_widths(policy, result, axes)
    loop_dims = set(chunks)
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


def _loop_chunk_widths(
    policy: ResolvedPolicy, result: Sequence[VarSpec], axes: tuple[LoopAxis, ...]
) -> dict[str, int]:
    """Pick a chunk width per loop dim: the budget, then the policy override.

    Only the first loop dim is auto-chunked; the others default to their
    full size, the same shape Store.expand already assumes when it slabs
    duplicates along the first loop dim only. Any dim can still be set
    explicitly through ``policy.loop_chunks``, including the first.
    """
    loop_dims = tuple(a.name for a in axes)
    if not loop_dims:
        return {}

    widths = {dim: axis.size for dim, axis in zip(loop_dims, axes, strict=True)}
    row_items = 1
    for axis in axes[1:]:
        row_items *= axis.size
    row_bytes = _bytes_per_loop_row(result, len(axes), row_items)
    first = loop_dims[0]
    widths[first] = min(axes[0].size, max(1, _LOOP_CHUNK_BUDGET // max(row_bytes, 1)))

    for dim, size in policy.loop_chunks.items():
        if dim not in widths:
            raise PolicyError(
                f"loop_chunks names dim {dim!r}, which is not a loop dim; "
                f"loop dims are {list(loop_dims)!r}"
            )
        widths[dim] = size
    return widths


def _bytes_per_loop_row(
    result: Sequence[VarSpec], n_loop_dims: int, row_items: int
) -> int:
    """Estimate bytes for one row along the first loop dim, other loop dims whole.

    An undetermined call-dim size (still ``None`` before the probe) counts as
    one: an underestimate biases towards a wider chunk, which only costs a
    bigger read on the rare contract where the probe reveals a much wider
    axis, never an incorrect one.
    """
    total = 0
    for var in result:
        cells = 1
        for size in var.sizes[n_loop_dims:]:
            cells *= size if size is not None else 1
        total += cells * np.dtype(var.dtype).itemsize
    return total * row_items


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
    for const_var in contract.const:
        var = space[const_var.name]
        shape = tuple(
            (batches[d][0].stop - batches[d][0].start)
            if d in batches and d not in const_var.protected
            else int(var.sizes[d])
            for d in map(str, var.dims)
        )
        specs.append(ArgSpec(const_var.name, "const", str(var.dtype), shape))
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


def _axis_rows(plan: Plan) -> list[tuple[str, str, int]]:
    """Return (dim, origin, size) for each loop axis."""
    return [
        (a.name, ", ".join(a.carriers) if a.is_zipped else "axis", a.size)
        for a in plan.axes
    ]


def _dedup_summary(plan: Plan) -> str:
    """Return a unique/duplicate summary, or "disabled" when dedup is off.

    Always shown, never omitted: a forgotten dedup on a large map is the
    single most expensive mistake this report can make visible before a
    single call runs.
    """
    if plan.n_unique is None:
        return "disabled"
    duplicates = plan.n_points - plan.n_unique
    noun = "duplicate" if duplicates == 1 else "duplicates"
    return f"{plan.n_unique} unique ({duplicates} {noun})"


def _batch_summaries(plan: Plan) -> list[str]:
    """Return one summary per batched dim: how many batches, and their sizes."""
    lines = []
    for dim, slices in plan.batches.items():
        widths = sorted({sl.stop - sl.start for sl in slices}, reverse=True)
        detail = f"<= {widths[0]}"
        if len(widths) > 1:
            detail += f", last {widths[-1]}"
        lines.append(f"{dim}: {len(slices)} batches ({detail})")
    return lines


def _arg_rows(plan: Plan) -> list[tuple[str, str, str, str]]:
    """Return (name, kind, dtype, shape) for each call argument."""
    rows = []
    for arg in plan.call_signature:
        shape = "" if arg.shape is None else str(list(arg.shape))
        rows.append((arg.name, arg.kind, arg.dtype, shape))
    return rows


def _result_rows(plan: Plan) -> list[tuple[str, str, str]]:
    """Return (name, dims, dtype) for each result variable."""
    rows = []
    for var in plan.result:
        sizes = ", ".join(
            f"{d}: {s if s is not None else '?'}"
            for d, s in zip(var.dims, var.sizes, strict=True)
        )
        rows.append((var.name, sizes, var.dtype))
    return rows


def _store_chunk_note(plan: Plan) -> str | None:
    """Return a note naming any loop_chunks override, or ``None`` if all auto."""
    if plan.store is None:
        return None
    overrides = sorted(plan.policy.loop_chunks)
    if not overrides:
        return None
    return "overridden: " + ", ".join(overrides)


def render(plan: Plan) -> str:
    """Render a plan as a report, using rich tables when available.

    Parameters
    ----------
    plan
        The plan to describe.

    Returns
    -------
    str
        A multi-section report: contract, space, counts, per-call
        arguments, result and store layout. Falls back to plain text when
        rich is not installed, so this never adds a hard dependency.
    """
    try:
        import rich  # noqa: F401
    except ImportError:
        return _render_plain(plan)
    return _render_rich(plan)


def _render_plain(plan: Plan) -> str:
    """Render a plan as plain, aligned text."""
    lines: list[str] = []
    title = f"{plan.name} (v{plan.contract.version})"
    lines.append(f"+- {title} " + "-" * max(0, 58 - len(title)) + "+")
    lines.append(f"| {plan.contract.render()}")
    lines.append("+" + "-" * 60 + "+")
    lines.append("")

    lines.append("SPACE")
    for dim, origin, size in _axis_rows(plan):
        lines.append(f"  {dim:<10} {origin:<15} {size:>6}")
    lines.append(f"  {'points':<26} {plan.n_points:>6}")
    lines.append(f"  {'dedup':<26} {_dedup_summary(plan)}")
    for summary in _batch_summaries(plan):
        lines.append(f"  {'batch':<26} {summary}")
    lines.append("")

    lines.append("CALLS")
    lines.append(f"  {'to compute':<12} {plan.n_to_compute:>6}")
    lines.append(f"  {'cached':<12} {plan.n_cached:>6}")
    lines.append(f"  {'skipped':<12} {plan.n_skipped:>6}")
    lines.append(f"  {'total':<12} {len(plan.work_items):>6}")
    lines.append("")

    lines.append("ARGUMENTS")
    for name, kind, dtype, shape in _arg_rows(plan):
        lines.append(f"  {name:<12} {kind:<7} {dtype:<9} {shape}")
    lines.append("")

    lines.append("RESULT")
    for name, dims, dtype in _result_rows(plan):
        lines.append(f"  {name} ({dims}) {dtype}")
    if not plan.determined:
        lines.append("  sizes marked '?' are undetermined; discovered by a probe call")
    lines.append("")

    if plan.store is not None and plan.store.path is not None:
        lines.append(f"STORE      {plan.store.path}")
        chunks = ", ".join(f"{d}: {n}" for d, n in plan.store.chunks.items())
        lines.append(f"  chunks   {chunks or '1 per point'}")
        note = _store_chunk_note(plan)
        if note is not None:
            lines.append(f"           ({note})")
        lines.append(f"  regions  {plan.store.n_regions}")
    else:
        lines.append("STORE      none (in-memory: no cache, no resume)")
    lines.append(f"EXECUTOR   {plan.executor}")
    return "\n".join(lines)


def _render_rich(plan: Plan) -> str:
    """Render a plan as boxed, aligned tables, using rich."""
    from io import StringIO

    from rich import box
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table

    def section(title: str, *, header: bool = False) -> Table:
        """Start a new section: a bold heading, then a tight, borderless table."""
        console.print(f"[b]{title}[/b]")
        return Table(
            box=box.SIMPLE_HEAD if header else None,
            show_header=header,
            show_edge=False,
            expand=False,
            padding=(0, 1, 0, 0),
        )

    buffer = StringIO()
    console = Console(file=buffer, width=88)

    console.print(
        Panel(
            plan.contract.render(),
            title=f"[b]{plan.name}[/b]",
            title_align="left",
            subtitle=f"v{plan.contract.version}",
            subtitle_align="right",
            box=box.ROUNDED,
        )
    )
    console.print()

    space = section("SPACE")
    space.add_column()
    space.add_column()
    space.add_column(justify="right")
    for dim, origin, size in _axis_rows(plan):
        space.add_row(dim, origin, str(size))
    space.add_row("points", "", str(plan.n_points), style="bold")
    space.add_row("dedup", _dedup_summary(plan), "")
    for summary in _batch_summaries(plan):
        space.add_row("batch", summary, "")
    console.print(space)
    console.print()

    calls = section("CALLS")
    calls.add_column()
    calls.add_column(justify="right")
    calls.add_row("to compute", str(plan.n_to_compute))
    calls.add_row("cached", str(plan.n_cached))
    calls.add_row("skipped", str(plan.n_skipped))
    calls.add_row("total", str(len(plan.work_items)), style="bold")
    console.print(calls)
    console.print()

    args = section("ARGUMENTS", header=True)
    args.add_column("name")
    args.add_column("kind")
    args.add_column("dtype")
    args.add_column("shape")
    for row in _arg_rows(plan):
        args.add_row(*row)
    console.print(args)
    console.print()

    result = section("RESULT", header=True)
    result.add_column("name")
    result.add_column("dims")
    result.add_column("dtype")
    for result_row in _result_rows(plan):
        result.add_row(*result_row)
    console.print(result)
    if not plan.determined:
        console.print(
            "[dim]sizes marked '?' are undetermined; discovered by a probe call[/dim]"
        )
    console.print()

    if plan.store is not None and plan.store.path is not None:
        store = section("STORE")
        store.add_column()
        store.add_column(justify="right")
        store.add_row("path", escape(plan.store.path))
        for dim, width in plan.store.chunks.items():
            store.add_row(f"chunk {dim}", str(width))
        store.add_row("regions", str(plan.store.n_regions))
        console.print(store)
        note = _store_chunk_note(plan)
        if note is not None:
            console.print(f"[dim]{note}[/dim]")
    else:
        console.print("[b]STORE[/b]  none (in-memory: no cache, no resume)")

    console.print(f"[b]EXECUTOR[/b]  {plan.executor}")
    return "\n".join(line.rstrip() for line in buffer.getvalue().split("\n")).strip(
        "\n"
    )
