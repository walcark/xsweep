# Phase 1 data model: xsweep v0

Entities, their fields, the rules that validate them and when those rules
fire. "Definition time" means import or decoration; "planning" means before
any call of the wrapped function; "execution" means during the sweep.

## Contract (frozen, hashable)

Property of the physics. Enters the fingerprint.

| Field | Type | Notes |
|-------|------|-------|
| `loop` | `tuple[LoopVar, ...]` | One value per call |
| `vec` | `tuple[VecVar, ...]` | Whole axis, or batches |
| `const` | `tuple[str, ...]` | Delivered whole to every call |
| `out` | `tuple[OutVar, ...]` | Named outputs with call-level dims |
| `version` | `str` | Cache-invalidation declaration, default `"0"` |

Derived properties: `inputs` (loop + vec + const names), `outputs` (out
names), `out_dims` (union of declared output dims).

**LoopVar**: `name: str`, `deliver: Literal["scalar", "array"] = "scalar"`.

**VecVar**: `name: str`, `max_batch: int | None = None`.

**OutVar**: `name: str`, `dims: tuple[str, ...]`, `sizes: tuple[int | None,
...] = ()` (declared sizes, `None` meaning probe).

### Validation (definition time)

1. No variable name appears in two clauses.
2. No output name duplicates another output name.
3. `max_batch` is set only on a vec variable that is 1-D in the space at
   planning time; syntactically, `@ N` is accepted only on a single name
   (FR-010). A multi-dim vec carrying `@ N` is rejected pointing at the
   policy form.
4. `max_batch >= 1`.
5. A batch marker on a dim absent from every declared output is rejected
   (FR-008, edge case 12): the function reduces over it, so batching would
   corrupt the result.
6. At least one output.
7. Names are valid Python identifiers.

## SweepPolicy (frozen)

Property of the run. Never enters the fingerprint. Ten fields, see
[research.md R1](./research.md). Every field defaults to the `UNSET`
sentinel; `DEFAULTS` holds the resolved values; `resolve()` folds layers in
the order defaults, decorator, instance, call.

### Validation (planning)

1. Dims named in `chunks` and in `dedup` exist in the space; otherwise fail
   listing the available dims (FR-021).
2. `chunks` may only name dims of vec variables; a chunk on a loop dim is a
   different concept and is rejected explicitly.
3. `retries >= 0`; `max_workers` is `None` or `>= 1`.
4. `skip_where` is callable.
5. `executor` is a known name or satisfies the executor protocol.

## Space (an xarray Dataset supplied by the user)

Not an xsweep type. Its rules:

1. Every contract input name exists as a variable (FR-006), otherwise fail
   naming the missing ones and listing what the space does contain.
2. Values are primitives: float, int, str, bool, datetime64 (FR-004,
   principle IX). Object dtype is rejected pointing at the label idiom.
3. Shared dims align exactly; near-identical coordinates fail rather than
   inner-join (FR-003).
4. 0-d variables are accepted and create no result dim.
5. Loop variables are in memory; lazily-backed ones are computed with a
   warning (FR-005).

The sweep space is the union of the dims of the loop variables. Result loop
dims equal that union, whatever the policy (FR-002).

## Plan (frozen)

Produced by planning, consumed by execution, inspectable by the user.

| Field | Type | Notes |
|-------|------|-------|
| `contract` | `Contract` | Including version |
| `policy` | `SweepPolicy` | Fully resolved |
| `loop_dims` | `tuple[str, ...]` | With sizes and product/zip origin |
| `n_points` | `int` | Before dedup |
| `n_unique` | `int \| None` | `None` when dedup is off |
| `batches` | `Mapping[str, tuple[int, ...]]` | Batch sizes per vec dim, last one possibly shorter |
| `n_calls` | `int` | Points times batches |
| `n_cached` | `int` | Points already `ok` in the store |
| `call_signature` | `tuple[ArgSpec, ...]` | Name, kind, dtype, shape per argument, statics included |
| `result` | `tuple[VarSpec, ...]` | Name, dims, dtype, size; sizes may be undetermined |
| `store` | `StoreSpec \| None` | Path, chunk grid, region count |
| `executor` | `str` | Rendered description |
| `duration_hint` | `Duration \| None` | Only from timings recorded by a previous run, provenance stated |
| `work_items` | `tuple[WorkItem, ...]` | The single stream execution consumes (FR-037) |

**WorkItem** (frozen): `point_index: tuple[int, ...]` position in the loop
grid, `unique_index: int | None` representative under dedup, `slices:
Mapping[str, slice]` batch slices per vec dim, `done: bool` already satisfied
by the store, `skipped: bool` excluded by the skip predicate.

Point enumeration, dedup, batching and resume are resolved INTO this stream
during planning. Execution iterates it without conditional branches, so the
interaction between those four mechanisms is decided in exactly one place.

Planning never calls the wrapped function (FR-029, FR-031). Undetermined
output sizes are reported as undetermined, never probed.

## Store

One implementation over two zarr store backends: a directory store for the
persistent mode, the in-memory store for the store-less mode. The store-less
mode skips fingerprint and lock and materialises the result; everything else
is the same code path (research R13).

Operations: `allocate(plan)`, `write_region(index, dataset)`,
`read_status()`, `set_status(index, code)`, `finalise()`, plus `lock()` and
`fingerprint()` for the persistent backend only.

**Sidecars** (persistent backend, JSON inside the store):

- `xsweep-meta.json`: fingerprint, space signature, chunk grid, contract
  rendering, xsweep version.
- `xsweep-lock.json`: pid, hostname, start time, plan digest.

**Status variable**: `uint8` over the loop dims, `0` pending, `1` ok, `2`
failed, `3` skipped; mapping in the attributes.

### Point lifecycle

```
pending ──skip_where true──────────────► skipped
   │
   ├── call raises, retries exhausted, on_error="nan" ──► failed  (NaN slice)
   ├── call raises, on_error="raise" ─────────────────► run aborts
   └── call returns ─────────────────────────────────► ok
```

Resume recomputes everything that is not `ok`: `pending` was never reached,
`failed` deserves another chance, and `skipped` is re-evaluated because the
predicate may have changed.

## Statics

Ordinary keyword arguments, forwarded verbatim, entering the fingerprint.
Must be JSON-serialisable or expose `__cache_token__()`; otherwise planning
fails naming the argument (edge case 7). `space` and `policy` are reserved
names (research R2).

## Sweeper, SweepModule

`Sweeper` binds a contract, a default policy and a callable, and exposes
planning and execution. `@sweep(...)` is sugar over it. `SweepModule`
validates its class-level contract in `__init_subclass__` (with an
`abstract=True` escape hatch), takes a policy at instantiation, orchestrates
in `__call__` and keeps pure physics in `forward`.
