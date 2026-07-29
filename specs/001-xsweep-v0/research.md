# Phase 0 research: xsweep v0

Fifteen decisions, each with the alternatives that were rejected. The two
questions the planning step was explicitly asked to settle are R1/R2
(SweepPolicy fields and precedence) and R3 (NaN-skip default).

## R1. SweepPolicy field list

**Decision**: ten fields, all run-level, none physical.

| Field | Type | Resolved default | Purpose |
|-------|------|------------------|---------|
| `store` | `str \| Path \| None` | `None` | Store location; `None` selects the in-memory mode |
| `chunks` | `Mapping[str, int]` | `{}` | Dim-keyed batch sizes, overriding contract `@ N` |
| `dedup` | `bool \| tuple[str, ...]` | `False` | All loop dims, or the named subset |
| `executor` | `str \| Executor` | `"serial"` | `"serial"`, `"process"`, or an object satisfying the protocol |
| `max_workers` | `int \| None` | `None` | Process executor only; `None` means the runtime default |
| `on_error` | `"nan" \| "raise"` | `"nan"` | NaN slice and continue, or fail fast |
| `retries` | `int` | `0` | Retries per point before it is marked failed |
| `skip_where` | `Callable[[Mapping[str, Any]], bool] \| None` | `None` | Predicate marking a point skipped without calling |
| `load` | `bool` | `False` | Materialise the returned handle instead of returning it lazily |
| `force_unlock` | `bool` | `False` | Break a lock left by a dead run |

**Rationale**: every field changes cost, reporting or location, never a
value. `version` is deliberately absent: it lives in `Contract` (FR-018),
because a run-level setting that selected a different cached result would
break principle III. `rename` is absent: designed, deferred with the DAG
layer (FR-026). No `progress` callback: decided against in clarification
(FR-027). No `auto_chunks`: decided against (FR-035).

**Consequence to reconcile in the spec**: `store=None` as the resolved
default means a sweep with no store named runs in memory. FR-034 currently
says the persistent mode "stays the default", which reads as contradictory.
The accurate statement is that persistence is engaged by naming a store, and
that the in-memory mode is never silent: the run summary states that results
were not persisted and no cache was consulted.

**Alternatives rejected**: making `store` mandatory so persistence is the
literal default (rejected: it makes every test and every throwaway
exploration ceremonial, for a guarantee the log line already provides);
folding `max_workers` into the executor string (`"process:8"`) (rejected:
stringly-typed configuration).

## R2. Precedence mechanics

**Decision**: every `SweepPolicy` field defaults to a module-private `UNSET`
sentinel. A module-level `DEFAULTS` holds the resolved values of R1.
Resolution folds layers left to right: `resolve(DEFAULTS, decorator_policy,
instance_policy, call_policy)`, each layer overriding only the fields it set.
Call-time overrides are passed through an explicit `policy=` keyword.

**Rationale**: the sentinel is what distinguishes "not specified" from
"specified as `False`" or `0`, which plain defaults cannot do and which
`dataclasses.replace` alone does not solve. The explicit `policy=` keyword is
required because statics already occupy the keyword namespace: `f(space,
dedup=True)` would be ambiguous with a physics parameter named `dedup`.

**Consequence**: `space` and `policy` become reserved argument names; a
static using either MUST be rejected with a message naming the collision.

**Alternatives rejected**: policy fields as loose keyword arguments
(rejected: collides with statics, which are arbitrary by design); a mutable
global default policy (rejected: action at a distance, and it would make
results depend on import order).

## R3. Automatic NaN-skip default

**Decision**: off. Skipping is opt-in through `skip_where`. No automatic
behaviour keyed on NaN values in loop variables.

**Rationale**: an unexpected NaN in a loop variable is more often a bug in
space construction than a deliberate mask. Silently skipping it would hide
the bug and produce a result full of unexplained NaN slices, which is exactly
the silent-failure class principle VII exists to prevent. When masking is
intended, it is one line and it is visible in the code and in the plan:
`skip_where=lambda p: math.isnan(p["aot"])`.

**Alternatives rejected**: on by default (rejected above); a `skip_nan=`
convenience field (rejected: a second way to express what `skip_where` covers,
against principle II; the idiom is documented instead).

## R4. Dedup across mixed dtypes

**Decision**: factorise per column, then find unique rows on the integer
codes. For each loop variable, `np.unique(column, return_inverse=True)` gives
codes within a homogeneous dtype; stack the code columns into an `(n, k)`
int64 matrix and run `np.unique(matrix, axis=0, return_inverse=True)` on it.

**Rationale**: this never builds an object array, which is where adjeff's
`_dedup.py` breaks (edge case 10). Strings, datetimes and floats each get
factorised in their own dtype, and the row comparison happens on integers.

**Caveats to test**: NaN handling follows numpy's unique semantics and must
be pinned by a test; coordinate labels must be stripped before broadcasting,
as adjeff learned, otherwise xarray aligns by label instead of position.

**Alternatives rejected**: structured arrays (workable but the dtype
construction is fiddly and error messages become opaque); pandas
`MultiIndex.unique` (a heavy dependency for one function); hashing rows
(collisions would silently merge distinct points).

## R5. Cache key realisation

**Decision**: the cache is not a hash-keyed side table. It is the store
itself, identified by a fingerprint and addressed by coordinates. The
fingerprint is `blake2b` over canonical JSON of the contract (version
included) and the canonicalised statics. Point identity comes from the stored
coordinates plus the status variable: a point is already computed when its
status reads `ok` at its position, which is why alignment must be exact
(FR-003) and why a changed space is refused (FR-017).

**Rationale**: this is what makes cache, output and resume literally the same
artefact, and it removes an entire class of bugs where a key table and a data
store disagree. Per-point content hashing would only be needed for a
content-addressed cache shared across stores, which is not v0.

**Statics canonicalisation**: `json.dumps(sort_keys=True)` with a hook that
calls `__cache_token__()` when present. A static that is neither
JSON-serialisable nor token-bearing raises at planning time, naming the
offending argument (edge case 7).

**Alternatives rejected**: hashing function bytecode (rejected in design,
wrong in both directions); a sidecar key table (a second source of truth).

## R6. Store layout

**Decision**: zarr v3 directory store. Planning computes the full result
shape, allocates metadata only (`to_zarr(compute=False)`), then execution
writes one region per loop point. The chunk grid is 1 along every loop dim,
and along each call-output dim it is the batch size when the dim is batched,
the full extent otherwise. Two sidecar JSON files live inside the store:
`xsweep-meta.json` (fingerprint, space signature, chunk grid, contract
rendering, xsweep version) and `xsweep-lock.json` (R7).

**Rationale**: a chunk grid of 1 along loop dims is what makes concurrent
region writes safe without coordination (edge case 6): two workers can never
touch the same chunk because they never share a loop point.

**Amended at T005, verified against zarr 3.2.1 and xarray 2026.7.0**:
allocation MUST go through the zarr API (`zarr.open_group` then
`create_array` with `dimension_names` and a `fill_value`), not through
xarray's `to_zarr(compute=False)`. The xarray path returns a delayed object
and therefore imports dask, which would make an optional dependency
mandatory and violate principle X. Two consequences, both verified:
`ds.chunk()` is equally unusable for the same reason, so chunk sizes are
declared at array creation; and `zarr.consolidate_metadata(store)` must be
called after allocation, otherwise every open emits a warning and falls back
to a slower non-consolidated read. Region writes themselves go through
xarray's `to_zarr(store, region=...)`, which needs no dask, and
`xr.open_zarr(store, chunks=None)` returns a lazily-indexed Dataset with no
dask involved (research R17).

**Alternatives rejected**: chunking several loop points together (better
write throughput, but it coarsens resume granularity and couples the executor
to the store layout; explicitly deferred, see spec Assumptions); storing
metadata in zarr attributes only (workable, but a plain JSON file is
readable, diffable and greppable without opening the store).

## R7. Lock

**Decision**: `xsweep-lock.json` created with `O_EXCL`, holding pid,
hostname, start time and the plan digest. Acquired by a context manager that
releases on normal exit and on SIGINT/SIGTERM. A pre-existing lock raises
`StoreLockedError` naming its owner; `force_unlock=True` overrides it.

**Rationale**: `O_EXCL` creation is atomic on local filesystems, which is the
v0 target. Writers only: readers never take the lock (FR-033).

**Explicitly not done**: heartbeats and automatic staleness detection. A dead
process on another host cannot be probed reliably, and a wrong guess either
blocks a legitimate run or lets two writers proceed. Manual override is
honest and one flag away.

## R8. Status encoding

**Decision**: `uint8` variable over the loop dims, with `0` pending, `1` ok,
`2` failed, `3` skipped. The mapping lives in the variable's attributes.

**Rationale**: compact, cheap to scan for progress and resume, and
pre-allocation naturally initialises to pending.

**Alternatives rejected**: fixed-width strings (wasteful and awkward to
compare); a boolean plus a separate failure mask (two variables that can
disagree).

## R9. Probe call

**Decision**: when output sizes are not declared in the contract, execution
runs the first non-skipped point as a probe, derives the shapes, allocates
the store, and KEEPS the result (writes it as a normal point). Planning never
probes: an undetermined output shape is reported as such (FR-031). If every
point is skipped, no probe happens and the run fails with a clear message
unless sizes were declared.

**Rationale**: the callee costs minutes, so discarding a probe result would
be indefensible. The plan/execute split makes this natural: planning reports
"shape undetermined, will be probed", execution resolves it.

## R10. Contract parser

**Decision**: hand-written tokeniser plus recursive-descent parser, around
150 lines, no dependency. Errors carry the character offset and the expected
token.

**Rationale**: the grammar has five clause keywords and three punctuation
tokens. A parsing library would be a dependency and an indirection for less
code than it costs, and the error messages matter more here than the parsing
(this is the surface users get wrong first).

**Alternatives rejected**: lark or pyparsing (dependency, worse control over
messages); regular expressions only (fragile, and offsets in errors become
guesswork).

## R11. Executor protocol

**Decision**: a `Protocol` exposing `map_unordered(fn, items) ->
Iterator[tuple[int, Outcome]]`, so results stream back as they complete and
each carries its point index. Implementations in v0: serial and
`ProcessPoolExecutor`. Dask lives behind a lazy import in a factory and is
never imported by core paths.

**Consequence**: with the process executor, the wrapped callable and the
statics must be picklable, which in practice means module-level functions.
This is documented, and a pickling failure is caught and re-raised naming the
offending object rather than surfacing as an opaque `PicklingError`.

**Rationale**: unordered streaming keeps memory flat and lets slow points
overlap with writes. GPU callees run serial per device, which the serial
executor already gives.

## R12. Plan artefact

**Decision**: `Plan` is a frozen dataclass holding exactly what FR-030
requires, produced by the planning phase and consumed by execution. Its
`__repr__` renders the textual report. No HTML representation in v0.

**Rationale**: making the plan the input of execution, rather than a report
printed alongside it, is what guarantees the plan cannot drift from what
actually runs.

## R13. Store-less mode

**Decision**: no second backend implementation. The store-less mode uses
zarr's in-memory store, so both modes run the same code path, skipping
fingerprint and lock and materialising the result at the end.

**Rationale**: a dedicated backend would be roughly 150 lines duplicating
`store.py`, and every later store feature (coalesced regions, per-version
namespacing, cooperative writers) would have to be written twice or the
memory path would silently lag. Sharing one path also makes the equivalence
between the two modes true by construction, which matters because the sacred
property suite compares them.

**Accepted cost**: data passes through the array library's chunk encoding
(with compression disabled) instead of a direct in-memory assignment, so a
write costs microseconds rather than hundreds of nanoseconds. Against the
millisecond of a filesystem write, the store-less mode is still two to three
orders of magnitude cheaper, so the justification for FR-034 holds. If
profiling later shows the encoding dominates, a fast path can be added behind
the same interface without breaking anything.

**No dask implication**: the in-memory store is a key-value backend,
orthogonal to dask. Dask enters only through the optional executor or
through chunked parallel reads of a result; without it, a returned handle is
lazily indexed by xarray and the store-less mode returns plain arrays.

**To confirm at T00x**: the exact in-memory store class name and the way it
is handed to `to_zarr` in the pinned zarr version. No environment with zarr
was available when this plan was written.

## R16. One stream of work items

**Decision**: planning emits a single list of work items, each carrying its
point index, its unique representative under dedup, its batch slices and
whether the store already satisfies it. Execution iterates that list with no
conditional branches.

**Rationale**: point enumeration, dedup, batching and resume are four
transformations of the same stream. Implemented independently and combined at
run time, they produce cross-cutting conditionals and eight interaction cases
(dedup times batching times resume) to write and to test. Resolved once
during planning, they collapse to one loop and one place where their
interaction is decided. It also makes the plan literally what runs, which is
the guarantee FR-029 to FR-031 rest on.

**Alternatives rejected**: four independent passes (each piece reads in
isolation, but the interaction cost lands in the execution loop); partial
unification of dedup and resume only (halves the problem and keeps the
asymmetry between filters and multipliers, for no simplification of the
final loop).

## R17. Lazy does not mean dask

**Decision**: the persistent mode returns a lazily-opened handle whether or
not dask is installed. Without dask, xarray's own lazy indexing backs it;
with dask, reads become chunked and parallel.

**Rationale**: FR-014 promises laziness and constitution principle X forbids
depending on dask. Both hold, but the two situations differ in what a user
gets, and stating it prevents "lazy" from being read as "dask-backed".

## R14. Error hierarchy

**Decision**: one base `XsweepError`, with `ContractError`, `SpaceError`,
`PolicyError`, `StoreError`, `StoreLockedError` and `PointFailed`. Every
message names what to change, not only what is wrong.

**Rationale**: users catch one base. The subclasses map to the phase that
raised, which is also the phase the user must fix: definition, space
construction, policy, or run.

## R15. Tooling deltas from radtrans

**Decision**: same `pyproject.toml` shape, same pixi tasks, same ruff
configuration (line 88, target py311, rules E/F/I/UP/B), dev environment
pinned to 3.12 with the library targeting 3.11. Two deliberate differences:
mypy `strict = true` from the first commit, and no pydantic.

**Rationale**: strictness is cheap on an empty codebase and expensive to
retrofit. Pydantic would add a dependency and a second modelling idiom for
objects that are frozen dataclasses with a handful of invariants; the
validation that matters here is semantic (does this dim exist in the space)
and no schema library performs it.
