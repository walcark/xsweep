# xsweep, design reference

Status: v0 implemented, 2026-07-29 (design 2026-07-27, open questions
resolved 2026-07-29). This document consolidates the full design
discussion (starting point: analysis of adjeff's internal `sweep` module,
`~/dev/current/adjeff/src/adjeff/sweep`).

This document is the reference for WHY: rationale, worked examples, rejected
alternatives, known limits. The operational WHAT lives in
`specs/001-xsweep-v0/spec.md` (spec-kit), governed by
`.specify/memory/constitution.md`.

## 1. Goal and scope

xsweep lifts a "point function" (typically an expensive radiative-transfer
engine call: uvspec subprocess, Smart-G GPU kernel) into a gridded xarray
computation: Cartesian products and zipped axes, per-call caching, streaming
persistence, resumability, optional parallelism.

It is content-agnostic: it assumes nothing about what the wrapped function
does. It is deliberately minimal: four public objects (`Sweeper`,
`SweepPolicy`, `SweepModule`, the `@sweep` decorator) plus the `Contract`
description. Everything else (pipelines, DAG engines, engine-specific
planners) lives ABOVE it.

Identified consumers (rule of three satisfied):

- adjeff: replaces `SweepBundle` + `UniqueIndex` + `SceneModuleSweep`
  plumbing (see section 12).
- radtrans: is the "generic bookkeeping" half of its planned `core/sweep.py`;
  the per-engine planner (e.g. folding vza into one uvspec `umu` line) stays
  in radtrans.
- Short sensitivity studies in Earth observation.

## 2. Why not pure dask (and prior art)

Dask is an execution engine, not a sweep semantics. It provides none of:

1. Product/zip parameter-space construction with label bookkeeping.
2. Persistent, per-call, cross-run memoisation with resume (the killer
   feature for expensive RT runs).
3. The notion that a callee accepts vectors along SOME dims only (the
   loop/vec/const contract below).
4. A sensible fit for subprocess/GPU callees: a fine-grained graph adds
   nothing to a binary that occupies a whole GPU; a simple executor plus
   region writes gets parallelism and resume with far less machinery.

Therefore: dask is an OPTIONAL backend behind a map-like executor interface,
never the foundation.

Prior art surveyed: `xr.apply_ufunc` / xbatcher (array chunking, no
call-per-point, no cache), hydra multirun / wandb sweeps (config products,
not xarray-aware, no zip), parasweep (simulation sweeps via input-file
templates, no xarray data model), hamilton (name-based DAG wiring, the model
for the future pipeline layer), snakemake/joblib.Memory (per-artifact
caching). The xarray-native "lift a point function into a gridded cached
function" cell is genuinely underserved.

## 3. Core model

### 3.1 The space: semantics come from xarray, not from a DSL

The user builds ONE Dataset describing the whole sweep space. Dim semantics
are xarray's own:

- shared dims = covariance (zip): `aot(y, x)` and `rh(y, x)` vary together;
- distinct dims = Cartesian product: `aot(aot)` x `rh(rh)`;
- the sweep space is the union of the loop variables' dims.

No einops-like description of the inputs: the arrays already encode it.

```python
space = xr.Dataset(
    {
        "aot": ("aot", [0.05, 0.1, 0.3]),
        "rh": ("rh", [30.0, 70.0]),
        "wl": ("wl", np.arange(400.0, 900.0, 5.0)),
        "afgl_type": ("profile", ["afgl_ms", "afgl_t"]),  # str axes are first-class
    }
)
```

0-d variables are accepted (fixed-but-present, e.g. `"sza": ((), 35.0)`);
promoting one to a swept axis is just giving it a dim.

### 3.2 The contract: describes the CALL, not the data

The contract declares, per input variable, how one call consumes it, and
names what one call produces:

```
loop(aot, rh, sza, profile) vec(wl @ 8) const(srf) -> tdir_down(wl)
```

- `loop(x)`: one value per call; its dims (in the space) become loop axes of
  the result. Delivered as native Python scalars by default.
- `vec(x @ N)`: the whole axis (or chunks of max N) in one call; its dim
  appears as a call-output dim. Chunkable by promise (see 4.2).
- `const(x)`: context data (SRF tables, LUTs), passed whole to every call;
  its dims may or may not appear in the call output. If `x` shares a dim
  with a batched `vec` variable it auto-aligns to the active batch instead
  of causing a shape mismatch, unless that dim is declared protected:
  `const(bias(x, y))` keeps `x` and `y` whole on `bias` regardless of what
  the policy batches (added post-v0, see docs/implementation-findings.md).
- `-> name(dims), name2(dims2)`: NAMED outputs with their call-level dims.
  Needed for DAG wiring, store pre-allocation and nominative validation of
  the function's return. There is NO `in` clause: inputs are already
  `loop + vec + const` (a separate clause would duplicate and diverge).
- Statics (`n_ph`, `species`, ...) are ordinary Python kwargs, outside the
  contract: configuration, not data (see 5.3).

Contract clauses name VARIABLES, never sweep dims. Sweep dims are a runtime
property of the space. In Cartesian sweeps variable and dim names coincide
(`aot` on dim `aot`), which hides the distinction; in map sweeps they do not
(`aot` on dims `(y, x)`). The same contract serves both. Asymmetry to keep
documented: parentheses in `loop`/`vec`/`const` contain variable names;
parentheses in the output clause contain DIM names.

### 3.3 Three temporalities

1. **Contract** = property of the physics. Fixed at function/class writing
   time (decorator argument or `contract` ClassVar).
2. **Policy** (`SweepPolicy`: store, chunks, dedup, executor, error policy)
   = property of the run. Fixed at init by the user. Precedence:
   call > instance > decorator default.
3. **Data** (the space + static kwargs) = property of the call.

adjeff already had this split (ClassVar dims vs `__init__` params) without
naming it; xsweep makes it explicit.

## 4. Contract object and string DSL

### 4.1 Object first, string as sugar

```python
@dataclass(frozen=True)
class LoopDim:
    name: str
    deliver: Literal["scalar", "array"] = "scalar"


@dataclass(frozen=True)
class VecDim:
    name: str
    max_batch: int | None = None


@dataclass(frozen=True)
class OutVar:
    name: str
    dims: tuple[str, ...]


@dataclass(frozen=True)
class Contract:
    loop: tuple[LoopDim, ...]
    vec: tuple[VecDim, ...]
    const: tuple[str, ...]
    out: tuple[OutVar, ...]

    @classmethod
    def parse(cls, spec: str) -> "Contract": ...

    # introspection: .inputs, .outputs, .dims
```

Everywhere a contract is expected, `str | Contract` is accepted and
coerced via `parse` (the pydantic ConfigDict spirit). Benefits of the
object: programmatic construction (engine planners), extensibility without
growing the string grammar (per-dim options become fields), typed
introspection for the DAG layer, and frozen => hashable => enters the cache
key (changing the contract invalidates the cache, correctly).

### 4.2 Unified granularity model (the deep invariant)

Chunkability is a property of the (function, dim) pair: "is the function
independent/local along this dim?"

- Elementwise ops are local along their dims: chunk freely.
- An SRF integration is non-local along wl: the whole axis is required;
  chunking it silently corrupts the result.
- A spatial convolution is local-with-neighbourhood: chunkable only with an
  overlap halo (dask map_overlap style). Real taxonomy cell, out of v0.

The three clauses are ONE mechanism at three granularities per input dim:
granularity 1 = `loop` (scalar delivery), granularity N = `vec @ N`,
granularity infinite (whole) = `const` or an unannotated vec dim. `const` ==
"variable all of whose dims are whole". Keep the three keywords at the
surface (they state physical intent: scalar engine / vectorisable axis /
context data); implement one mechanism underneath. Error messages benefit:
"dim wl is declared whole because forward reduces over it; chunking it is
invalid".

Consequences:

- Chunk annotations attach to DIMS, not variables. `vec(wl @ 8)` hid this
  (1-D variable coincides with its dim); per-variable chunk declarations on
  shared dims would be incoherent. RESOLVED 2026-07-29 (see section 11): the
  contract declares WHICH variables are batchable, batch SIZES are dim-keyed
  policy (`chunks={"y": 500, "x": 500}`), and `@ N` survives only as a 1-D
  shorthand. No extra grammar; multi-dim vec comes for free because slicing
  each dim and iterating over the product of slices is the same mechanism at
  any rank (adjeff already does exactly this in `bundle.py:_iter_chunks`,
  with sizes living in `__init__`, i.e. in policy).
- The same data admits different contracts depending on the callee. For
  A(x, y), B(x, y) -> C: `loop(A, B) -> C()` when the callee is a
  scalar-only engine (xsweep iterates, n_unique calls);
  `const(A, B) -> C(y, x)` when the callee is internally vectorised (one
  degenerate call; xsweep then only adds cache/store/DAG uniformity, which
  is legitimate). The contract encodes a property of the PHYSICS, not of the
  data.

### 4.3 Sacred property: policy never changes the result

Result loop dims = union of the dims of the loop variables in the space.
Dedup reduces the CALL COUNT (n_unique instead of the full grid), chunking
splits calls, executors reorder them; the output is identical either way.
To be enforced by dedicated tests (bit-identical results with and without
dedup, with and without chunking).

## 5. API surfaces

### 5.1 Decorator

```python
@sweep(
    "loop(aot, rh, sza) vec(wl @ 8) -> tdir_down(wl)",
    store="runs/tdir.zarr",
    version="1",
)
def tdir_down(
    wl: xr.DataArray, aot: float, rh: float, sza: float, *, n_ph: int
) -> xr.DataArray: ...


result = tdir_down(space, n_ph=int(1e6))
```

`@sweep(...)` is sugar over `Sweeper(contract, policy)(f)`.

### 5.2 Unpacking and delivery

`unpack=True` is the DEFAULT: the function receives named arguments drawn
from the contract (no signature introspection: the contract names
everything). `unpack=False` delivers the raw point Dataset for functions
that want the whole object. Internally the contract is Dataset-in/
Dataset-out: cache, store and executor see one hashable object per call.

Delivery policy: loop variables arrive as native Python scalars (`.item()`:
float/int/str/bool) because the sweeper re-attaches loop coords itself when
placing results, so 0-d coords are useless inside the function, and native
scalars are what external codes want. Per-variable override:
`LoopDim("aot", deliver="array")`. Vec/const variables arrive as DataArrays.

### 5.3 Statics

`f(space, **static)` outside, `forward(...args, **static)` inside, passed
verbatim. Rules:

1. Statics ENTER the cache key (JSON-serialised; or `__cache_token__`
   protocol for objects; explicit exclusion possible). `n_ph` changes MC
   noise, `species` changes physics: non-negotiable.
2. Promotion path: a static that must become swept moves into the space as a
   variable with its own dim; zero change in xsweep, the function reads it
   from its arguments either way.

### 5.4 Return-value normalisation

The contract names outputs, so `forward` may return:

- a bare (even unnamed) DataArray when the contract declares a single output
  (xsweep names it);
- an ordered tuple for multi-output contracts (mapped to declaration order);
- an explicit Dataset (validated nominatively; wrong names = clear error).

This kills the adjeff constraint "func must return an UNNAMED DataArray"
(a combine_by_coords artefact).

### 5.5 Module facade (pytorch-style)

```python
class SweepModule:
    contract: ClassVar[str | Contract]

    def __init_subclass__(cls, abstract: bool = False, **kw) -> None:
        super().__init_subclass__(**kw)
        if abstract:
            return
        spec = getattr(cls, "contract", None)
        if spec is None:
            raise TypeError(...)
        cls._spec = Contract.parse(spec) if isinstance(spec, str) else spec

    def __init__(self, policy: SweepPolicy | None = None): ...
    def __call__(self, space: xr.Dataset, **static) -> xr.Dataset: ...
    def forward(self, ...): raise NotImplementedError
```

`__init_subclass__`, NOT a metaclass: no metaclass conflicts (ABC,
third-party bases), readable, and the contract is parsed AT IMPORT TIME
(syntax errors and incoherences explode at class definition, not after
twenty minutes of runs). `abstract=True` is the escape hatch for
intermediate bases. Contract is inherited and overridable. `__call__` does
orchestration; `forward` stays pure physics, testable alone.

## 6. Execution

### 6.1 Store: cache == output == resume

Pre-allocated zarr, one region write per call, restart skips
already-written regions. Unifies caching, streaming persistence and
resumability; fixes the accumulate-then-combine memory profile of adjeff.

- OUTPUT IS LAZY BY DEFAULT: `mod(space)` returns `xr.open_zarr(store)`
  (lazy, chunked); `.load()` is the user's choice. Memory bounded by one
  chunk regardless of sweep size.
- Execution has exactly one writer (the parent process collecting
  outcomes), so the store's loop-dim chunk grid is sized from a memory
  budget rather than pinned to one point per chunk. The original "one loop
  point = one chunk, for safe parallel writers" rationale did not describe
  this architecture, since there is only ever one writer; see
  docs/implementation-findings.md. Execution buffers outcomes per chunk and
  flushes each chunk once, which is what makes a wider grid a pure win
  rather than a read-modify-write per point.
- A sidecar `status` variable (ok/failed/skipped) accompanies the data,
  otherwise resume and failure are indistinguishable.
- Store pre-allocation needs output sizes: declared in the contract or
  discovered by a probe call (first point).

### 6.2 Cache key

hash(point values) + hash(statics) + contract hash + user-declared
`version` + engine/data identifiers when applicable. Function bytecode
hashing is rejected as fragile; the user increments `version="1"` instead.

### 6.3 Executor

Pluggable behind a map-like interface: serial, ProcessPoolExecutor, dask
distributed as an option. GPU engines typically want serial-per-device.

### 6.4 Error policy

Default: NaN slice + status=failed + continue; global fail as option; retry
count. Skip policy for invalid points (`skip_where=` or automatic NaN-skip
on loop values).

### 6.5 Dedup

UniqueIndex-style (from adjeff, kept as opt-in policy): collapse loop
variables to unique rows, sweep uniques, re-expand. `dedup=True` = dedup
over all loop dims (the common map case, no dim naming needed);
`dedup=("y", "x")` for partial dedup (e.g. dedup space, keep time). Must be
fixed for mixed dtypes (np.unique on object arrays breaks). Policy
validation happens against the space at runtime: unknown dims in `dedup=` or
`chunks=` fail immediately, listing available dims.

## 7. DAG positioning

xsweep ENABLES composition but does not implement a pipeline engine (scope
discipline; hamilton/snakemake territory):

- edges are Datasets: multi-input convergence is `xr.merge` + a call;
- contracts are introspectable (`.inputs`/`.outputs`): a thin pipeline layer
  can statically check "each consumed variable has exactly one producer or
  comes from the initial space, no cycles, no duplicate producers";
- per-module caching makes DAG re-runs incremental for free.

Two documented limits of DAG checking:

1. **Full output dims are only known at runtime.** Statically, a node gives
   output NAMES and call-level dims (`tdir_down(wl)`); the final dims (loop
   prefix: `aot, rh, ...` or `y, x` after dedup) depend on the space. So:
   static check on names, complete shape/dims validation pass at runtime
   before launching computations.
2. **Undeclared passthrough would be invisible to the check.** Hence `const`
   is part of the contract ("everything that flows is named"). Anything
   still implicit stays outside DAG guarantees.

DAGs will mix swept nodes and plain xarray post-processing functions (e.g.
the 3-albedo Lambertian inversion consuming the `albedo` dim); the check
must tolerate non-sweep nodes.

## 8. Findings from ten worked examples

Ten cases were written against the formalism: Cartesian sensitivity, pixel
map + dedup, map x diurnal axis, multi-geometry vec (`vec(wl @ 8, vza,
phi)`, the declarative version of radtrans' umu-folding planner), MC
convergence, band integration with SRF, 3-albedo inversion, parametrised
profiles, cross-engine comparison, dask-backed scene. Seven survived
unchanged; the findings:

1. **Replication dims need a carrier variable.** A bare "rep" dim is not
   loopable (contracts loop over variables). Official idiom: a `seed`
   variable on dim "rep": makes the 20 repetitions distinct (identical calls
   would be legitimately collapsed by cache/dedup, std would be 0), each
   individually reproducible and cacheable. MC noise = `out.std("rep")`.
2. **`const` is required in v0**: band integration needs `srf(band, wl)`
   passed whole to every call.
3. **Batching a reduced dim must be rejected at validation time**: if a vec
   dim is absent from the declared output (the function reduces over it),
   any `@ N` on it silently corrupts results.
4. **Dependent/ragged axes are out of the model**: a wl support that depends
   on the band does not fit rectangular axes. Workaround: union grid + zero
   weights in the SRF (physically clean). Documented limitation.
5. **Sweep values are primitives only** (float, int, str, bool, datetime64).
   Object-valued loop variables break cache hashing, zarr coord
   serialisation and dedup. Idiom: label string + static mapping, the
   mapping entering the cache key via `__cache_token__` (otherwise mutating
   the mapped object would not invalidate the cache).
6. **Instance-level output renaming is needed** (e.g.
   `SweepPolicy(rename={"rho_atm": "rho_atm_smartg"})`): two engine modules
   produce the same contract-fixed name, breaking both the single-producer
   DAG check and the cross-engine comparison use case (a central one).
   Design now, implement with the DAG layer.
7. **Only the LOOP variables are assumed in-memory in v0.** Output side is
   lazy by design (6.1). Input side: nothing structural forbids streaming
   (blockwise two-pass unique for dedup, LUT-gather expand as lazy
   map_blocks); deferred as implementation debt, no current design decision
   blocks it. v0 rule: loop variables loaded in memory (two 10980^2 float64
   maps are ~1.9 GB, tolerable); dask-backed loop variables computed
   explicitly with a warning.

## 9. Edge-case table

| # | Case | Symptom if unhandled | v0 policy | Later |
|---|------|----------------------|-----------|-------|
| 1 | Slightly different coords on a zip dim | xarray inner join silently drops points | Validate `join="exact"`, fail loudly | |
| 2 | Heterogeneous output grids between calls (z levels vary with profile) | Region writes corrupt / misalign | Detect, clear error | Declared union grid + NaN padding |
| 3 | Output dims of unknown size before first call | Store cannot be pre-allocated | Probe call, or sizes declared in contract | |
| 4 | Invalid points (masked pixels, NaN axes, absurd combos) | Wasted or crashing engine calls | `skip_where=` / NaN-skip + `status` var | |
| 5 | Call failure mid-sweep | One failure kills the sweep; resume ambiguous | NaN slice + status=failed + continue; optional global fail; retry count | |
| 6 | Store chunk grid finer than what execution flushes | A read-modify-write per point instead of per chunk | Size loop-dim chunks from a memory budget, buffer writes per chunk (see docs/implementation-findings.md) | |
| 7 | Non-serialisable statics | Cache key impossible | `__cache_token__` protocol or explicit exclusion | |
| 8 | Function code changes | Stale cache reused | User-declared `version=` in decorator | |
| 9 | Resume with modified space (axis extended) | Silent misalignment with store | Detect mismatch, refuse with clear message | Reindex store on pure extension |
| 10 | Non-numeric axes (str, datetime64) | float coercion / np.unique break (adjeff `bundle.py:175`, `_dedup.py`) | First-class support, tested | |
| 11 | Last chunk smaller than max batch | Function assuming fixed size breaks | Documented; golden test | |
| 12 | Chunking a dim the function reduces over | Silently wrong integrals | Contract validation: no `@ N` on dims absent from output | |
| 13 | Duplicate values on a loop axis | Redundant expensive calls | Opt-in dedup | |
| 14 | Replication without a carrier variable | `loop(rep)` impossible, user confusion | Document seed-variable idiom | Reserved seed kwarg helper |
| 15 | Object-valued sweep coordinates | Hash/serialise/unique failures | Enforce primitives, clear error, label idiom documented | |
| 16 | Same output name from two instances | DAG single-producer check breaks | Documented; rename designed | `SweepPolicy(rename=...)` |
| 17 | Dask-backed loop variables | Uncontrolled materialisation | Explicit compute + warning | Streaming dedup + blockwise iteration |
| 18 | MC reproducibility | Cache honesty, per-point replay | Seed variable idiom | Per-point derived-seed helper |

## 10. v0 scope

In: strict alignment, error policy + status variable, probe call, complete
cache key, native-scalar delivery, dedup opt-in (`True` or dim tuple), zarr
region store with lazy-open return, serial + process executors, `const`
clause, reduced-dim batching rejection, primitive-only sweep values
enforcement, contract validation at import (modules) / decoration
(functions).

Out (documented limitations): union grids, store reindexing on space change,
seed helper, dask executor and dask-backed spaces, streaming dedup,
instance-level output rename (designed, implemented with the DAG layer),
halo/map_overlap granularity.

## 11. Open questions

Resolved on 2026-07-29 while writing `specs/001-xsweep-v0/spec.md`, which is
now the normative record:

- **DSL keyword vocabulary**: FIXED to `loop` / `vec` / `const`, batch
  marker `@ N`, output clause `->`. (`pass` being a Python keyword is why
  the clause is named `vec` and not `pass`.) `batch` was considered for
  `vec` and rejected: it names the mechanism, whereas `vec` names the
  physical property that justifies it (the callee accepts a vector along
  this axis). Likewise `whole` was considered for `const` and rejected: it
  is the granularity, not the intent (context data).
- **Multi-dim vec**: ADOPTED, without new grammar. The contract declares
  batchability, policy declares dim-keyed sizes, `@ N` remains a 1-D
  shorthand and is rejected on multi-dim vec variables as ambiguous. See
  4.2. Consequence: `@ N` in a contract is a physics-informed DEFAULT,
  overridable by policy, which is coherent with the sacred property (batch
  size never changes the result on a vec dim, by definition of vec).
- **Store layout**: one write region per call, aligned on store chunk
  boundaries. The chunk grid itself was pinned to one loop point per chunk in
  v0; post-v0 it is sized from a memory budget instead, with execution
  buffering writes per chunk (see docs/implementation-findings.md). A store belongs to one sweep
  configuration in v0: it carries a fingerprint of (contract, version,
  statics) and refuses to open under a different one. Sharing a store
  between instances is deferred with the output-rename feature.

Four further decisions taken on 2026-07-29 during clarification, recorded in
FR-018, FR-036, FR-027, FR-028:

- **`version` belongs to the contract**, not to the policy. It is the manual
  declaration that the computation changed, covering what the system cannot
  observe: the function body, the external engine binary, external data
  files. Bytecode hashing stays rejected because it is wrong in both
  directions (renaming a local or upgrading Python throws away days of
  cache; an upgraded uvspec binary is invisible to it). Putting it in the
  policy would let a run-time setting select a different cached result,
  violating the sacred property. Comparing two versions is a store-level
  workflow (one store per version, explicit concat), never a sweep axis:
  version selects code, not data.
- **Target scale: about 1e4 loop points with callees costing seconds to
  minutes.** At v0's one-loop-point-per-chunk grid this made bookkeeping
  free (under 1% overhead), and the regime is self-limiting: 1e6 points at
  ten seconds each would run for months, so "very many points" and
  "expensive callee" are mutually exclusive in practice. Large maps reach the
  target range through dedup. Fast callees are NOT excluded: a fast function
  is usually vectorisable, hence `vec` (one call for a whole axis, not a
  million loop points), and the store-less in-memory mode removes the
  per-point write cost entirely. The conjunction that stayed a non-goal
  under v0's grid (cheap + scalar-only + very many points + persistence) is
  addressed post-v0 by coalescing regions into memory-budgeted chunks,
  buffered by execution rather than the store layout (see docs/implementation-findings.md).
- **No automatic batch sizing for `vec`/call dims.** Still true post-v0: the
  idea splits in two, a memory guard valuable in every regime and throughput
  tuning that only pays off at scale, and timing-based sizing is rejected
  outright (the first call always lies because of engine warm-up, and
  compute time is not linear in batch size). A later deterministic form
  (declared memory budget plus the output size measured by the probe call)
  must decide once, persist the choice in the store and reuse it on resume,
  because batch size determines the store's call-dim chunk width: two runs
  choosing different sizes would produce incompatible grids. Note the
  library can only bound its own buffers, never the callee's internal
  allocation. This is distinct from the loop-dim chunk width, which post-v0
  IS sized automatically from a memory budget (see docs/implementation-findings.md): a loop dim
  carries no callee-facing batch, so choosing its width is purely an
  execution/store concern with no callee semantics at stake.
- **Observability: standard-library structured logging only.** No progress
  callback in v0, because the status variable already makes progress
  inspectable from the store, and a callback would grow both the public
  surface and the policy field list (principle II).
- **One writing run per store, enforced by a lock**, with a documented
  escape for locks left by dead runs. The lock binds writers only: reading
  the store while a sweep runs is the supported way to follow progress.
  Crucially it does NOT constrain CPU parallelism, which happens inside one
  process pool under one lock; it only forbids several independent runs
  targeting one store. For that case the documented idiom is to partition
  the space at submission (one store per array task) and merge afterwards:
  deterministic, no coordination, and how most cluster sweeps are launched
  anyway. Cooperative multi-writer stores are the natural v1 extension, but
  the hard part is the filesystem, not the concept: atomic claims are
  feasible, expiry and orphan recovery are where such systems break, and
  duplicate work cannot be dismissed as benign since two Monte-Carlo calls
  on the same point produce different values.

- **Planning is a phase, and the plan is inspectable** (FR-029 to FR-031).
  The dominant failure mode of this library is not a crash, it is launching
  a three-day run that was misconfigured. Showing the resolved plan before
  execution catches the three expensive mistakes at zero cost: dedup
  forgotten on a map sweep (millions of calls instead of thousands), an
  accidental Cartesian product where a zip was intended (the dims were not
  actually shared), and a batch size of one on a vectorisable axis. The
  right mental model is a database EXPLAIN or snakemake's dry run rather
  than dask's graph view. This is not a bolted-on feature: FR-024 already
  requires validating policy against the space before any expensive call,
  so a planning phase must exist; exposing its artefact costs one method on
  existing objects, not a fifth public object. Honesty rules matter more
  than completeness here: never call the function to build the plan (an
  unknown output size is reported as undetermined, not probed), admit that
  counting unique points is real work on large spaces, and never invent a
  duration (report call counts; a duration only when derived from timings
  recorded by a previous run on the same store, with that provenance
  stated). Rendering is deliberately left open.

Still open, deferred to planning:

- Exact `SweepPolicy` field list and precedence mechanics.
- Package layout, pixi setup (same conventions as radtrans: src/ layout,
  py311, ruff/mypy/pytest).

Noted for later, not v0: type annotations partially duplicate the `loop`
(scalar) vs `vec`/`const` (array) distinction. They can never replace the
contract, which additionally carries vec-vs-const (same annotation, opposite
semantics), swept-vs-static (same annotation), output names and call-level
dims (absent from signatures), batch sizes and delivery overrides, and which
must be a hashable, introspectable, programmatically constructible VALUE
(cache key, DAG wiring, engine planners) rather than signature metadata.
Signature introspection is also the source of adjeff's silent kwarg
filtering (`bundle.py:196-201`). The redundancy is usable in the safe
direction only: a decoration-time cross-check when annotations exist.

## 12. adjeff migration sketch

`SceneModuleSweep` becomes a subclass of `SweepModule`: `_get_configs`
becomes "build the space from configs", `_compute` keeps the ImageDict
adaptation at the edges (ImageDict stays an adjeff concept). The
`scalar_dims`/`vector_dims`/`output_vars` ClassVars are absorbed into
`contract`. `_smartg.py` signatures survive unchanged under `unpack=True`
(or with a one-line shim). Known adjeff defects fixed by construction:
float-only axes, unnamed-DataArray constraint, accumulate-then-combine
memory profile, no per-call cache/resume, one failure killing the sweep.

## 13. Relation to radtrans

radtrans phase 2 planned "core/sweep.py: generic bookkeeping + libradtran
planner". The generic bookkeeping half IS xsweep; radtrans keeps the
planner: engine-aware contract construction (programmatic `Contract`), e.g.
folding the vza axis into a single uvspec `umu` line via `vec(vza, phi)`,
wavelength batching via `vec(wl @ N)` aligned with `wavelength_grid_file`
LUT strategies.

**Status as of 2026-07-29: xsweep has reached v0**, so radtrans no longer
needs to plan a generic bookkeeping layer. Its `core/sweep.py` becomes a thin
planner producing `Contract` objects, and the ROADMAP entry for phase 2
should be rewritten accordingly. That change belongs to the radtrans
repository and is left to its own commit.

What a radtrans planner will build, concretely:

```python
Contract(
    loop=(LoopVar("aot"), LoopVar("rh")),
    vec=(VecVar("wl", max_batch=n_per_lut), VecVar("vza"), VecVar("phi")),
    out=(OutVar("radiance", ("wl", "vza", "phi")),),
    version=uvspec_version,
)
```

Everything engine-aware stays there: which axes uvspec can fold into one
invocation, how `wavelength_grid_file` bounds the batch size, and which
binary version the results belong to. xsweep sees none of it, which is the
whole point of the split.
