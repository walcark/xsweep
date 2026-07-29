# Feature Specification: xsweep v0

**Feature Directory**: `specs/001-xsweep-v0`

**Created**: 2026-07-29

**Status**: Draft

**Input**: User description: "Specify xsweep v0, a minimal content-agnostic
parameter-sweep library for xarray. The complete design already exists in
docs/design/xsweep.md: read it first, it is the source of truth for rationale
and scope. The spec must cover exactly the v0 scope of its section 10.
Incorporate the 18-row edge-case table of section 9 as acceptance criteria
with their v0 policies. Incorporate the seven findings of section 8. Do not
re-litigate decisions already made in the doc; mark as [NEEDS CLARIFICATION]
only the open questions of section 11."

**Design reference (WHY)**: `docs/design/xsweep.md`. This spec is the
operational WHAT; decisions recorded there are not re-litigated here.

## Clarifications

### Session 2026-07-29

- Q: Where does the cache-invalidation `version` string live? → A: A field
  of the contract (physics), set through the decorator keyword or the
  class-level declaration, not part of the string grammar and never a
  policy field.
- Q: What loop-point scale must v0 sustain? → A: The persistent mode is
  calibrated for around 1e4 loop points with expensive callees (seconds to
  minutes per call); larger grids reach that range through dedup. Cheap
  callees are served either by the `vec` clause (one call for a whole axis
  instead of a million loop points) or by the store-less in-memory mode;
  what stays a non-goal is the conjunction cheap + scalar-only + very many
  points + persistence.
- Q: What progress signal ships in v0? → A: Structured standard-library
  logging only (per-point debug events, run summary at info); no progress
  callback; the status variable already makes progress inspectable.
- Q: What happens when two processes write the same store concurrently? →
  A: A lock marks the store as being written; a second process refuses to
  start with a clear message, with a documented escape for stale locks. The
  lock constrains writers only and does not affect single-node CPU
  parallelism, which runs inside one process pool under one lock; the
  documented answer for cluster array jobs is to partition the space at
  submission and merge one store per task.
- Q: Should batch sizes be picked automatically when the user gives none? →
  A: No automatic mode in v0. Explicit sizes only (contract default or
  policy), an unannotated vec dim meaning the whole axis. Auto sizing is
  designed for later in a deterministic, memory-budget form, never from
  call timings.
- Q: Does a mode without persistence exist? → A: Yes, a store-less
  in-memory mode returning materialised results, without cache or resume:
  the mode for cheap callees, small sweeps and the test suite.
- Q: Can a user see how a sweep will execute before launching it? → A: Yes,
  planning is a distinct phase exposed as an inspectable plan (call counts,
  batching, cache state, result and store shapes, per-call arguments),
  produced without ever calling the wrapped function. Its rendering is left
  to implementation.
- Q: Does v0 ship the raw per-call Dataset delivery opt-out? → A: No. No
  identified consumer needs it and the contract already names every
  argument; adding it later is backward compatible.
- Q: Is the plan object part of the public surface? → A: Yes, exported. It
  is a frozen data holder rather than machinery, and returning a type the
  user cannot import to annotate would be a defect.
- Q: How is the store-less mode implemented? → A: Through the array
  library's in-memory store, so both modes share one code path minus
  fingerprint and lock, rather than through a second backend implementation.
- Q: How do point enumeration, dedup, batching and resume compose? → A:
  Planning emits a single stream of work items, each carrying its point
  index, unique representative, batch slices and already-done state, so
  execution is a loop with no branching and the combinations are resolved
  once, at planning.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Lift a point function into a cached gridded sweep (Priority: P1)

A scientist has an expensive point function (a radiative-transfer engine
call: subprocess or GPU kernel) computing one result for one combination of
parameter values. They decorate it with a contract string, build one xarray
Dataset describing the parameter space (distinct dims = Cartesian product,
shared dims = zipped/covarying axes), and call the decorated function on that
space. They get back a lazily-opened result with one labelled axis per swept
dim, computed once per point, persisted as it streams.

**Why this priority**: this is the core lift; every other capability
decorates it. Without it there is no library.

**Independent Test**: decorate a cheap synthetic function with
`loop(a, b) -> out()`, sweep a 3 x 4 Cartesian space, assert the result has
dims `(a: 3, b: 4)`, correct values, correct coords, and that exactly 12
calls were made.

**Acceptance Scenarios**:

1. **Given** a decorated function with contract `loop(aot, rh) -> t()` and a
   space with `aot(aot)` (3 values) and `rh(rh)` (2 values), **When** the
   user calls it on the space, **Then** the result has dims
   `(aot: 3, rh: 2)`, the function was called 6 times, and each call
   received native Python scalars.
2. **Given** the same function and a space where `aot(y, x)` and `rh(y, x)`
   share dims, **When** called, **Then** the result has dims `(y, x)` (zip,
   not product) and one call was made per pixel.
3. **Given** a completed sweep, **When** the user calls again with the same
   space, statics and version, **Then** zero point-function calls are made
   and the same result is returned.
4. **Given** a space with a 0-d variable (`sza = 35.0`), **When** swept,
   **Then** the value is delivered to every call and does not create a
   result dim.

---

### User Story 2 - Survive failures and resume an interrupted sweep (Priority: P2)

A sweep of thousands of expensive calls is interrupted (crash, wall-time
limit) or some individual calls fail. The user relaunches the same sweep:
already-computed points are skipped, failed or missing points are recomputed,
and per-point status is always inspectable.

**Why this priority**: resumability is the killer feature versus ad-hoc
loops and pure task graphs; expensive engine hours must never be recomputed
or silently lost.

**Independent Test**: run a sweep whose function fails on a known subset of
points, assert failed points carry NaN and `status=failed` while others
succeed; relaunch and assert only the failed/missing points are recomputed.

**Acceptance Scenarios**:

1. **Given** a sweep where one call raises, **When** the default error
   policy is active, **Then** that point's output slice is NaN, its status
   is `failed`, and the sweep continues to completion.
2. **Given** a sweep interrupted after k of n points, **When** relaunched
   identically, **Then** exactly n - k point calls are made.
3. **Given** a resumed sweep against a store written with a different space
   (an axis was extended), **When** launched, **Then** the mismatch is
   detected and the run is refused with a clear message.
4. **Given** a fail-fast policy, **When** a call raises, **Then** the sweep
   stops with the original error.

---

### User Story 3 - Deduplicate a pixel map sweep (Priority: P3)

A user sweeps a per-pixel function over maps `aot(y, x)`, `rh(y, x)` where
many pixels share identical parameter values. With `dedup=True`, only unique
value combinations are computed; results are re-expanded to the full map.
The result is bit-identical to the non-dedup run.

**Why this priority**: turns O(n_pixels) engine calls into O(n_unique),
often orders of magnitude; but it is an optimisation of Story 1, not a
prerequisite.

**Independent Test**: sweep a map with known duplicates with and without
dedup; assert identical results and the reduced call count.

**Acceptance Scenarios**:

1. **Given** maps with d unique (aot, rh) rows among n pixels, **When**
   swept with `dedup=True`, **Then** exactly d point calls are made and the
   result equals the non-dedup result bit for bit.
2. **Given** `dedup=("y", "x")` on a space that also has a `time` dim,
   **When** swept, **Then** dedup collapses only spatial duplicates and
   `time` remains fully swept.
3. **Given** loop variables of mixed dtypes (float + str), **When** swept
   with dedup, **Then** dedup works (no dtype coercion failure).
4. **Given** `dedup=("z",)` where `z` is not a dim of the space, **When**
   launched, **Then** the run fails immediately listing the available dims.

---

### User Story 4 - Batch a vectorisable axis and pass context data (Priority: P4)

A function accepts a wavelength vector natively (`vec(wl @ 8)`) and needs a
full SRF table in every call (`const(srf)`), e.g. band integration: the wl
axis is consumed by the integration and absent from the output.

**Why this priority**: matches how real engines are called efficiently
(vectorised axes, context LUTs); required by the band-integration and
multi-geometry consumers.

**Independent Test**: sweep with a vec axis of 20 values and max batch 8;
assert the function received batches of 8, 8 and 4, and the concatenated
result is correct; assert the const variable arrived whole in every call.

**Acceptance Scenarios**:

1. **Given** `vec(wl @ 8)` over 20 wavelengths, **When** swept, **Then**
   calls receive wl chunks of sizes 8, 8, 4 and outputs are reassembled in
   order.
2. **Given** `const(srf)` with `srf(band, wl)`, **When** swept, **Then**
   every call receives the entire srf array unchunked.
3. **Given** a contract batching a dim absent from the declared output
   (`vec(wl @ 8) -> band_int(band)`), **When** the contract is validated,
   **Then** it is rejected: batching a reduced dim silently corrupts
   integrals.
4. **Given** a multi-dim vec pair `A(y, x)`, `B(y, x)` with no loop clause,
   **When** swept with no chunks, **Then** one call receives both maps
   whole; **When** swept with chunks of 500 on each of `y` and `x`, **Then**
   the function is called once per tile and the result is identical.
5. **Given** a `@ N` marker on a multi-dim vec variable, **When** the
   contract is validated, **Then** it is rejected as ambiguous with a
   message pointing to the dim-keyed policy form.

---

### User Story 5 - Write sweeps as module classes (Priority: P5)

A user packages physics as a class: `contract` as a class-level declaration,
pure physics in `forward`, run configuration (store, dedup, chunks,
executor) given at instantiation. Contract errors explode at class
definition time, not after minutes of runs.

**Why this priority**: the ergonomic facade for real projects (adjeff,
radtrans); sugar over Story 1's machinery.

**Independent Test**: define a subclass with a bad contract string and
assert the error is raised at class definition; define a valid one and
assert `__call__` on a space equals the decorator path result.

**Acceptance Scenarios**:

1. **Given** a subclass without a `contract` declaration, **When** the class
   body is evaluated, **Then** a clear error is raised at definition time
   (unless the class is declared abstract).
2. **Given** a valid subclass, **When** instantiated with a policy and
   called on a space, **Then** behaviour is identical to the equivalent
   decorated function.
3. **Given** an intermediate base declared abstract, **When** defined
   without a contract, **Then** no error is raised; concrete subclasses
   still require one.

---

### Edge Cases

The 18 cases below (design doc section 9) are acceptance criteria; each row's
v0 policy is normative. "Later" items are documented limitations, not v0
work.

| # | Case | Symptom if unhandled | v0 policy (normative) | Later |
|---|------|----------------------|-----------------------|-------|
| 1 | Slightly different coords on a zip dim | Inner join silently drops points | Validate exact alignment, fail loudly | |
| 2 | Heterogeneous output grids between calls | Region writes corrupt or misalign | Detect, clear error | Declared union grid + NaN padding |
| 3 | Output dims of unknown size before first call | Store cannot be pre-allocated | Probe call, or sizes declared in contract | |
| 4 | Invalid points (masked pixels, NaN axes, absurd combos) | Wasted or crashing engine calls | Skip predicate or NaN-skip + status variable | |
| 5 | Call failure mid-sweep | One failure kills the sweep; resume ambiguous | NaN slice + status=failed + continue; optional fail-fast; retry count | |
| 6 | Parallel store writes across region boundaries | Corrupted shared chunks | Derive store chunks from contract (1 loop point = 1 chunk or an integer multiple) | |
| 7 | Non-serialisable statics | Cache key impossible | Cache-token protocol or explicit exclusion | |
| 8 | Function code changes | Stale cache reused | User-declared version identifier | |
| 9 | Resume with modified space (axis extended) | Silent misalignment with store | Detect mismatch, refuse with clear message | Reindex store on pure extension |
| 10 | Non-numeric axes (str, datetime64) | Float coercion / unique-row extraction break | First-class support, tested | |
| 11 | Last batch smaller than max batch | Function assuming fixed size breaks | Documented; golden test | |
| 12 | Batching a dim the function reduces over | Silently wrong integrals | Contract validation: no batch marker on dims absent from output | |
| 13 | Duplicate values on a loop axis | Redundant expensive calls | Opt-in dedup | |
| 14 | Replication without a carrier variable | Looping a bare dim impossible, user confusion | Document the seed-variable idiom | Reserved seed helper |
| 15 | Object-valued sweep coordinates | Hash / serialise / unique failures | Enforce primitives, clear error, label idiom documented | |
| 16 | Same output name from two instances | Single-producer DAG check breaks | Documented; rename designed, not implemented | Policy-level output rename |
| 17 | Dask-backed loop variables | Uncontrolled materialisation | Explicit compute + warning | Streaming dedup + blockwise iteration |
| 18 | Monte-Carlo reproducibility | Cache honesty, per-point replay | Seed-variable idiom | Per-point derived-seed helper |

## Requirements *(mandatory)*

### Functional Requirements

**Space semantics**

- **FR-001**: The system MUST accept the sweep space as a single Dataset;
  shared dims mean zipped (covarying) axes, distinct dims mean Cartesian
  product; the sweep space is the union of the loop variables' dims. 0-d
  variables are accepted as fixed-but-present values.
- **FR-002**: Result loop dims MUST equal the union of the dims of the loop
  variables in the space, independently of any policy (sacred property,
  constitution principle III).
- **FR-003**: Alignment on shared dims MUST be exact; near-identical coords
  MUST fail loudly rather than be silently inner-joined (edge case 1).
- **FR-004**: Sweep coordinate values MUST be primitives (float, int, str,
  bool, datetime64); other types MUST be rejected with an error pointing to
  the documented label + cache-token idiom (edge cases 10, 15; finding 5).
  String and datetime64 axes MUST be supported first-class and tested.
- **FR-005**: In v0, loop variables are assumed in-memory; lazily-backed
  loop variables MUST be computed explicitly with a warning (edge case 17;
  finding 7).

**Contract**

- **FR-006**: A contract MUST declare, per input variable, how one call
  consumes it (one value per call; whole axis or batches of at most N; whole
  context data) and MUST name each output with its call-level dims.
  Contracts MUST be accepted both as an object and as a string coerced to
  the object. The clause vocabulary is fixed: `loop` (one value per call),
  `vec` (batchable axis), `const` (context data delivered whole), `@ N` as
  the batch-size marker, `->` introducing named outputs.
- **FR-007**: Contract parsing and coherence validation MUST happen at class
  definition time for modules and at decoration time for functions; no
  contract error may surface only at sweep time.
- **FR-008**: A batch annotation on a dim absent from the declared outputs
  MUST be rejected at validation time (edge case 12; finding 3).
- **FR-009**: The `const` clause MUST be supported in v0: a const variable
  is delivered whole to every call and never chunked (finding 2).
- **FR-010**: Vec variables MAY carry any number of dims. The contract
  declares WHICH variables are batchable (physics); batch SIZES are
  dim-keyed run policy (e.g. chunks of 500 on `y` and 500 on `x`). The
  `@ N` marker in the contract is allowed only when the vec variable is
  1-D, where variable and dim names coincide unambiguously, and then acts
  as a physics-informed default overridable by policy; `@ N` on a multi-dim
  vec variable MUST be rejected as ambiguous, pointing to the policy form.
  Batching MUST slice each annotated dim independently and iterate over the
  product of slices (tiles). Batches MUST respect the declared maximum; the
  last batch along any dim MAY be smaller and this MUST be covered by a
  golden test (edge case 11).

**Call delivery and return**

- **FR-011**: By default the wrapped function MUST receive named arguments
  drawn from the contract; loop variables arrive as native Python scalars
  (per-variable array override available); vec and const variables arrive as
  labelled arrays. This is the only delivery mode in v0: an opt-out handing
  the raw per-call Dataset to the function is deliberately absent, since no
  identified consumer needs it and the contract already names every
  argument. Adding it later is backward compatible.
- **FR-012**: Static kwargs MUST be passed verbatim to every call and MUST
  enter the cache key (JSON-serialisable, cache-token protocol, or explicit
  visible exclusion) (edge case 7).
- **FR-013**: Return values MUST be normalised: a bare (even unnamed) array
  for single-output contracts (the system names it), an ordered tuple mapped
  to declaration order for multi-output contracts, or an explicit Dataset
  validated nominatively with a clear error on wrong names.

**Execution, store, cache**

- **FR-014**: Results MUST stream to a pre-allocated persistent store via
  per-call region writes; the sweep returns a lazy handle on that store by
  default; peak output-side memory MUST stay bounded by one chunk regardless
  of sweep size. Store chunk boundaries MUST be derived from the contract so
  parallel writers never share a chunk (edge case 6). Write granularity is
  one loop point per region.
- **FR-032**: A store is owned by one sweep configuration in v0. The store
  MUST carry a fingerprint of (contract, version, statics); opening a store
  whose fingerprint differs MUST fail with a clear message rather than mix
  results. Sharing one store between instances is a documented v0
  limitation, designed together with the output-rename feature.
- **FR-033**: Reading a store MUST never be blocked, including while a
  sweep is writing it: opening the store to inspect partial results and the
  status variable is the supported way to follow a running sweep (FR-027).
- **FR-034**: A store-less in-memory mode MUST exist: results are held in
  memory and returned materialised, with no region writes, no cache and no
  resume. It is the mode for cheap callees, small sweeps and the test suite;
  it removes the per-point write cost, leaving only in-memory bookkeeping.
  Persistence is engaged by naming a store, so a sweep with no store named
  runs in memory; that MUST never be silent, and the run summary MUST state
  that results were not persisted and no cache was consulted.
- **FR-035**: Batch sizes MUST be explicit in v0: the contract default
  (`@ N`) or the dim-keyed policy, and an unannotated vec dim means the
  whole axis in one call. Automatic batch sizing MUST NOT ship in v0. It is
  designed for later in a deterministic form only (derived from a declared
  memory budget and the output size measured by the probe call, never from
  call timings, which are unreliable on first call and non-linear in batch
  size), and it MUST then decide once, persist the choice in the store
  metadata and reuse it on resume, since batch size determines the store
  chunk grid (edge case 6). The library can bound its own buffers, never the
  callee's internal allocation.
- **FR-036**: A store MUST tolerate only one writing run at a time. While a
  sweep is writing, the store MUST carry a lock identifying the owning run;
  a second process opening it for writing MUST refuse to start with a clear
  message naming the owner. A documented escape MUST exist for locks left
  behind by a dead run. The lock constrains writers only (FR-033). It is
  independent of the process-pool executor, whose workers write under the
  single lock of their parent run, so single-node CPU parallelism is
  unaffected. Multi-process cooperation on one store (cluster array jobs
  claiming points atomically) is out of v0 scope; the documented answer for
  that case is to partition the space at submission time, one store per
  task, and merge the stores afterwards.
- **FR-015**: A status sidecar variable (ok / failed / skipped) MUST
  accompany the data so resume and failure are distinguishable.
- **FR-016**: When output sizes are unknown before the first call, the
  system MUST discover them via a probe call, or use sizes declared in the
  contract (edge case 3). Heterogeneous output grids across calls MUST be
  detected and reported as errors (edge case 2).
- **FR-017**: Resume MUST skip already-written points; resuming against a
  store written from a different space MUST be detected and refused with a
  clear message (edge case 9).
- **FR-018**: The cache key MUST include point values, statics, the
  contract, and a user-declared version string; function bytecode MUST NOT
  be hashed (edge case 8). The version string is a field of the CONTRACT,
  declared beside it (decorator keyword or class-level declaration) and not
  part of the string grammar. It MUST NOT be a policy field: a run-time
  setting must never select a different cached result (constitution
  principle III). Bumping it is the user's declaration that the computation
  changed, including changes the system cannot observe (function body,
  external engine binary, external data files).
- **FR-019**: Error policy: default = NaN slice + status failed + continue;
  fail-fast MUST be available; a retry count MUST be supported; an explicit
  skip predicate MUST mark points skipped without calling them (edge cases
  4, 5). Skipping is opt-in only: NaN values in loop variables MUST NOT
  trigger an automatic skip, because an unexpected NaN is more often a bug
  in space construction than a deliberate mask, and silently skipping it
  would produce a result full of unexplained NaN slices.
- **FR-020**: Executors MUST be pluggable behind a map-like interface with
  serial and process-pool implementations in v0; core semantics MUST NOT
  depend on any distributed framework (constitution principle X).
- **FR-021**: Dedup MUST be opt-in: enabled over all loop dims, or over a
  named subset of dims; it MUST handle mixed primitive dtypes; policy
  options naming unknown dims (dedup, chunks) MUST fail before any call,
  listing available dims (edge cases 13, 10).
- **FR-022**: Bit-identity MUST hold and be tested: identical results with
  and without dedup, with and without batching, across executors
  (constitution principle III).
- **FR-027**: Observability in v0 is structured logging through the standard
  library only: one event per point at debug level, and a run summary at
  info level (points computed, reused from cache, failed, skipped, elapsed).
  No progress callback and no new policy field: live progress is already
  obtainable by inspecting the status variable in the store. The library
  MUST NOT configure logging handlers on the user's behalf.

**Surfaces**

- **FR-023**: The decorator surface MUST wrap a plain function given a
  contract and default policy; the module surface MUST require a
  class-level contract at subclass definition (with an explicit abstract
  escape hatch for intermediate bases, contract inherited and overridable),
  keep orchestration in the call path and pure physics in a separately
  testable method; both surfaces MUST produce identical results for
  equivalent inputs.
- **FR-024**: Policy precedence MUST be call > instance > decorator default,
  and policy MUST be validated against the space before any expensive call.
- **FR-029**: Planning MUST be a distinct phase, separate from execution.
  Given a contract, a space, statics and a policy, planning resolves the
  loop points, dedup, batching, result and store shapes, and the resume
  state, and MUST complete without calling the wrapped function even once.
  Execution consumes the plan.
- **FR-037**: Planning MUST emit a single stream of work items, each
  carrying its point index, its unique representative under dedup, its batch
  slices and whether it is already satisfied by the store. Point
  enumeration, dedup, batching and resume MUST be resolved into that stream
  rather than combined at execution time, so that execution is one loop
  without conditional branches and the interactions between those four
  mechanisms are decided in one place.
- **FR-030**: The plan MUST be inspectable by the user before execution,
  through a method on the decorated function and on the module (no new
  public object). It MUST report at least: the contract and its version;
  the loop space with its dims, whether each is a product or a zip, and the
  resulting point count; the unique point count when dedup is enabled; the
  batch division per vec dim including the smaller last batch; the total
  call count; how many of those are already satisfied by the store and how
  many remain; the name, dtype and shape of each argument one call receives,
  statics included; the result variables with their full dims, dtype and
  size; the store path, chunk grid and region count; and the executor.
  Rendering (plain text, notebook representation) is an implementation
  decision, deliberately left open.
- **FR-031**: Plan honesty rules. Inspecting a plan MUST NOT call the
  wrapped function, so output sizes that would require a probe call MUST be
  reported as undetermined rather than probed. Counting unique points is
  real work on large spaces and MUST be reported as such. Duration MUST NOT
  be invented: the plan reports call counts, and may report a duration
  estimate only when derived from timings recorded by a previous run on the
  same store, stating that provenance.

**Documented idioms and limitations**

- **FR-025**: The replication idiom MUST be documented: a carrier variable
  (e.g. seed) on the replication dim makes repetitions distinct, cacheable
  and reproducible; a bare replication dim is not loopable (edge cases 14,
  18; finding 1).
- **FR-028**: The version-comparison idiom MUST be documented: results
  produced under different versions are compared by keeping one store per
  version and concatenating them along an explicit axis built by the user.
  Version is never itself a sweep axis, because it selects code rather than
  data; the system MUST NOT invent store paths per version on the user's
  behalf in v0.
- **FR-026**: A single index of v0 limitations MUST be documented. It is an
  index of cross-references, not a restatement, so that a limitation is
  described in exactly one place and cannot drift:

  | Limitation | Stated in |
  |------------|-----------|
  | Sharing one store between instances | FR-032 |
  | Automatic batch sizing | FR-035 |
  | Multi-process cooperation on a shared store | FR-036 |
  | Progress callbacks | FR-027 |
  | Automatic per-version store namespacing | FR-028 |
  | Raw per-call Dataset delivery | FR-011 |
  | Instance-level output rename | edge case 16, finding 6 |
  | Union output grids and NaN padding | edge case 2 |
  | Store reindexing on space change | edge case 9 |
  | Reserved seed helper | edge cases 14, 18 |
  | Streaming dedup and lazily-backed spaces | edge case 17, finding 7 |
  | Distributed-framework executor | FR-020 |
  | Halo and overlap granularity | design doc 4.2 |
  | Dependent or ragged axes | finding 4 |
  | Contract cross-check against type annotations | design doc 11 |

### Key Entities

- **Space**: one Dataset describing the whole sweep; variables carry the
  values, dims carry the product/zip semantics; 0-d variables allowed.
- **Contract**: frozen, hashable description of one call: loop variables
  (scalar delivery), vec variables (batchable axes), const variables
  (context data), named outputs with call-level dims, plus the version
  string. Property of the physics; enters the cache key.
- **SweepPolicy**: run configuration (store, dim-keyed batch chunks, dedup,
  executor, error policy); property of the run; never changes results, and
  therefore never carries the version string. Exact field list is settled at
  planning time.
- **Sweeper**: the engine binding a contract and a policy around a function;
  the decorator is sugar over it.
- **Plan**: the resolved execution description produced by the planning
  phase (loop points, dedup, batches, call count, resume state, result and
  store shapes, per-call arguments). Inspectable before execution and
  consumed by it; never produced by calling the wrapped function.
- **SweepModule**: class facade; class-level contract, pure `forward`,
  policy at init.
- **Store**: pre-allocated persistent array store; unifies cache, streaming
  output and resume; carries the status sidecar variable.
- **Statics**: ordinary kwargs passed verbatim to every call; configuration,
  not data; enter the cache key.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Re-running a completed sweep with identical space, statics and
  version performs zero point-function calls.
- **SC-002**: A sweep interrupted after k of n points completes after
  relaunch with exactly n - k additional point calls.
- **SC-003**: Results are bit-identical with and without dedup, with and
  without batching, and across serial and process executors, on the
  dedicated test suite.
- **SC-004**: A map sweep with d unique parameter rows among n pixels
  performs exactly d point calls when dedup is enabled, and the result
  equals the non-dedup result.
- **SC-005**: Every invalid configuration in the edge-case table with a
  "fail loudly" v0 policy (cases 1, 2, 8-as-version-mismatch, 9, 12, 15,
  unknown policy dims) produces its error before the first expensive call
  (or at definition/decoration time where specified).
- **SC-006**: Nine of the ten worked examples of the design doc (section 8)
  are expressible in v0 without library changes; the tenth (ragged axes) is
  covered by the documented union-grid workaround.
- **SC-007**: The adjeff samplers (e.g. the spherical-albedo sampler) can be
  expressed with their existing point-function signatures unchanged, with
  behaviour matching their current output.
- **SC-008**: A sweep 10x larger than available output memory completes with
  peak output-side memory bounded by one store chunk.
- **SC-009**: On a sweep of 1e4 loop points whose calls each take at least
  one second, library overhead (region writes, cache lookups, bookkeeping)
  accounts for under 1% of wall time.
- **SC-010**: Launching a second writing run against a store already in use
  fails within seconds, before any point call, naming the owning run.
- **SC-011**: The three costly misconfigurations are visible in the plan
  without a single call of the wrapped function: dedup forgotten on a map
  sweep (point count in the millions instead of thousands), an accidental
  Cartesian product where a zip was intended (one axis per variable instead
  of the shared ones), and a batch size of one on a vectorisable axis (call
  count multiplied by the axis length).

## Assumptions

- `docs/design/xsweep.md` is the source of truth for rationale; its section
  10 defines the v0 in/out boundary verbatim; decisions recorded there are
  not re-litigated by this spec.
- The three open questions of design doc section 11 that affected spec
  scope were settled on 2026-07-29 and are recorded in FR-006 (vocabulary
  `loop` / `vec` / `const`, `@ N`, `->`), FR-010 (vec may be multi-dim;
  batchability in the contract, batch sizes in dim-keyed policy) and
  FR-014/FR-032 (one loop point per region, store owned by one sweep
  configuration). The FR-010 resolution follows the existing adjeff
  mechanism, whose per-dim slice product already yields tiles for
  multi-dim vectors and whose chunk sizes already live in run
  configuration. The remaining section 11 questions (exact SweepPolicy
  field list and precedence mechanics beyond the stated order; package
  layout and environment setup) are planning-time decisions, deferred to
  `/speckit-plan`.
- Target scale of the persistent mode: around 1e4 loop points per sweep,
  with callees costing seconds to minutes each. One write region per loop
  point is negligible at that ratio (measured at about five milliseconds
  against ten seconds; the first estimate of one millisecond was optimistic
  by a factor of five, most of the cost sitting inside the array library
  rather than on the filesystem), and
  the regime is bounded by physics anyway: 1e6 points at ten seconds each
  would run for months. Larger parameter grids (a 1e6-pixel map) reach that
  range through dedup. Cheap callees are served either by the `vec` clause,
  which turns a million scalar calls into one vectorised call, or by the
  in-memory mode (FR-034). Only their conjunction stays a non-goal: cheap,
  scalar-only, very many points, and needing persistence. Coalescing several
  loop points per region would serve that corner, at the price of coarser
  resume granularity, a coarser status variable and an executor coupled to
  the store layout; it stays possible later as a purely internal change.
- Consumers assumed: adjeff (replacing its internal sweep plumbing),
  radtrans (generic half of its sweep layer), short sensitivity studies.
  The rule of three is satisfied; no speculative generality beyond these.
- Engineering standards (language version, layout, tooling) are fixed by the
  constitution and not repeated here.
- Naming the array/store substrate (xarray Datasets, persistent chunked
  store) in requirements is domain vocabulary, not implementation leakage:
  the product is by definition an xarray library (constitution principles
  IV and VIII).
