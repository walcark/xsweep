# Tasks: xsweep v0

**Input**: Design documents from `specs/001-xsweep-v0/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/,
quickstart.md

**Tests**: INCLUDED. Not optional here: constitution principle III makes the
bit-identity suite a release gate, FR-022 requires it explicitly, and
quickstart.md defines eleven validation scenarios.

**Organization**: grouped by user story so each is independently
implementable and testable.

**Revised 2026-07-29** after `/speckit-analyze`: no separate grammar, cache
or in-memory-store modules (twelve modules total), planning emits one stream
of work items (FR-037), `unpack` dropped from v0, `Plan` exported.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelisable (different files, no dependency on unfinished work)
- **[Story]**: US1 to US5, on user-story phases only
- Paths are repository-relative; the package lives in `src/xsweep/`

---

## Phase 1: Setup

**Purpose**: project skeleton and tooling, mirroring radtrans conventions

- [X] T001 Create `pyproject.toml` with project metadata (name xsweep, version 0.1.0, requires-python >=3.11, author, MIT), hatchling build backend, hard dependencies xarray/zarr/numpy, optional extra `dask`, pixi workspace (conda-forge, linux-64), dev feature with ruff/mypy/pytest, tasks fmt/fmt-check/lint/type-check/test/all, ruff line-length 88 target py311 rules E,F,I,UP,B, mypy strict true python_version 3.12, pytest testpaths tests
- [X] T002 [P] Create `.gitignore` covering `.pixi/`, `pixi.lock`, `__pycache__/`, `*.egg-info/`, build artefacts and the ruff/mypy/pytest caches
- [X] T003 [P] Create the package skeleton `src/xsweep/__init__.py` (empty public surface for now) and `src/xsweep/py.typed`
- [X] T004 [P] Create the test tree `tests/unit/`, `tests/contract/`, `tests/property/`, `tests/integration/` with `tests/conftest.py` holding the shared synthetic-callee fixtures (a counting callable recording every call and its arguments)
- [X] T005 Verify `pixi run -e dev all` is green on the empty package, and confirm in `specs/001-xsweep-v0/research.md` R13 the exact in-memory store class of the pinned zarr version and how it is handed to `to_zarr`

**Checkpoint**: tooling runs, the zarr in-memory store API is confirmed

---

## Phase 2: Foundational (blocking prerequisites)

**Purpose**: the pieces every user story needs. No story work starts before
this phase completes.

- [X] T006 [P] Implement the exception hierarchy in `src/xsweep/errors.py`: `XsweepError` base, `ContractError`, `SpaceError`, `PolicyError`, `StoreError`, `StoreLockedError`, `PointFailed`, each documented with the phase that raises it
- [X] T007 Implement the contract objects in `src/xsweep/contract.py`: frozen `LoopVar`, `VecVar`, `OutVar`, `Contract` with `version`, the `inputs`, `outputs` and `out_dims` properties, then the tokeniser and recursive-descent parser following the EBNF of contracts/contract-dsl.md, then `Contract.parse` and the definition-time coherence checks (rules 1 to 7 of data-model.md). One module so parse errors and coherence errors share one catalogue
- [X] T008 [P] Write parser tests in `tests/unit/test_grammar.py`: every production, clause order freedom, whitespace insensitivity, and one test per row of the error catalogue asserting the message and the character offset
- [X] T009 [P] Write contract coherence tests in `tests/unit/test_contract.py`: name in two clauses, duplicate output, `@ N` below 1, no outputs, invalid identifier, hashability, equality of the string and object forms
- [X] T010 Implement `SweepPolicy`, the `UNSET` sentinel, `DEFAULTS` and the layered `resolve()` in `src/xsweep/policy.py` per research R1 and R2 (ten fields, folding defaults then decorator then instance then call)
- [X] T011 [P] Write policy resolution tests in `tests/unit/test_policy.py`: a field explicitly set to `False` or `0` overrides a truthy lower layer, an unset field inherits, the four-layer order holds, the reserved static names `space` and `policy` are rejected
- [X] T012 Implement space validation in `src/xsweep/space.py`: contract inputs present, primitive dtype whitelist with the label-idiom message, exact alignment on shared dims, 0-d variables accepted, lazily-backed loop variables computed with a warning
- [X] T013 Implement loop-point enumeration in `src/xsweep/space.py`: the sweep space as the union of loop variable dims, product and zip origin per dim, and an index-to-values mapping that never materialises the full Cartesian table
- [X] T014 [P] Write space tests in `tests/unit/test_space.py`: product versus zip, string and datetime64 axes first-class, object dtype rejected with the documented message, near-identical coordinates failing rather than inner-joining, 0-d variables creating no result dim, a lazily-backed loop variable warning
- [X] T015 Implement argument delivery and return normalisation in `src/xsweep/delivery.py`: loop values as native Python scalars (`deliver="array"` override), vec and const as DataArrays, statics verbatim, and the three accepted return shapes with nominative validation. No raw-Dataset opt-out in v0 (FR-011)
- [X] T016 [P] Write delivery tests in `tests/unit/test_delivery.py`: scalar types received, unnamed DataArray named after the single declared output, ordered tuple mapped to declaration order, Dataset with a wrong name raising `ContractError`
- [X] T017 Implement the fingerprint and statics canonicalisation in `src/xsweep/store.py` per research R5: blake2b over canonical JSON of the contract (version included) and the statics, with the `__cache_token__` hook and a clear failure naming a non-serialisable static
- [X] T018 [P] Write fingerprint tests in `tests/unit/test_fingerprint.py`: changing the version changes the fingerprint, changing a static changes it, reordering statics does not, a non-serialisable static fails naming the argument, an object exposing `__cache_token__` succeeds
- [X] T019 Implement the executor protocol and the serial executor in `src/xsweep/executors.py`: `map_unordered(fn, items)` streaming `(index, outcome)` pairs
- [X] T020 Implement `Plan` and `WorkItem` in `src/xsweep/plan.py` as frozen dataclasses with the fields of data-model.md, and export `Plan` from `src/xsweep/__init__.py`
- [X] T021 Implement the planning phase in `src/xsweep/plan.py`: resolve policy against the space, validate chunk and dedup dims against available dims, compute result and store shapes, and emit ONE stream of work items into which point enumeration, dedup, batching and resume are resolved (FR-037), all without calling the wrapped function even once (FR-029)
- [X] T022 [P] Write planning tests in `tests/unit/test_plan.py`: an unknown dim in `chunks` or `dedup` fails listing available dims, a chunk on a loop dim is rejected, the wrapped callable is never invoked, batch division including a shorter last batch, and the work-item stream carrying point index, unique representative, slices and done flag

**Checkpoint**: contract, policy, space, delivery, fingerprint and planning
exist and are unit-tested. User stories can begin.

---

## Phase 3: User Story 1 - Lift a point function into a cached gridded sweep (P1) 🎯 MVP

**Goal**: decorate a point function, sweep a space, get a labelled lazy
result computed once per point and persisted as it streams.

**Independent Test**: decorate a counting synthetic callable with
`loop(a, b) -> out()`, sweep a 3 by 4 space, assert dims `(a: 3, b: 4)`,
correct values and coordinates, exactly 12 calls; re-run and assert zero
calls.

- [X] T023 [US1] Implement store allocation in `src/xsweep/store.py`: derive the chunk grid from the contract (1 along every loop dim, batch size or full extent along call-output dims), pre-allocate metadata only, write `xsweep-meta.json` with fingerprint, space signature, chunk grid, contract rendering and xsweep version
- [X] T024 [US1] Implement region writes and the status variable in `src/xsweep/store.py`: one region per work item, `uint8` status over the loop dims with the 0/1/2/3 mapping in the attributes (research R8)
- [X] T025 [US1] Implement the store-less mode in `src/xsweep/store.py` over zarr's in-memory store (research R13): same allocation and write path, skipping fingerprint and lock, returning a materialised Dataset. No separate backend module
- [X] T026 [US1] Implement fingerprint verification on open in `src/xsweep/store.py`: a mismatching contract, version or statics raises `StoreError` naming what differs rather than mixing results (FR-032)
- [X] T027 [US1] Implement the execution loop in `src/xsweep/sweeper.py`: consume `plan.work_items` in one loop with no conditional branching, deliver arguments, normalise returns, write regions, set status, and return the lazily-opened handle (materialised when `load` is true or in the store-less mode)
- [X] T028 [US1] Implement `Sweeper` and the `@sweep` decorator in `src/xsweep/sweeper.py`, parsing and validating the contract at decoration time (FR-007), and export the public names from `src/xsweep/__init__.py`
- [X] T029 [P] [US1] Write contract tests in `tests/contract/test_public_api.py`: the six public names importable, signatures matching contracts/public-api.md, a malformed contract raising at decoration
- [X] T030 [P] [US1] Write integration tests in `tests/integration/test_core_lift.py` for quickstart scenario 1: Cartesian sweep with call count and native scalar types, zipped maps producing `(y, x)` with one call per pixel, second run with zero calls, 0-d variable delivered without creating a dim, and the logged run summary including the not-persisted notice in the store-less mode
- [X] T031 [P] [US1] Write store tests in `tests/integration/test_store_layout.py`: chunk grid matches the contract, one region per loop point, status pre-allocated to pending, `xsweep-meta.json` content, fingerprint mismatch refused, store-less mode producing an identical result to the persistent one

**Checkpoint**: the core lift works end to end with caching and resume by
status. This is the MVP.

---

## Phase 4: User Story 2 - Survive failures and resume an interrupted sweep (P2)

**Goal**: individual failures never kill a sweep, interrupted runs resume
exactly, and per-point status is always inspectable.

**Independent Test**: a callee failing on a known subset yields NaN and
`status=failed` there and `ok` elsewhere; relaunching recomputes only the
failed and missing points.

- [ ] T032 [US2] Implement the error policy in `src/xsweep/sweeper.py`: `on_error="nan"` writes a NaN slice, sets status failed and continues; `on_error="raise"` aborts with the original exception surfaced as `PointFailed`
- [ ] T033 [US2] Implement the retry count in `src/xsweep/sweeper.py`: up to `retries` attempts per work item before marking it failed, each attempt logged
- [ ] T034 [US2] Implement the skip predicate in `src/xsweep/plan.py`: `skip_where` evaluated per point during planning and carried on the work item, so skipped points are never called; no automatic NaN-skip (research R3, FR-019)
- [ ] T035 [US2] Implement resume in `src/xsweep/plan.py`: work items already `ok` in the store are marked done, so an identical re-run performs zero calls (SC-001), and everything not `ok` is recomputed
- [ ] T036 [US2] Implement space-change detection in `src/xsweep/store.py`: compare the stored space signature with the current one and refuse with a message naming the differing dim or coordinate (FR-017, edge case 9)
- [ ] T037 [P] [US2] Write integration tests in `tests/integration/test_failure_resume.py` for quickstart scenario 2: failing subset yields NaN and failed status while the sweep completes, relaunch recomputes only what is not ok, interruption mid-run then relaunch recomputes exactly the missing points, `on_error="raise"` surfaces the original error, extended axis refused
- [ ] T038 [P] [US2] Write unit tests in `tests/unit/test_error_policy.py`: retries exhausted marks failed, a retry that succeeds marks ok, `skip_where` never calls the function, NaN loop values are NOT skipped by default

**Checkpoint**: stories 1 and 2 both work independently.

---

## Phase 5: User Story 3 - Deduplicate a pixel map sweep (P3)

**Goal**: collapse duplicate parameter rows to unique calls and re-expand,
with a bit-identical result.

**Independent Test**: sweep a map with known duplicates with and without
dedup; identical results, call count equal to the unique row count.

- [ ] T039 [US3] Implement unique-row reduction in `src/xsweep/dedup.py` using per-column factorisation then unique rows on the integer codes (research R4), stripping coordinate labels before broadcasting to avoid label alignment
- [ ] T040 [US3] Implement expansion in `src/xsweep/dedup.py`: map results computed on unique rows back onto the full loop grid
- [ ] T041 [US3] Wire dedup into the work-item stream in `src/xsweep/plan.py`: `dedup=True` covers all loop dims, a tuple covers the named subset, each item carries its unique representative, the unique count enters the plan, unknown dims fail before any call
- [ ] T042 [P] [US3] Write dedup unit tests in `tests/unit/test_dedup.py`: mixed dtypes including strings, NaN semantics pinned, partial dims leaving `time` fully swept, round trip reduce then expand being the identity
- [ ] T043 [P] [US3] Write integration tests in `tests/integration/test_dedup_map.py` for quickstart scenario 3: unique call count, result equality with the non-dedup run, unknown dim failing with the available-dims listing

**Checkpoint**: stories 1 to 3 work independently.

---

## Phase 6: User Story 4 - Batch a vectorisable axis and pass context data (P4)

**Goal**: hand whole axes or batches to vectorised callees and pass context
tables whole.

**Independent Test**: 20 wavelengths with max batch 8 produce batches of 8,
8 and 4 reassembled in order; the const variable arrives whole every time.

- [ ] T044 [US4] Implement batch division in `src/xsweep/plan.py`: per-dim slice lists from the contract `@ N` default and the policy `chunks` override, multiplying work items over the product of slices so multi-dim vec variables yield tiles (FR-010)
- [ ] T045 [US4] Implement vec and const delivery in `src/xsweep/delivery.py`: vec variables sliced per the work item, const variables passed whole and never chunked
- [ ] T046 [US4] Implement the reduced-dim batching rejection in `src/xsweep/contract.py`: a batch marker on a dim absent from every declared output is refused at definition with the catalogue message (FR-008, edge case 12)
- [ ] T047 [US4] Implement the multi-dim `@ N` rejection in `src/xsweep/plan.py`: `@ N` on a vec variable that is not 1-D in the space fails, pointing at the dim-keyed policy form
- [ ] T048 [P] [US4] Write batching unit tests in `tests/unit/test_batching.py`: exact batch sizes including a shorter last batch, tiles from the product of per-dim slices, whole axis when unannotated
- [ ] T049 [P] [US4] Write integration tests in `tests/integration/test_batch_const.py` for quickstart scenario 4: batches of 8, 8, 4 reassembled in order, const whole in every call, reduced-dim batching rejected at decoration, multi-dim `@ N` rejected, two maps swept whole then tiled giving identical results

**Checkpoint**: stories 1 to 4 work independently.

---

## Phase 7: User Story 5 - Write sweeps as module classes (P5)

**Goal**: package physics as a class whose contract errors explode at class
definition.

**Independent Test**: a subclass with a bad contract fails at class
definition; a valid one produces results identical to the decorated function.

- [ ] T050 [US5] Implement `SweepModule` in `src/xsweep/module.py`: `__init_subclass__` validating the class-level contract at import with the `abstract=True` escape hatch, contract inherited and overridable, policy taken at instantiation
- [ ] T051 [US5] Implement `__call__`, `explain` and the `forward` contract in `src/xsweep/module.py`, keeping orchestration in `__call__` and pure physics in `forward`, and export `SweepModule` from `src/xsweep/__init__.py`
- [ ] T052 [P] [US5] Write module tests in `tests/contract/test_module.py` for quickstart scenario 5: missing contract failing at class definition, `abstract=True` silent, contract inheritance and override, `forward` testable in isolation
- [ ] T053 [P] [US5] Write equivalence tests in `tests/integration/test_surface_equivalence.py`: a module and the equivalent decorated function producing identical results for the same space, statics and policy (FR-023)

**Checkpoint**: all five user stories work independently.

---

## Phase 8: Cross-cutting execution concerns

**Purpose**: capabilities spanning every story: parallelism, store
discipline, observability and plan inspection.

- [ ] T054 [P] Implement the process executor in `src/xsweep/executors.py` over `ProcessPoolExecutor`, honouring `max_workers`, catching pickling failures and re-raising them naming the offending object (research R11)
- [ ] T055 [P] Implement the optional dask executor behind a lazy import in `src/xsweep/executors.py`, so no core path imports dask
- [ ] T056 Implement the store lock in `src/xsweep/store.py`: `xsweep-lock.json` created with `O_EXCL` holding pid, hostname, start time and plan digest, released by a context manager on normal exit and on SIGINT or SIGTERM, `StoreLockedError` naming the owner, `force_unlock` override; readers never take the lock (FR-033, FR-036)
- [ ] T057 Implement structured logging in `src/xsweep/sweeper.py`: one debug event per work item, an info run summary with computed, cached, failed, skipped and elapsed, an explicit info line when no store is named stating that results are not persisted, and no handler configuration by the library (FR-027)
- [ ] T058 Implement `explain()` on `Sweeper` and `SweepModule` returning the `Plan` without calling the wrapped function, and the textual `Plan.__repr__` reporting every element required by FR-030
- [ ] T059 Implement the plan honesty rules in `src/xsweep/plan.py`: an output size needing a probe is reported as undetermined and never probed, the unique-point count is announced as real work, and no duration is reported unless derived from timings recorded by a previous run on the same store, with that provenance stated (FR-031)
- [ ] T060 Implement the probe call in `src/xsweep/sweeper.py`: at execution only, on the first work item that is neither done nor skipped, its result KEPT and written as a normal point; a fully skipped space with undeclared sizes fails clearly (research R9, edge case 3)
- [ ] T061 Implement heterogeneous output detection in `src/xsweep/store.py`: a call returning a shape incompatible with the allocated store raises naming the work item and both shapes (edge case 2)
- [ ] T062 [P] Write concurrency tests in `tests/integration/test_lock.py` for quickstart scenario 8: a second writing process failing within seconds naming the owner, `force_unlock` overriding, reading during a run succeeding, process-executor workers never sharing a chunk
- [ ] T063 [P] Write plan tests in `tests/integration/test_explain.py` for quickstart scenario 6, SC-011: no call of the wrapped function, reported counts and shapes, undetermined size reported as such, the probe result kept (a one-point sweep leaves the counter at one), and the three costly misconfigurations visible

---

## Phase 9: Polish and release gates

**Purpose**: the invariants that must hold across everything, plus
documentation.

- [ ] T064 Write the sacred-property suite in `tests/property/test_policy_invariance.py`: bit-identical results parametrised over dedup on and off, several batch sizes, serial and process executors, and persistent versus store-less mode, so adding a policy field means adding a parameter rather than a test (FR-022, SC-003, constitution III)
- [ ] T065 [P] Write the fail-loudly suite in `tests/integration/test_fail_loudly.py` for quickstart scenario 11 and SC-005: one parametrised test walking every edge-case row whose v0 policy is to fail, each asserting the error arrives before the first call or at definition time, with a call counter proving it
- [ ] T066 [P] Write the idiom tests in `tests/integration/test_idioms.py` for quickstart scenario 9: string and datetime64 axes, object dtype rejected pointing at the label idiom, `seed` carrier variable giving a non-zero standard deviation across `rep`, two versions stored separately and concatenated along an explicit axis
- [ ] T067 [P] Write the memory-ceiling test in `tests/integration/test_memory_ceiling.py`: a sweep larger than the output memory completing with peak output-side memory bounded by one chunk (SC-008)
- [ ] T068 [P] Write the overhead benchmark in `tests/integration/test_overhead.py`: on 1e4 loop points with a callee sleeping one second (scaled down with a documented factor for CI), library overhead under 1% of wall time (SC-009)
- [ ] T069 [P] Write the adjeff-shape validation in `tests/integration/test_adjeff_shape.py` for quickstart scenario 10: an adjeff sampler expressed with its existing point-function signature unchanged (SC-007)
- [ ] T070 [P] Write the worked-examples coverage test in `tests/integration/test_worked_examples.py`: the nine expressible examples of design doc section 8 declared as contracts and planned without library changes (SC-006)
- [ ] T071 [P] Write `docs/idioms.md` documenting the two mandated idioms: the seed carrier variable for replication (FR-025) and one store per version with explicit concatenation for version comparison (FR-028), each with a runnable snippet
- [ ] T072 [P] Write `docs/limitations.md` from the FR-026 index, one section per limitation, each pointing at the requirement that states it rather than restating it
- [ ] T073 [P] Write the README usage section and audit the public docstrings, ensuring every public name carries a NumPy imperative docstring in English
- [ ] T074 Update `docs/design/xsweep.md` section 13 and the radtrans ROADMAP reference so radtrans depends on xsweep at v0
- [ ] T075 Run the full quickstart validation and `pixi run -e dev all`, then record any deviation in the spec rather than in code comments

---

## Dependencies and execution order

### Phase dependencies

- **Setup (Phase 1)**: no dependencies
- **Foundational (Phase 2)**: depends on Setup, blocks every user story
- **User stories (Phases 3 to 7)**: all depend on Foundational; US1 first as the MVP, then US2 to US5 which are mutually independent
- **Cross-cutting (Phase 8)**: depends on US1 for the store and execution loop; the plan-inspection tasks additionally want US3 and US4 present so the report can show dedup and batching
- **Polish (Phase 9)**: depends on US1 to US4 for the property suite

### Within each user story

- Store and execution before the surfaces
- Implementation before its tests, except where a test pins an error message the implementation must produce verbatim
- A story is complete when its independent test passes

### Parallel opportunities

- T002, T003, T004 during Setup
- In Foundational: T008, T009, T011, T014, T016, T018, T022 once their implementation lands
- Once Foundational completes, US2 to US5 can proceed in parallel if staffed
- T054 and T055 together; every Phase 9 task marked [P]

### Parallel example, User Story 1

```bash
Task: "Contract tests for the public API in tests/contract/test_public_api.py"
Task: "Integration tests for the core lift in tests/integration/test_core_lift.py"
Task: "Store layout tests in tests/integration/test_store_layout.py"
```

---

## Implementation strategy

### MVP first

1. Phase 1 Setup
2. Phase 2 Foundational, the blocking phase
3. Phase 3 User Story 1
4. Stop and validate: sweep, cache, resume by status, lazy return
5. That alone already replaces the core of adjeff's `SweepBundle`

### Incremental delivery

Foundation, then US1 (MVP), then US2 for robustness on long runs, then US3
which is what makes map sweeps affordable, then US4 for vectorised engines,
then US5 for ergonomics. Each increment is usable by adjeff without the
following ones.

### Notes

- `[P]` means different files and no dependency on unfinished work
- Commit after each task or coherent group, with Conventional Commits
  prefixes, only when explicitly asked
- Every error message added must state what to change, not only what is wrong
- Formatting, linting and typing always go through pixi tasks
