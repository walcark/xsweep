<!--
Sync Impact Report
- Version change: none (template) -> 1.0.0
- Modified principles: none (initial adoption)
- Added sections: Core Principles (I-X), Engineering Standards,
  Documentation Hierarchy, Governance
- Removed sections: none (template placeholders filled)
- Follow-up TODOs: none
-->

# xsweep Constitution

## Core Principles

### I. Content-Agnostic

The library MUST assume nothing about what the wrapped function computes.
No introspection of the function body, no domain-specific behaviour, no
special-casing of any engine. Everything xsweep needs to know about a
function is stated in its Contract and its SweepPolicy.

Rationale: xsweep serves radiative-transfer engines today and arbitrary
point functions tomorrow; any content assumption is a future bug.

### II. Minimal Surface

The public API is exactly four objects (`Sweeper`, `SweepPolicy`,
`SweepModule`, the `@sweep` decorator) plus the `Contract` description.
Pipelines, DAG engines and engine-specific planners live ABOVE the
library, never inside it. Any proposal that grows the public surface MUST
be rejected or redesigned to fit the existing objects.

Rationale: the library's value is that it can be learned in minutes and
composed under larger systems; surface growth destroys both.

### III. Sacred Property: Policy Never Changes the Result (NON-NEGOTIABLE)

Policy choices (dedup, chunking, executor, store) MUST NOT change the
result values or shape, only the cost of obtaining them. Result loop dims
are the union of the dims of the loop variables in the space, regardless
of policy. This MUST be enforced by dedicated bit-identity tests:
identical results with and without dedup, with and without chunking,
across executors.

Rationale: users tune policy for performance; a policy that alters
science output is silent data corruption.

### IV. Semantics From xarray, Not From a DSL

Sweep-space semantics are xarray's own: shared dims = zip (covariance),
distinct dims = Cartesian product; the sweep space is the union of the
loop variables' dims. The contract describes the CALL (how one invocation
consumes and produces variables), never the data layout. No einops-like
description of inputs is permitted.

Rationale: the arrays already encode the space; a second description
could only agree (redundant) or disagree (a bug).

### V. Three Temporalities Strictly Separated

1. Contract = property of the physics, fixed at function/class writing
   time (decorator argument or `contract` ClassVar).
2. SweepPolicy = property of the run, fixed at init, with precedence
   call > instance > decorator default.
3. Data = property of the call (space Dataset + static kwargs).

A concern MUST NOT migrate between temporalities: no policy in the
contract, no physics in the policy, no data at init.

Rationale: this split is what makes modules reusable across runs and
runs reproducible across data.

### VI. Cache Honesty

The cache key MUST include: the point values, the static kwargs
(JSON-serialised or via the `__cache_token__` protocol), the contract
hash, and a user-declared `version` string. Function bytecode hashing is
rejected as fragile; the user increments `version` on semantic change.
Changing the contract invalidates the cache. A static kwarg excluded
from the key MUST be excluded explicitly and visibly.

Rationale: a cache that can silently return stale physics is worse than
no cache.

### VII. Fail Loudly, Fail Early

Contracts MUST be parsed and validated at import time (modules, via
`__init_subclass__`) or decoration time (functions). Policy MUST be
validated against the space before any expensive call (unknown dims in
`dedup=` or `chunks=` fail immediately, listing available dims).
Alignment on zip dims MUST use `join="exact"`: slightly different coords
fail loudly instead of being silently inner-joined. Batching a dim the
function reduces over MUST be rejected at validation time.

Rationale: the wrapped calls are expensive (GPU kernels, subprocesses);
every error caught before the first call saves real hours.

### VIII. Lazy Output by Default

Results stream to a pre-allocated zarr store via region writes; the call
returns `xr.open_zarr(store)` (lazy, chunked) and `.load()` is the
user's choice. Memory MUST stay bounded by one chunk regardless of sweep
size. Zarr chunking MUST be derived from the contract (one loop point =
one chunk or an integer multiple) so parallel region writers never share
chunks. A `status` sidecar variable (ok/failed/skipped) MUST accompany
the data so resume and failure are distinguishable.

Rationale: the store unifies caching, streaming persistence and
resumability; an eager return reintroduces the accumulate-then-combine
memory profile xsweep exists to fix.

### IX. Primitive Sweep Values Only

Sweep coordinate values MUST be primitives: float, int, str, bool,
datetime64. Object-valued loop variables MUST be rejected with a clear
error pointing to the documented idiom (label string + static mapping
entering the cache key via `__cache_token__`).

Rationale: objects break cache hashing, zarr coordinate serialisation
and dedup's np.unique; the label idiom covers the use case cleanly.

### X. Dask Is a Backend, Never the Foundation

Executors are pluggable behind a map-like interface: serial,
ProcessPoolExecutor, and optionally dask distributed. Core semantics
(space construction, contract, cache, store, dedup) MUST NOT depend on
dask. In v0, loop variables are loaded in memory; dask-backed loop
variables are computed explicitly with a warning.

Rationale: dask is an execution engine, not a sweep semantics; binding
the core to it buys nothing for subprocess/GPU callees and costs
simplicity everywhere.

## Engineering Standards

- Python 3.11, `src/` layout, package name `xsweep`.
- pixi manages the environment and tasks; formatting, lint and type
  checking are delegated to pixi tasks (`fmt`, `lint`, `type-check`,
  `test`), never applied by hand.
- ruff for lint/format, mypy for typing (full typing required), pytest
  for tests. Line length 88.
- Docstrings: NumPy style, imperative mood, in English. All code,
  comments and identifiers in English.
- Hard dependencies: xarray, zarr, numpy. Optional extra: dask.
- Commit messages: Conventional Commits (`feat`, `fix`, `refactor`,
  `docs`, `test`, `chore`, ...), no AI attribution trailers.

## Documentation Hierarchy

`docs/design/xsweep.md` is the reference for WHY: rationale, worked
examples, rejected alternatives, known limits. Spec-kit artifacts
(`.specify/`, specs, plans, tasks) are the operational WHAT/HOW, derived
from it. Once a spec exists, implementation decisions are recorded in
spec-kit artifacts; the design document is amended only for genuine
design changes, with a pointer kept between the two.

## Governance

This constitution supersedes ad-hoc practice for xsweep. All specs,
plans and code reviews MUST verify compliance with the Core Principles;
Principle III additionally requires its dedicated bit-identity tests to
exist and pass before any release.

Amendments: propose the change with rationale, update this file, bump
the version (MAJOR for principle removal or redefinition, MINOR for a
new or materially expanded principle, PATCH for clarification), record
the change in the Sync Impact Report comment, and propagate to dependent
spec-kit artifacts in the same change.

**Version**: 1.0.0 | **Ratified**: 2026-07-29 | **Last Amended**: 2026-07-29
