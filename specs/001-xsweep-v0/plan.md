# Implementation Plan: xsweep v0

**Feature**: `001-xsweep-v0` | **Date**: 2026-07-29 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-xsweep-v0/spec.md`

## Summary

Lift an expensive point function into a gridded, cached, resumable xarray
computation. One core mechanism (a `Sweeper` binding a `Contract` and a
`SweepPolicy` around a callable), three surfaces (`@sweep` decorator,
`SweepModule` class, `Contract` object). Execution splits into two phases:
planning resolves points, dedup, batching and store shape without ever
calling the function; execution consumes the plan and streams results into a
zarr store through per-point region writes, which unifies cache, output and
resume.

## Technical Context

**Language/Version**: Python 3.11 (library target); development environment
pinned to 3.12 like radtrans, since numpy's typed stubs need 3.12+ to type
check.

**Primary Dependencies**: xarray, zarr, numpy (hard). dask as an optional
extra, used only as an executor backend, never imported by core paths.

**Storage**: zarr v3 directory store, pre-allocated, written by region. Two
sidecar JSON files inside the store: fingerprint metadata and write lock. The
store-less mode uses the same code path over zarr's in-memory store. Note
that "lazy return" does not mean "dask": without dask the returned handle is
lazily indexed by xarray; dask only adds chunked parallel reads.

**Testing**: pytest. Three layers: unit (parser, policy resolution, dedup,
cache key), contract (public API shapes and error messages), and property
tests for the sacred property (bit identity across dedup, batching,
executors).

**Target Platform**: Linux (primary), local filesystem stores. Object-store
backends are not v0.

**Project Type**: Library, no CLI, no runtime services.

**Performance Goals**: Library overhead under 1% of wall time on a sweep of
1e4 loop points whose calls each take at least one second (SC-009). Peak
output-side memory bounded by one store chunk (SC-008).

**Constraints**: Planning never calls the wrapped function (FR-029, FR-031).
Policy never changes results (constitution III). Contracts validated at
import or decoration time (FR-007). Region writes aligned to chunk
boundaries so parallel writers never share a chunk (FR-014, edge case 6).

**Scale/Scope**: Persistent mode calibrated for about 1e4 loop points with
callees costing seconds to minutes; larger grids reach that range through
dedup. Roughly 2000 lines of library code across 12 modules, plus tests.

## Constitution Check

*GATE: passed before Phase 0, re-checked after Phase 1 design.*

| Principle | Gate | Verdict |
|-----------|------|---------|
| I. Content-agnostic | No introspection of the wrapped callable, no domain branch | PASS. The contract names everything; `inspect` is used nowhere in the call path. |
| II. Minimal surface | Public names countable on one hand | PASS. `sweep`, `Sweeper`, `SweepPolicy`, `SweepModule`, `Contract` plus the error hierarchy. `Plan` is returned by a method, not constructed by users. |
| III. Sacred property | Bit-identity tests exist | PASS. `tests/property/test_policy_invariance.py` is a release gate, parametrised over dedup, batch sizes and executors. |
| IV. Semantics from xarray | No second description of the space | PASS. The contract describes the call only; product and zip come from dims. |
| V. Three temporalities | No concern migrates | PASS. `version` sits in `Contract` (FR-018), never in `SweepPolicy`; the policy has no physics field. |
| VI. Cache honesty | Key covers what changes results | PASS. Key = point values + statics + contract (version included). No bytecode hashing. |
| VII. Fail loudly, early | Errors precede expensive calls | PASS. Parse at import/decoration, validate policy against the space during planning, `join="exact"` on zip dims. |
| VIII. Lazy output | One-chunk memory ceiling | PASS. Region writes plus `open_zarr` return; the in-memory mode is opt-in and explicitly non-persistent. |
| IX. Primitive values | Non-primitives rejected | PASS. Space validation enforces the dtype whitelist with a message pointing at the label idiom. |
| X. Dask is a backend | Core independent of dask | PASS. `executors.py` holds the only dask import, behind an optional extra and a lazy import. |

No violations, so Complexity Tracking stays empty.

## Project Structure

### Documentation (this feature)

```text
specs/001-xsweep-v0/
├── plan.md              # This file
├── research.md          # Phase 0: resolved unknowns and rejected options
├── data-model.md        # Phase 1: entities, fields, validation, states
├── quickstart.md        # Phase 1: runnable validation scenarios
├── contracts/
│   ├── public-api.md    # The five public names and their signatures
│   └── contract-dsl.md  # Grammar, tokens, validation rules, error messages
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Created later by /speckit-tasks
```

### Source Code (repository root)

```text
src/xsweep/
├── __init__.py          # Public re-exports only
├── contract.py          # Contract objects, tokeniser, parser, coherence checks
├── policy.py            # SweepPolicy, UNSET sentinel, layered resolution
├── space.py             # Space validation, alignment, point enumeration
├── dedup.py             # Unique-row reduction and expansion
├── plan.py              # Plan, WorkItem, and the planning phase
├── delivery.py          # Argument assembly, scalar conversion, return shaping
├── store.py             # Fingerprint, allocation, region writes, status, lock
├── executors.py         # Executor protocol, serial, process, optional dask
├── sweeper.py           # Sweeper and the @sweep decorator
├── module.py            # SweepModule and __init_subclass__ validation
├── errors.py            # Exception hierarchy
└── py.typed

tests/
├── unit/                # Parser, policy resolution, dedup, cache key, space
├── contract/            # Public API shapes, error messages, definition-time failures
├── property/            # Sacred-property bit-identity suite
└── integration/         # End-to-end sweeps, resume, failure, lock, plan output
```

**Structure Decision**: single `src/` layout package, mirroring radtrans.
Twelve modules, split by phase (space, plan, delivery, store, executors)
rather than by object, because the planning and execution phases must stay
separable (FR-029) and because each phase is independently testable. The
public surface lives only in `__init__.py`; every other module is internal
and free to change.

Three boundaries were deliberately NOT drawn, to keep the line count down
without losing separation of concerns:

- the tokeniser and parser live inside `contract.py` rather than a separate
  grammar module: parsing and coherence checking produce the same error
  catalogue and splitting them scatters those messages across two files;
- fingerprint computation lives inside `store.py`, its only real consumer,
  rather than a `cache.py` of sixty lines;
- there is no second in-memory backend module. The store-less mode uses the
  array library's in-memory store, so both modes run the same code path
  minus fingerprint and lock (research R13). That removes roughly 150 lines
  and, more importantly, makes the equivalence between the two modes true by
  construction rather than a thing to test across two implementations.

**Execution shape**: planning emits ONE stream of work items (FR-037).
Point enumeration, dedup, batching and resume are four transformations of
the same stream and MUST be resolved into it during planning, so that
execution is a single loop with no conditional branches. Implementing them
as four independent passes combined at run time would produce cross-cutting
conditionals and eight interaction cases to test; resolving them once in the
plan collapses that to one.

### Tooling

`pyproject.toml` carries project metadata, pixi workspace, dev feature tasks
(`fmt`, `fmt-check`, `lint`, `type-check`, `test`, `all`) and tool config, as
in radtrans. Differences from radtrans, both deliberate: `strict = true` for
mypy from the start (the codebase is new and small, so strictness costs
nothing now and would cost a lot later), and no pydantic (the contract and
policy are frozen dataclasses; a validation framework would be surface and
dependency for no gain).

## Phase 0 and Phase 1 outputs

- [research.md](./research.md): 14 resolved decisions, including the exact
  `SweepPolicy` field list, the precedence mechanism, the NaN-skip default,
  the mixed-dtype dedup algorithm, the lock and fingerprint formats, and the
  parser strategy. Each records the rejected alternatives.
- [data-model.md](./data-model.md): entities, fields, validation rules and
  the point lifecycle (pending, ok, failed, skipped).
- [contracts/public-api.md](./contracts/public-api.md) and
  [contracts/contract-dsl.md](./contracts/contract-dsl.md): the frozen public
  surface and the DSL grammar with its error catalogue.
- [quickstart.md](./quickstart.md): validation scenarios proving the v0
  scope end to end.

## Complexity Tracking

No constitution violations, so nothing to justify.
