# Quickstart and validation scenarios: xsweep v0

How to set the project up and how to prove the v0 scope works. Each scenario
maps to requirements and success criteria; none of them needs a real
radiative-transfer engine, since the library is content-agnostic and cheap
synthetic callees exercise every path.

## Setup

```bash
cd ~/dev/current/xsweep
pixi install
pixi run -e dev all      # fmt, lint, type-check, test
```

Individual tasks: `pixi run -e dev fmt`, `lint`, `type-check`, `test`.
Formatting, linting and typing are always run through pixi tasks, never by
hand.

## Scenario 1: the core lift (US1, FR-001 to FR-013, SC-001)

Sweep a synthetic function over a small Cartesian space with a store.
Expect: result dims equal the union of loop variable dims; one call per
point; loop values arriving as native Python scalars; a second identical run
performing zero calls.

Then repeat with maps sharing dims (`aot(y, x)`, `rh(y, x)`) and expect zip
semantics, one call per pixel, and no Cartesian blow-up.

Check the run summary is logged with computed, cached, failed, skipped and
elapsed counts, and that a sweep with no store named says so explicitly.

## Scenario 2: failure and resume (US2, FR-015 to FR-019, SC-002)

Run a sweep whose callee raises on a known subset. Expect NaN slices and
`status == failed` there, `ok` elsewhere, and completion. Relaunch and expect
only the failed points recomputed. Kill the process mid-run, relaunch, and
expect exactly the missing points recomputed. Switch to `on_error="raise"`
and expect the original exception to surface.

Then extend an axis of the space and expect a refusal naming the mismatch.

## Scenario 3: dedup (US3, FR-021, SC-004)

Sweep a map with known duplicates twice, with and without `dedup=True`.
Expect identical results and a call count equal to the number of unique
rows. Repeat with mixed dtypes (float plus str loop variables) and with
`dedup=("y", "x")` on a space that also has `time`. Name a dim that does not
exist and expect an immediate failure listing the available dims.

## Scenario 4: batching and context data (US4, FR-008 to FR-010)

Sweep with `vec(wl @ 8)` over 20 wavelengths and expect batches of 8, 8 and
4, reassembled in order. Add `const(srf)` and expect the whole array in every
call. Declare a contract batching a dim absent from the outputs and expect
rejection at decoration. Put `@ N` on a multi-dim vec variable and expect the
message pointing at the policy form. Sweep two maps as multi-dim vec, once
whole and once with `chunks={"y": 500, "x": 500}`, and expect identical
results with different call counts.

## Scenario 5: modules (US5, FR-023)

Define a subclass without a contract and expect the error at class
definition. Define an intermediate base with `abstract=True` and expect
silence. Define a valid subclass and expect results identical to the
equivalent decorated function.

## Scenario 6: the plan (FR-029 to FR-031, SC-011)

Call `explain` on each of the scenarios above and check that no call of the
wrapped function occurs (a counter in the callee proves it), that call
counts, batch division, cache state and shapes are reported, and that an
undetermined output size is reported as undetermined rather than probed.

Then sweep a contract with undeclared output sizes and check the probe
result is KEPT: a one-point sweep must leave the call counter at one, not
two. Discarding it would waste an engine call, which is the whole argument
behind the decision.

Then check the three costly misconfigurations are visible: dedup forgotten
on a map sweep (point count in the millions), an accidental Cartesian
product where a zip was meant (one axis per variable), and a batch size of
one on a vectorisable axis.

## Scenario 7: the sacred property (FR-022, SC-003)

The release gate. Sweep the same space with every combination of dedup on
and off, batch sizes varying, serial and process executors, store and
in-memory, and assert bit-identical results throughout. Parametrised so that
adding a policy field means adding a parameter, not a new test.

## Scenario 8: store discipline (FR-014 to FR-033, SC-008, SC-010)

Check the chunk grid matches the contract (one loop point per region), that
concurrent writers never share a chunk under the process executor, that a
second writing process fails within seconds naming the owner, that
`force_unlock=True` overrides it, that reading the store during a run works,
and that peak memory stays bounded by one chunk on a sweep larger than
memory.

## Scenario 9: primitives and idioms (FR-004, FR-025, FR-028)

Sweep string and datetime64 axes and expect first-class support. Sweep an
object-valued variable and expect rejection pointing at the label idiom.
Sweep a `seed` carrier variable on a `rep` dim and expect distinct results
per repetition with a non-zero standard deviation. Produce two stores under
two versions and concatenate them along an explicit axis.

## Scenario 11: fail loudly, early (SC-005)

One parametrised test walking every edge-case row whose v0 policy is to
fail: mismatched coordinates on a zip dim, heterogeneous output grids, a
fingerprint mismatch, a changed space on resume, batching a reduced dim, an
object-valued coordinate, and unknown dims in `dedup` or `chunks`. Each must
raise before the first call of the wrapped function, or at definition time
where the spec says so, with a call counter proving it.

## Scenario 10: the adjeff shape (SC-007)

Express an adjeff sampler with its existing point-function signature
unchanged and check the output matches what the current implementation
produces.
