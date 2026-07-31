# Policy

Everything in `SweepPolicy` is a cost decision, never a value one. A
parametrised suite checks that the result is bit-identical across policy
combinations as a release gate.

Three things are decided at three different moments, and keeping them apart
is most of what makes a sweep readable a year later:

- **Contract** is physics, fixed when the function is written.
- **Policy** is run configuration, set at the decorator, the instance, or the
  call, in that order of increasing precedence.
- **Data** is the space and the statics, given at the call.

```python
@sweep("loop(tau, ssa) -> reflectance()", dedup=True)  # default
def layer(tau, ssa): ...


layer(space)  # deduplicated
layer(space, policy=SweepPolicy(dedup=False))  # not
```

## The knobs

`store`
: A path. Turns the result into a cache and a resume point; see
  [the store](store.md).

`dedup`
: Compute each distinct input row once; see [deduplication](dedup.md).

`chunks={"wl": 500}`
: Override a `vec` batch size, keyed by dim. `"auto"` sizes it from a memory
  budget instead of a number you pick. The dim still has to be named,
  because only you know whether the callee is safe to run on pieces of that
  axis.

`store_chunks={"y": 256}`
: Size the store's loop-dim chunk grid. Sized from a memory budget by
  default and not worth touching unless you have measured a reason.

`executor="process"`, `max_workers=N`
: Run points in parallel across processes. The callable has to be picklable,
  which in practice means defined at module level. A function defined inside
  another function is not.

`on_error="raise"`
: Stop at the first failure instead of recording it and continuing. Right
  during development, wrong for a long unattended run.

`retries=N`
: Give a call another attempt before recording a failure. For a transient
  fault, not for a bug.

`skip_where=lambda point: ...`
: Exclude points from a predicate on their loop values, without ever calling
  the function.

## Choosing between them

There is no general answer, but the shape of one is:

- deduplication wins whenever the space repeats, and costs a little when it
  does not;
- processes win once a call is expensive enough to dwarf the cost of
  starting one, and the speed-up is always less than the worker count;
- a store costs a fraction of a millisecond per point and buys you the
  ability to stop.

[`explain`](plan.md) tells you the call count for any combination before you
run it, and [the benchmarks](../benchmarks.md) measure what the bookkeeping
itself costs.

---

Worked in [policy](../auto_examples/07_policy_never_changes_the_result.rst).
