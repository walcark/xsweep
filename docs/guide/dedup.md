# Deduplication

A satellite scene rarely has as many distinct parameter rows as it has
pixels. Retrieval products come off a classifier or a coarse grid, so
neighbouring pixels repeat.

```python
result = layer(space, policy=SweepPolicy(dedup=True))
```

Identical input rows are collapsed before anything is called, each distinct
one is computed once, and the results are expanded back onto every position
that shares them.

## What it costs and what it buys

The plan reports the distinct-row count before running:

```text
SPACE
y      tau, ssa      300
x      tau, ssa      300
points              90000
dedup  60 unique (89940 duplicates)
```

With an expensive engine, removing 89940 calls is the whole story. The
duplicate positions still have to be filled, which is a gather in memory and
a batched write pass into a store, and that cost does not go away. So:

- **expensive callee, repeating scene**: near-total saving;
- **free callee**: deduplication is pure overhead, and the plan tells you so
  before you pay for it.

The distinct-row count is a property of the scene rather than of its size, so
the saving grows as the image does.

## It never changes a value

Deduplication is a cost decision. It moves no value, no dim and no shape, and
a parametrised suite enforces bit-identical results across policy
combinations as a release gate. Duplicated positions are copied, not
recomputed, so even a stochastic engine gives exactly equal results with and
without it.

That property is also a trap worth knowing about: repetitions that are
*meant* to differ have to differ in their inputs, or deduplication will
correctly collapse them into one. That is what the seed carrier variable in
[idioms](idioms.md) is for.

## What it does not do

Unique rows are extracted in memory. There is no streaming deduplication, so
a space too large to hold its own loop variables is out of scope.

---

Worked in [dedup on a scene](../auto_examples/04_dedup_on_a_scene.rst).
