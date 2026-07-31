# The store

Point a policy at a path and the same sweep gains a cache, bounded memory and
resumability, with no change to the function or the contract:

```python
result = layer(space, policy=SweepPolicy(store="runs/layer.zarr"))
```

That is one mechanism, not three. Results stream into a zarr store as they
land, one region per point, and a `status` variable rides along saying what
happened to each.

## status

| code | label | meaning |
|---|---|---|
| 0 | `pending` | never attempted |
| 1 | `ok` | computed and stored |
| 2 | `failed` | the call raised; the value is `nan` |
| 3 | `skipped` | excluded by `skip_where` |

It comes back with the result and is written to the store, so "what did that
run actually do" is answerable without re-running anything, including while
the run is still going: read the status variable from the store from another
process.

## Resuming

Only points without an `ok` status are computed. Interrupt a long sweep and
call it again with the same store; a fresh run against a finished store makes
zero calls and costs a read.

```python
layer(space, policy=policy)   # 60 calls
layer(space, policy=policy)   # 0 calls
```

Because the points that succeeded are read back rather than recomputed, they
are bit-identical rather than merely close. With a stochastic engine that is
the difference between a cache and a coincidence.

## What a store is tied to

A store carries a fingerprint of the contract, the `version` string and the
statics, and refuses to open under a different one. This is deliberate: it is
what stops results computed before a bug fix from mixing silently with
results computed after it. Bump `version` when the physics changes, and keep
one store per version; [idioms](idioms.md) shows the comparison that makes
possible.

Extending an axis and resuming is refused rather than silently misaligned.
There is no store reindexing.

## One writer at a time

A lock enforces one writing run per store. Readers are never blocked. Since a
dead process on another host cannot be probed reliably, `force_unlock=True`
is the honest override rather than a staleness heuristic.

Multi-process cooperation, where cluster tasks claim points atomically, is
out of scope. The documented answer is to partition the space at submission,
one store per task, and merge afterwards.

## The chunk grid

The store's loop-dim chunk grid is sized from a memory budget (64 MB by
default), not pinned to one point per chunk, and writes are buffered into it.
A size-1 grid would turn every point's write into a read-modify-write of that
whole chunk, which measures about eighteen times slower on a 40000-point
sweep. Override it per dim with `store_chunks={"y": 256}` if you need to.

The trade is that a wider chunk keeps more computed-but-unflushed points in
memory. Nothing un-flushed was ever persisted, so a hard kill loses only what
the open chunk was holding, and it is recomputed rather than lost. A
`PointFailed` raise flushes every open chunk before propagating.

---

Worked in [the store](../auto_examples/02_store_cache_and_resume.rst) and
[failures](../auto_examples/09_when_points_fail.rst).
