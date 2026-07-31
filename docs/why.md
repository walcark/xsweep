# Why xsweep

## The situation it is for

You have a function that computes one point, and you need a grid of them.

If that function is a closed form, stop reading: numpy already does this.
Write the expression over the whole grid and it runs in one vectorised pass,
faster than any sweep library could dispatch the calls. `xr.apply_ufunc`
handles the xarray bookkeeping around it, and it does that well.

xsweep is for the other kind of function, the one that has no array shape to
broadcast over:

- a **Monte-Carlo solver**, where one point is a loop whose length depends on
  the random numbers it draws;
- an **iterative scheme** that stops when it has converged, at a different
  iteration for every input;
- an **external engine**, a subprocess, a licensed binary, a GPU kernel, that
  takes one configuration at a time and hands back one result;
- anything that is vectorised along *some* of its axes and not others: a
  spectral engine takes one atmosphere and returns a whole spectrum.

For these, the loop is not a failure of imagination. It is what the problem
is. And once you write that loop, it grows.

## What the loop grows into

```python
results = np.empty((len(taus), len(albedos)))
for i, tau in enumerate(taus):
    for j, ssa in enumerate(albedos):
        results[i, j] = engine(tau, ssa)
```

Ten lines later this has a `try/except` so one bad point does not lose the
other 4999. Then a check for whether the answer is already on disk, because
the run takes six hours and you would like to be able to restart it. Then a
dictionary of already-seen inputs, because the scene has 90000 pixels and
only 60 distinct atmospheres. Then a print statement, because you would like
to know how far along it is. Then a `ProcessPoolExecutor`. Then a way to tell
which of the saved results came from before you fixed that bug.

None of that is your science. All of it is the same every time.

## What xsweep replaces it with

```python
@sweep("loop(tau, ssa) -> reflectance()")
def layer(tau: float, ssa: float) -> float:
    return engine(tau, ssa)


result = layer(space, policy=SweepPolicy(store="runs/layer.zarr", dedup=True))
```

Four things come with that, and they are the whole of the library:

**You can see what a run will cost before starting it.** `layer.explain(space)`
resolves the sweep and stops without calling the engine once: how many points,
how many of them the store already has, what one call will receive. Multiply
by the cost of a single call and you have your estimate. It is also where an
accidental Cartesian product shows up as a point count with five too many
digits, which is the cheapest possible place to find it.

**The result is the cache.** Points stream into a zarr store as they land,
with a `status` variable saying what happened to each. Interrupt at point
4000 of 5000 and run the same line again: only the missing points are
computed. Run it against a finished store and it makes no calls at all.

**Repeats are computed once.** `dedup=True` collapses identical input rows
before calling anything and expands the results back afterwards. On a
classified satellite scene that is routinely a hundredfold.

**The shape of the sweep comes from your data.** Variables sharing a dim vary
together; variables on distinct dims multiply. The same contract reads a
parameter study and a pixel map without a character changing, because the
arrays already say which one you meant.

## What it is not

It is not a scheduler, not a workflow engine, not a cluster tool. One writing
process owns a store at a time. It does not know anything about your physics,
and it will not check your units.

And it is not free: every point carries about a tenth of a millisecond of
bookkeeping ([the benchmarks](benchmarks.md) measure it). Against an engine
that costs seconds, that is nothing. Against an engine that costs
microseconds, it is everything, and you should be using numpy.

## Where to go next

The [example gallery](auto_examples/index.rst) is the fastest way in: ten
pages, one idea each, all driving the same Monte-Carlo solver, with the real
output of every run printed underneath. The guide covers the same ground more
slowly, starting with [the contract](guide/contract.md).
