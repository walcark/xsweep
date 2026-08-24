# The contract

A contract describes **one call**: what each variable looks like when it
arrives, and what the call gives back. It says nothing about how many calls
there are, which is why the same contract serves a parameter study and a
satellite scene.

```
loop(tau, ssa)       one value per call, as a native Python scalar
batch(tau, ssa)      several values per call, as a 1-D array
vec(wl @ 8)          a whole axis, or batches of at most 8
const(srf)           context data, handed whole to every call
-> reflectance(wl)   named outputs, with the dims one call produces
```

## loop

The default. Each `loop` variable arrives as a plain Python float, int, str,
bool or datetime. This is what an engine wants when it takes one
configuration at a time.

```python
@sweep("loop(tau, ssa) -> reflectance()")
def layer(tau: float, ssa: float) -> float:
    return engine(tau, ssa)
```

The empty parentheses in `reflectance()` are not decoration: they say the
call produces a scalar. The result's dims come from the sweep, not from the
call.

## vec

A `vec` variable arrives whole, as an `xr.DataArray`, and its dim usually
appears in the output too. Use it when the callee is internally vectorised
along that axis, which is exactly the case where calling it once per element
would be pure waste.

```python
@sweep("loop(aot, pressure) vec(wl) -> reflectance(wl)")
def spectrum(aot: float, pressure: float, wl: xr.DataArray) -> xr.DataArray:
    return solver(tau_of(wl, aot, pressure))
```

`wl` is not a sweep axis here. It is the shape of one call's argument, and
the plan lists it under arguments rather than under the space.

### Batching a vec axis

`vec(wl @ 500)` hands the axis over in pieces of at most 500, which bounds
memory when the axis is large. The same thing can be said in a policy, keyed
by dim, which is the only form available for a multi-dim `vec` variable:

```python
SweepPolicy(chunks={"wl": 500})
```

Batching is a memory decision and must not be a numerical one, so it is only
legal when the output still carries that dim. A callee that reduces over the
axis, or that needs its neighbours (a convolution, a moving average), is not
safe to run on pieces of it, and the contract refuses the marker in that
case, at decoration time.

## batch

Same semantics as `loop`, different delivery. The variables stay sweep
axes — they build the space, feed `dedup`, chunk the store and carry
status — but a call receives several points at once instead of one:

```python
@sweep("batch(aot, rh) vec(wl) -> tdir(wl)")
def transmittance(aot: xr.DataArray, rh: xr.DataArray, wl: xr.DataArray):
    return engine(aot.values, rh.values, wl.values)  # one call, many states
```

Each batched variable arrives as a 1-D `xr.DataArray` over a dim named
`point`, aligned across variables: position `i` of every batched argument
describes the same point. The callable returns its outputs stacked along
that same `point` dim, and xsweep cuts them back apart.

Use it for an engine that is expensive per call *and* takes many
parameter sets at once, which is the shape of most batched solvers: a
fixed setup — building a profile, moving data to a GPU, loading a model —
that only amortises across the batch. `vec` gives the same grouping but
its axes are not sweep axes, so there is nothing for `dedup` to collapse
and an interruption discards the whole run; `loop` keeps both but calls
the engine once per point, paying the setup every time.

How many points a call receives is a run-time decision, so it lives in
the policy rather than the contract:

```python
SweepPolicy(batch_size=64)
```

One number, not one per dim: a group carries whole points, which span the
product of every loop dim rather than positions along one of them.

The output clause is unchanged, and that is the point:

```
loop(aot, rh)  vec(wl) -> tdir(wl)     # one state per call
batch(aot, rh) vec(wl) -> tdir(wl)     # 64 states per call
```

`tdir(wl)` says what **one state** produces. Switching delivery never
rewrites it, which is what makes batching a cost decision and nothing
more. A callable that forgets to stack its results is told so rather than
recorded as a failed point: it does not honour the contract, and retrying
it would fail identically.

Mixing the two clauses is allowed. A `loop` variable stays scalar and is
shared by the whole group, so xsweep only groups points that agree on it.

## const

Context the call needs and the sweep does not vary: a response table, a set
of coefficients, a reference profile. It lives in the space as an ordinary
data variable; the contract is what separates it from the swept ones.

```python
@sweep("loop(aot, pressure) vec(wl) const(srf) -> radiance(band)")
def bands(aot, pressure, wl, srf):
    return (spectrum(aot, pressure, wl) * srf).sum("wl")
```

If a `const` variable shares a dim with something batched elsewhere, it
aligns itself to the active batch rather than causing a shape mismatch.
`const(bias(x, y))` protects named dims from that, for a callee that needs
them whole regardless.

## Outputs

Everything after the arrow is named, with the dims **one call** produces:

```
-> reflectance(), transmittance()      two scalars per call
-> radiance(band)                      one vector per call
```

A single-output contract accepts a bare float or DataArray. A multi-output
one requires a dict keyed by the declared names, or a Dataset. A tuple is
refused on purpose: positional matching would let a swapped pair through
undetected.

An output cannot be named after one of its own dims, because in a Dataset
such a variable *is* that dim's coordinate.

## When it is checked

At decoration time, before any data exists. A malformed contract raises when
the module is imported, not after twenty minutes of engine time.

---

Worked in [Start here](../auto_examples/01_why_a_sweep_library.rst),
[vec](../auto_examples/05_vec_a_whole_spectrum.rst) and
[const](../auto_examples/06_const_and_band_integration.rst).
