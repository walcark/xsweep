# The contract

A contract describes **one call**: what each variable looks like when it
arrives, and what the call gives back. It says nothing about how many calls
there are, which is why the same contract serves a parameter study and a
satellite scene.

```
loop(tau, ssa)       one value per call, as a native Python scalar
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
