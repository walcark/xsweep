# Semantics come from your data

The contract never says how many points there are, or whether two variables
vary together. That is already written in the dims of the `Dataset` you pass:

```python
# distinct dims: Cartesian product, 3 x 2 = 6 points
xr.Dataset({"tau": ("tau", [0.1, 0.5, 1.0]), "ssa": ("ssa", [0.9, 1.0])})

# shared dims: zip, one point per pixel, not a product
xr.Dataset({"tau": (("y", "x"), tau_map), "ssa": (("y", "x"), ssa_map)})
```

A contract written as `loop(tau, ssa) -> reflectance()` serves both unchanged.
A 1000 x 1000 map is a million zipped points, not a trillion product ones.

## Why it works this way

There is exactly one description of how the data is laid out, and it is the
data. A second description in the contract could only ever agree with the
arrays or disagree with them, and the disagreeing case is a bug that no
amount of validation makes pleasant.

It also means the same function serves a sensitivity study and an image
without a wrapper in between, which is usually what you wanted.

## The mistake it makes possible

Two variables meant to vary together, declared on different dims by accident,
silently become every combination of themselves:

```python
# meant: 200 pixels. got: 40000 points.
xr.Dataset({"tau": ("pixel", tau_flat), "ssa": ("point", ssa_flat)})
```

Nothing about the code looks wrong. The point count is the only symptom, and
[`explain`](plan.md) reports it without making a single call, which is the
cheapest place there is to catch it.

## What a value may be

Sweep coordinates are primitives: float, int, str, bool, datetime64. An
object breaks fingerprinting, store serialisation and unique-row extraction
alike. Sweep a label instead and pass the object as a static that can
describe itself to the cache; see [idioms](idioms.md).

Axes are rectangular. A wavelength support that depends on the band does not
fit; the clean workaround is a union grid with zero weights in the response.

---

Worked in [Zip or product](../auto_examples/03_zip_or_product.rst).
