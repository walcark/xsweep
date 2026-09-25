# xsweep

<p align="center"><em>Content-agnostic parameter sweeps for xarray: turn an expensive point function into a gridded, cached, and resumable computation.</em></p>

<p align="center">
  <a href="https://github.com/walcark/xsweep/actions/workflows/ci.yml"><img src="https://github.com/walcark/xsweep/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/walcark/xsweep"><img src="https://codecov.io/codecov/gh/walcark/xsweep/branch/main/graph/badge.svg"></a>
  <a href="https://walcark.github.io/xsweep/"><img src="https://github.com/walcark/xsweep/actions/workflows/docs.yml/badge.svg"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue">
  <a href="https://pixi.sh"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json"></a>
  <a href="https://mypy-lang.org/"><img src="https://img.shields.io/badge/mypy-checked-2a6db2"></a>
  <img src="https://img.shields.io/badge/tested%20with-pytest-0a9edc?logo=pytest&logoColor=white">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue"></a>
</p>

<p align="center">
  <strong><a href="https://walcark.github.io/xsweep/">Documentation and examples</a></strong>
</p>

## What is xsweep?

Many scientific computations are naturally expressed as a **point function**:

> given one configuration, run an expensive computation and return one result.

For example:

- a Monte-Carlo simulation,
- an iterative solver whose stopping point depends on the input,
- an external executable that processes one configuration at a time,
- or any computation that cannot conveniently be vectorised over the parameters of interest.

Running such a function over a parameter space quickly requires more than a `for` loop:

- defining the parameter grid,
- avoiding duplicate evaluations,
- caching completed points,
- limiting memory usage,
- resuming interrupted runs,
- tracking progress and failures,
- and deciding how work should be batched or parallelised.

**xsweep provides this machinery once, while leaving the actual computation completely opaque.**

It takes a point function and lifts it into an xarray computation.

```python
import numpy as np
import xarray as xr

from xsweep import sweep


@sweep(
    "loop(tau, ssa) -> reflectance()",
    store="runs/layer.zarr",
    version="1",
)
def layer(tau: float, ssa: float) -> float:
    return monte_carlo(
        tau,
        ssa,
        n_photons=int(1e6),
    )


space = xr.Dataset(
    {
        "tau": ("tau", np.linspace(0.1, 3.0, 40)),
        "ssa": ("ssa", [0.9, 0.95, 1.0]),
    }
)

print(layer.explain(space))  # inspect the work without executing it

result = layer(space)
# reflectance(tau=40, ssa=3)
```

The function is called once for each point of the parameter space.

The resulting values are assembled into an xarray object and persisted to the configured store.

Running the same sweep again reuses the existing results. If the computation is interrupted, relaunching it evaluates only the missing points.

---

## The three ideas

### 1. Let xarray define the parameter space

xsweep does not invent another grid or parameter-space abstraction. It uses xarray's existing dimension semantics.

Variables sharing a dimension vary together:

```text
x: (time)
y: (time)
```

describe pairs `(x[i], y[i])`.

Variables on independent dimensions form a Cartesian product:

```text
x: (x)
y: (y)
```

produces every `(x[i], y[j])` combination.

This means the same mechanism can describe anything from a small parameter study to a multidimensional scientific dataset without introducing a separate notion of "sweep dimensions".

---

### 2. The contract describes the call, not the data

The sweep contract specifies how each variable participates in a call:

- `loop` — one value per call,
- `vec` — a whole axis is passed to the function,
- `const` — context passed through unchanged,
- `->` — names the values produced by one call.

For example:

```text
loop(tau, ssa) -> reflectance()
```

means:

> call the function once for each `(tau, ssa)` combination and store the returned value as `reflectance`.

The contract is deliberately independent of the scientific meaning of the data. The same mechanism can therefore describe a parameter sweep, a sensitivity analysis, or a computation over a satellite scene.

---

### 3. Cache, output, and resume are one artefact

xsweep persists sweep results directly into a Zarr store together with execution status.

This turns the output store into the state of the computation:

```text
                  ┌──────────────┐
                  │ parameter    │
                  │    space     │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   missing    │
                  │    points    │
                  └──────┬───────┘
                         │
                  execute only these
                         │
                         ▼
              ┌──────────────────────┐
              │      Zarr store      │
              │                      │
              │ values + status      │
              └──────────────────────┘
```

Consequently, the same mechanism provides:

- **memoisation** — completed points are reused,
- **bounded memory** — results need to remain in RAM,
- **resumability** — interrupted sweeps continue from their current state,
- **deduplication** — identical configurations need not be evaluated twice,
- **parallel execution** — independent points can be dispatched separately.

Execution details such as batching, deduplication strategy, and executor choice are deliberately kept separate from the scientific function.

A release gate ensures that these execution choices do not change the numerical result.

---

## Why not just use a loop?

A simple loop is often exactly what you need:

```python
for tau in taus:
    for ssa in ssas:
        result[tau, ssa] = model(tau, ssa)
```

The problem starts when the computation becomes expensive or long-lived.

You then need to answer questions such as:

- Where are the completed results stored?
- What happens if the process is interrupted?
- How do I restart without recomputing everything?
- What if two configurations are identical?
- How do I run points in parallel?
- How much work remains?
- How do I inspect the computation before launching it?
- How do I keep the same code while changing the executor?

xsweep moves these concerns out of the scientific function and into the sweep infrastructure.

---

## Install

```bash
pip install xsweep
```

xsweep requires Python 3.11+ and uses:

- [xarray](https://xarray.dev/)
- [NumPy](https://numpy.org/)
- [Zarr](https://zarr.dev/)

Dask is an optional execution backend. The core package does not depend on Dask.

### Development

To work on xsweep itself:

```bash
pixi install
pixi run -e dev all
```

---

## Documentation

The complete documentation, examples, and API reference are available at:

**https://walcark.github.io/xsweep/**

- **[Why xsweep](https://walcark.github.io/xsweep/why.html)** — motivation, design principles, and when xsweep is not the right tool.
- **[Examples](https://walcark.github.io/xsweep/auto_examples/index.html)** — ten focused examples built around a real Monte-Carlo solver, including the actual output of each run.
- **[Guide](https://walcark.github.io/xsweep/guide/contract.html)** — concepts and sweep contracts.
- **[API reference](https://walcark.github.io/xsweep/reference/api.html)** — complete Python API.
- **[Limitations](https://walcark.github.io/xsweep/reference/limitations.html)** — what the v0 design deliberately does not attempt to do.
- **[Benchmarks](https://walcark.github.io/xsweep/benchmarks.html)** — execution and per-point cost across releases.

---

## Status

**v0 — experimental**

The API is still evolving, but the core design is intentionally small:

> **describe the computation, provide the parameter space, and let xsweep manage execution state.**

---

## License

[Apache License 2.0](LICENSE).
