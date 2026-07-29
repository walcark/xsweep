# xsweep

Minimal, content-agnostic parameter sweeps for xarray: lift an expensive
"point function" into a gridded, cached, resumable xarray computation.

The problem it solves is not computing a grid, which xarray already does. It
is everything around an expensive call: computing each point exactly once,
keeping the results, surviving a failure at point 4000 of 5000, and knowing
what a run will cost before launching it.

```python
import numpy as np, xarray as xr
from xsweep import sweep, SweepPolicy


@sweep(
    "loop(aot, rh, sza) vec(wl @ 8) -> tdir(wl)", store="runs/tdir.zarr", version="1"
)
def tdir(
    aot: float, rh: float, sza: float, wl: xr.DataArray, *, n_ph: int
) -> xr.DataArray:
    return run_engine(aot, rh, sza, wl, n_ph)  # subprocess, GPU kernel, ...


space = xr.Dataset(
    {
        "aot": ("aot", [0.05, 0.1, 0.3]),
        "rh": ("rh", [30.0, 70.0]),
        "sza": ("sza", [0.0, 30.0, 60.0]),
        "wl": ("wl", np.arange(400.0, 900.0, 5.0)),
    }
)

print(tdir.explain(space, n_ph=int(1e6)))  # what it will cost, zero calls
result = tdir(space, n_ph=int(1e6))  # tdir(aot: 3, rh: 2, sza: 3, wl: 100)
```

Re-running that sweep makes zero calls. Interrupt it and relaunch: only the
missing points are recomputed.

## The three ideas

**Semantics come from xarray, not from a second description.** Variables
sharing a dim vary together (zip); variables on distinct dims multiply
(Cartesian product). The arrays already encode it, so the contract never
repeats it.

```python
# a 1000 x 1000 map: one million zipped points, not a trillion product ones
space = xr.Dataset({"aot": (("y", "x"), aot_map), "rh": (("y", "x"), rh_map)})
```

**The contract describes the CALL, not the data.** It says how each variable
is consumed, and names what one call produces:

```
loop(aot, rh, sza)   one value per call, as a native Python scalar
vec(wl @ 8)          a whole axis, or batches of at most 8
const(srf)           context data, handed whole to every call
-> tdir(wl)          named outputs, with their call-level dims
```

The same data admits different contracts depending on the callee: a
scalar-only engine wants `loop(A, B) -> C()`, an internally vectorised one
wants `vec(A, B) -> C(y, x)`. The contract encodes a property of the physics.

**Cache, output and resume are the same artefact.** Results stream into a
pre-allocated zarr store, one region per point, with a `status` sidecar
variable. That single mechanism gives memoisation, bounded memory,
resumability and safe parallel writes at once.

## Policy never changes the result

Deduplication, batch sizes, executors and the store are cost decisions. They
never move a value, a dim or a shape, and a parametrised suite enforces that
as a release gate.

```python
result = rho(space)  # one call per pixel
result = rho(space, policy=SweepPolicy(dedup=True))  # one per unique row
```

## As a class

```python
class RhoAtm(SweepModule):
    contract = "loop(aot, rh, sza) vec(wl @ 8) -> rho_atm(wl)"

    def forward(self, aot, rh, sza, wl, *, n_ph):
        return run_engine(aot, rh, sza, wl, n_ph)


mod = RhoAtm(SweepPolicy(store="runs/rho.zarr", dedup=True, executor="process"))
result = mod(space, n_ph=int(1e6))
```

Physics in `forward`, orchestration in `__call__`, run configuration at
instantiation. A malformed contract raises when the module is imported, not
after twenty minutes of engine time.

## Install and develop

```bash
pixi install
pixi run -e dev all      # fmt, lint, type-check, test
```

Python 3.11+, with xarray, zarr and numpy. Dask is an optional executor
backend and no core path imports it.

## Documentation

- [Design reference](docs/design/xsweep.md): why it is built this way, the
  worked examples, and the alternatives that were rejected.
- [Idioms](docs/idioms.md): replication seeds, comparing versions,
  object-valued parameters.
- [Limitations](docs/limitations.md): what v0 deliberately does not do.
- [Specification](specs/001-xsweep-v0/): requirements, plan and tasks.

## Status

v0. Motivating consumers: adjeff (Smart-G sweeps, replacing its internal
`SweepBundle` and `UniqueIndex`), radtrans (the engine-agnostic half of its
sweep layer), and short sensitivity studies in Earth observation.
