# The class facade

A bare `@sweep` function has nowhere clean to put state built once and reused
across calls: an engine handle, a loaded lookup table, an open connection. A
closure breaks the pickling the process executor needs, and a module global is
not much better.

`SweepModule` gives that state a home, modelled on the `torch.nn.Module`
idiom:

```python
class RhoAtm(SweepModule):
    contract = "loop(aot, rh) vec(wl @ 8) -> rho_atm(wl)"

    def __init__(self, policy=None, engine=None):
        super().__init__(policy)
        self.engine = engine  # built once, reused by every call

    def forward(self, aot, rh, wl, *, n_ph):
        return self.engine.run(aot, rh, wl, n_ph)


mod = RhoAtm(
    SweepPolicy(store="runs/rho.zarr", executor="process"), engine=load_engine()
)
result = mod(space, n_ph=int(1e6))
```

`contract` is a class attribute, so a malformed one raises when the module is
imported rather than after twenty minutes of engine time.

`forward` stays pure physics and is testable with no sweep involved:

```python
RhoAtm(engine=engine).forward(0.1, 50.0, wl, n_ph=100)
```

`__call__` does the orchestration, and `explain` works the same as on a
decorated function.

## When to reach for it

This is also the combination that most needs `executor="process"`: something
expensive, built once, then run in parallel. The instance's state travels
into the worker processes intact rather than being rebuilt per worker.

If a bare function is enough, the decorator is simpler and gets you the same
contract and policy machinery. Reach for the class when `__init__` earns its
keep.

---

Worked in [SweepModule](../auto_examples/08_module_with_state.rst).
