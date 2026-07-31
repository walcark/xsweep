# Idioms

Three situations xsweep deliberately solves with a convention rather than
with machinery. All are tested in `tests/integration/test_idioms.py`.

## Replication: the seed carrier variable

A Monte-Carlo convergence study repeats the same computation N times and
reads the spread. The obvious spelling does not work:

```python
# WRONG: "rep" is a bare dim, and contracts loop over VARIABLES
@sweep("loop(aot, rep) -> rho()")
```

Worse, if it did work it would be a trap: twenty identical calls are
legitimately collapsed by the cache and by dedup, and the standard deviation
would come out as exactly zero. The convergence study would silently measure
nothing.

The idiom is a carrier variable that makes each repetition genuinely
distinct:

```python
@sweep("loop(aot, seed) -> rho()")
def mc(aot: float, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    return simulate(aot, rng)


space = xr.Dataset(
    {
        "aot": ("aot", [0.1, 0.3]),
        "seed": ("rep", np.arange(20)),  # the carrier makes the "rep" dim
    }
)

result = mc(space)  # rho(aot: 2, rep: 20)
noise = result.rho.std("rep")
```

Each repetition is now individually reproducible, individually cacheable,
and resumable like any other point.

## Comparing two versions of the same physics

`version` selects CODE, not data, so it is never a sweep axis. Bumping it
changes the store fingerprint, which means the previous results are refused
rather than mixed in. They are not lost, though: keep one store per version
and compare them explicitly.

```python
@sweep("loop(a) -> rho()", version="1")
def rho(a: float) -> float: ...


v1 = rho(space, policy=SweepPolicy(store="runs/rho_v1.zarr"))
# ... fix a bug in the function body, bump to version="2" ...
v2 = rho(space, policy=SweepPolicy(store="runs/rho_v2.zarr"))

compared = xr.concat([v1.rho, v2.rho], dim=pd.Index(["1", "2"], name="version"))
delta = compared.diff("version")  # where and by how much the fix changed things
```

This is the regression check after a bug fix, and it is the reason the store
is never namespaced per version automatically: choosing the paths is what
makes the comparison explicit.

## Object-valued parameters: label plus cache token

Sweep coordinates must be primitives. An object breaks fingerprinting, store
serialisation and unique-row extraction alike. Sweep a label instead, and
pass the mapping as a static that can describe itself to the cache:

```python
class Profiles:
    def __init__(self, tag: str, table: dict[str, float]) -> None:
        self.tag, self.table = tag, table

    def __cache_token__(self) -> str:
        return self.tag  # changing the table means changing the tag


@sweep("loop(profile) -> rho()", version="1")
def rho(profile: str, *, profiles: Profiles) -> float:
    return compute(profiles.table[profile])


space = xr.Dataset({"profile": ("profile", ["afgl_ms", "afgl_t"])})
rho(space, policy=SweepPolicy(store="runs/rho.zarr"), profiles=Profiles("v1", table))
```

Without `__cache_token__` the static is refused outright, because a cache
that cannot tell when its inputs changed is worse than no cache.

---

The seed carrier is worked in
[replication](../auto_examples/10_replication_and_the_seed.rst), which also
shows the collapse that happens when the repetitions are not made distinct.
