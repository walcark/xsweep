---
layout: landing
---

# xsweep

Content-agnostic parameter sweeps for xarray: lift an expensive point
function into a gridded, cached, resumable computation.

```{button-ref} why
:color: primary
:class: sd-rounded-pill

Why xsweep
```

```{button-link} https://github.com/walcark/xsweep
:color: secondary
:outline:
:class: sd-rounded-pill

GitHub
```

Some functions cannot be vectorised along the dims you want to sweep: a
Monte-Carlo solver, an iterative scheme whose stopping point depends on the
data, an external engine invoked one configuration at a time. numpy has
nothing to offer there, and the loop you write by hand quietly grows a cache,
a resume path, and a way to guess how long the whole thing will take.

xsweep is that loop, written once.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Know the cost first
:link: guide/plan
:link-type: doc

`explain()` resolves the whole sweep and stops. Axes, call count, batches,
what the store already has, and what a run will actually compute: all before
the first call.
:::

:::{grid-item-card} Cache, output and resume in one artefact
:link: guide/store
:link-type: doc

Results stream into a zarr store as they land. Interrupt at point 4000 of
5000 and relaunch: only the missing points are recomputed.
:::

:::{grid-item-card} Never compute the same point twice
:link: guide/dedup
:link-type: doc

A satellite scene has far fewer distinct parameter rows than pixels.
Deduplication computes each unique row once and expands the result back.
:::

:::{grid-item-card} Semantics from xarray, not from a DSL
:link: guide/semantics
:link-type: doc

Variables sharing a dim vary together; variables on distinct dims multiply.
The arrays already say it, so the contract never repeats it.
:::

::::

```{toctree}
:hidden:

why
install
```

```{toctree}
:hidden:
:caption: Guide

guide/contract
guide/semantics
guide/plan
guide/store
guide/dedup
guide/policy
guide/module
guide/idioms
```

```{toctree}
:hidden:
:caption: Examples

auto_examples/index
benchmarks
```

```{toctree}
:hidden:
:caption: Reference

reference/api
reference/limitations
reference/design
reference/findings
```
