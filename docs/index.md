---
layout: landing
---

# xsweep

Content-agnostic parameter sweeps for xarray: lift an expensive point
function into a gridded, cached, resumable computation.

```{button-ref} guide
:color: primary
:class: sd-rounded-pill

Read the guide
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
:link: guide
:link-type: doc

`explain()` resolves the whole sweep and stops. Axes, call count, batches,
what the store already has, and what a run will actually compute: all before
the first call.
:::

:::{grid-item-card} Cache, output and resume in one artefact
:link: guide
:link-type: doc

Results stream into a zarr store as they land. Interrupt at point 4000 of
5000 and relaunch: only the missing points are recomputed.
:::

:::{grid-item-card} Never compute the same point twice
:link: guide
:link-type: doc

A satellite scene has far fewer distinct parameter rows than pixels.
Deduplication computes each unique row once and expands the result back.
:::

:::{grid-item-card} Semantics from xarray, not from a DSL
:link: guide
:link-type: doc

Variables sharing a dim vary together; variables on distinct dims multiply.
The arrays already say it, so the contract never repeats it.
:::

::::

```{toctree}
:hidden:
:caption: Guide

guide
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
