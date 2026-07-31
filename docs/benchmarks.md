# Benchmark evolution across versions

The examples in the gallery use an engine that is expensive on purpose,
because that is the situation xsweep is for. These measurements want the
opposite: **a callee as close to free as Python allows, over as many points
as is practical**, so that what gets timed is xsweep's own bookkeeping rather
than somebody's physics. If the engine dominated, the number would tell you
about the engine.

The cases are frozen, so the same work is timed release after release. They
run only on a deliberate
[`pixi run -e dev bench`](https://github.com/walcark/xsweep/blob/main/benchmarks/README.md),
never on a docs build, and are also rendered as a plain table in
[TIMING.md](https://github.com/walcark/xsweep/blob/main/benchmarks/results/TIMING.md).

The chart plots **cost per point**, which survives a case being resized in a
way that raw wall time does not. Both are on the hover line, along with the
host: times are only comparable against other runs on the same machine, and
say nothing across different hardware.

```{raw} html
<div id="benchmarks-chart">
  <p>
    <label for="benchmarks-case-select">Case:</label>
    <select id="benchmarks-case-select"></select>
  </p>
  <canvas id="benchmarks-canvas" width="760" height="360"
          style="max-width: 100%; height: auto;"></canvas>
  <p id="benchmarks-info" style="font-family: monospace; font-size: 0.9em;"></p>
  <p id="benchmarks-empty" style="display:none;">
    No recorded measurements yet.
  </p>
</div>
<script src="_static/benchmarks.js"></script>
```
