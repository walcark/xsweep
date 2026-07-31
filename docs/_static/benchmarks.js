// Renders the per-case cost-vs-version chart on the benchmarks page.
// Vanilla JS, no dependency: the dataset is small (few points per case) and
// this repo has no other JS tooling, so a small hand-rolled canvas chart is
// simpler than adding a charting library.
(function () {
  "use strict";

  const CANVAS_ID = "benchmarks-canvas";
  const SELECT_ID = "benchmarks-case-select";
  const EMPTY_ID = "benchmarks-empty";
  const INFO_ID = "benchmarks-info";
  const DATA_URL = "_static/benchmarks_history.json";

  const PADDING = { top: 24, right: 24, bottom: 56, left: 64 };
  const POINT_RADIUS = 4;

  function humanize(caseName) {
    return caseName.replace(/^\d+_/, "").replace(/_/g, " ");
  }

  function drawChart(canvas, infoEl, points) {
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Shibuya stamps "auto" | "light" | "dark" on the root element, and
    // resolves "auto" against the OS preference.
    const mode = document.documentElement.getAttribute("data-color-mode") || "auto";
    const prefersDark =
      window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = mode === "dark" || (mode === "auto" && prefersDark);
    const fg = isDark ? "#e6e6e6" : "#1a1a1a";
    const grid = isDark ? "#3a3a3a" : "#dddddd";
    const accent = isDark ? "#6ab0f3" : "#2166ac";

    const plotLeft = PADDING.left;
    const plotRight = width - PADDING.right;
    const plotTop = PADDING.top;
    const plotBottom = height - PADDING.bottom;

    // Per-point cost rather than wall time: it stays comparable when a
    // case is resized, which raw wall time does not.
    const cost = (p) => (p.wall_time_s / p.n_points) * 1000;
    const yMax = Math.max(...points.map(cost)) * 1.15 || 1;
    const yMin = 0;

    function xForIndex(i) {
      if (points.length === 1) return (plotLeft + plotRight) / 2;
      return plotLeft + (i / (points.length - 1)) * (plotRight - plotLeft);
    }
    function yForCost(t) {
      return plotBottom - ((t - yMin) / (yMax - yMin)) * (plotBottom - plotTop);
    }

    // Axes.
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotLeft, plotTop);
    ctx.lineTo(plotLeft, plotBottom);
    ctx.lineTo(plotRight, plotBottom);
    ctx.stroke();

    // Y gridlines and labels.
    ctx.fillStyle = fg;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
      const t = (yMax / yTicks) * i;
      const y = yForCost(t);
      ctx.strokeStyle = grid;
      ctx.beginPath();
      ctx.moveTo(plotLeft, y);
      ctx.lineTo(plotRight, y);
      ctx.stroke();
      ctx.fillText(t.toFixed(3) + " ms", plotLeft - 8, y);
    }

    // Line connecting points.
    ctx.strokeStyle = accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = xForIndex(i);
      const y = yForCost(cost(p));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Points + x labels.
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    points.forEach((p, i) => {
      const x = xForIndex(i);
      const y = yForCost(cost(p));
      ctx.fillStyle = accent;
      ctx.beginPath();
      ctx.arc(x, y, POINT_RADIUS, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = fg;
      ctx.fillText(p.version, x, plotBottom + 8);
    });

    // Hover: nearest point by x distance, shown in the info line below.
    canvas.onmousemove = function (evt) {
      const rect = canvas.getBoundingClientRect();
      const mx = ((evt.clientX - rect.left) / rect.width) * width;
      let nearest = 0;
      let best = Infinity;
      points.forEach((p, i) => {
        const d = Math.abs(xForIndex(i) - mx);
        if (d < best) {
          best = d;
          nearest = i;
        }
      });
      describe(points[nearest]);
    };
    if (points.length > 0) {
      describe(points[points.length - 1]);
    }

    function describe(p) {
      infoEl.textContent =
        `version ${p.version} · ${p.date} · host ${p.host} · ` +
        `${p.n_points} points · ${p.n_calls} calls · ` +
        `${p.wall_time_s.toFixed(4)}s total · ${cost(p).toFixed(4)} ms/point`;
    }
  }

  function init(dataset) {
    const select = document.getElementById(SELECT_ID);
    const canvas = document.getElementById(CANVAS_ID);
    const empty = document.getElementById(EMPTY_ID);
    const info = document.getElementById(INFO_ID);

    if (!dataset || dataset.length === 0) {
      canvas.style.display = "none";
      select.style.display = "none";
      empty.style.display = "block";
      return;
    }

    dataset.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.case;
      option.textContent = humanize(entry.case);
      select.appendChild(option);
    });

    function render() {
      const entry = dataset.find((e) => e.case === select.value) || dataset[0];
      drawChart(canvas, info, entry.points);
    }

    select.addEventListener("change", render);
    render();
  }

  document.addEventListener("DOMContentLoaded", function () {
    fetch(DATA_URL)
      .then((r) => r.json())
      .then(init)
      .catch(function () {
        const empty = document.getElementById(EMPTY_ID);
        if (empty) {
          empty.textContent = "Could not load benchmark data.";
          empty.style.display = "block";
        }
      });
  });
})();
