#!/usr/bin/env python3
"""Render a captured MoE schedule timeline as standalone HTML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CPU Expert Execution</title>
<style>
:root {
  color-scheme: light;
  --ink: #18202a;
  --muted: #667085;
  --line: #d8dee8;
  --row: #edf0f4;
  --gather: #269c85;
  --w13: #3975b9;
  --w2: #e3a008;
  --merge: #cf66a3;
  --cleanup: #8f3d72;
  --overhead: #8793a5;
}
* { box-sizing: border-box; letter-spacing: 0; }
html, body {
  margin: 0;
  background: #fff;
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { padding: 18px 20px 28px; }
h1 { margin: 0 0 5px; font-size: 24px; line-height: 1.2; font-weight: 650; }
.subtitle, .layout { margin: 0; color: #354052; font-size: 13px; line-height: 1.5; }
.metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  margin-top: 12px;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
.metric { padding: 9px 22px 9px 0; margin-right: 22px; }
.metric + .metric { border-left: 1px solid var(--line); padding-left: 22px; }
.metric-label { color: var(--muted); font-size: 11px; }
.metric-value { margin-top: 2px; font: 600 14px ui-monospace, SFMono-Regular, Menlo, monospace; }
.toolbar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 18px;
  margin: 13px 0 9px;
}
.segmented { display: inline-flex; border: 1px solid #aeb7c4; border-radius: 6px; overflow: hidden; }
.segmented button {
  min-width: 92px;
  height: 32px;
  padding: 0 12px;
  border: 0;
  border-right: 1px solid #aeb7c4;
  background: #fff;
  color: #354052;
  font: 600 12px inherit;
  cursor: pointer;
}
.segmented button:last-child { border-right: 0; }
.segmented button[aria-pressed="true"] { background: #25364a; color: #fff; }
.segmented button:disabled { background: #f2f4f7; color: #98a2b3; cursor: not-allowed; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; color: #354052; font-size: 12px; }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 14px; height: 14px; border: 1px solid rgba(0,0,0,.08); }
.chart { overflow: auto; border-top: 1px solid var(--line); }
svg { display: block; width: 1900px; max-width: none; height: auto; background: #fff; }
.axis { font-size: 11px; fill: var(--muted); }
.core { font: 10px ui-monospace, SFMono-Regular, Menlo, monospace; fill: var(--muted); }
.grid { stroke: var(--line); stroke-width: 1; }
.row { stroke: var(--row); stroke-width: 1; }
.group { stroke: #8793a5; stroke-width: 1.1; }
.tail-guide { stroke: #25364a; stroke-width: 1.2; fill: none; stroke-dasharray: 4 3; }
.tail-label { font: 600 10px ui-monospace, SFMono-Regular, Menlo, monospace; fill: #25364a; }
.wave-line { stroke: #25364a; stroke-width: 1.1; stroke-dasharray: 5 4; }
.wave-label { font-size: 10px; fill: #25364a; }
</style>
</head>
<body>
<main>
  <h1>CPU expert execution</h1>
  <p id="subtitle" class="subtitle"></p>
  <p id="layout" class="layout"></p>
  <div id="metrics" class="metrics"></div>
  <div class="toolbar">
    <div class="segmented" aria-label="Timeline view">
      <button id="actual-button" type="button" aria-pressed="true">Actual</button>
      <button id="predicted-button" type="button" aria-pressed="false">Predicted</button>
    </div>
    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background:var(--gather)"></span>Gather / Pack A</span>
      <span class="legend-item"><span class="swatch" style="background:var(--w13)"></span>W13 + SiLU</span>
      <span class="legend-item"><span class="swatch" style="background:var(--w2)"></span>W2</span>
      <span class="legend-item"><span class="swatch" style="background:var(--merge)"></span>Early merge</span>
      <span class="legend-item"><span class="swatch" style="background:var(--cleanup)"></span>Final merge</span>
    </div>
  </div>
  <div class="chart"><svg id="timeline" role="img" aria-label="Per-core CPU expert execution timeline"></svg></div>
</main>
<script id="timeline-data" type="application/json">__TIMELINE_DATA__</script>
<script>
const data = JSON.parse(document.getElementById("timeline-data").textContent);
const NS = "http://www.w3.org/2000/svg";
const tasks = new Map(data.plan.tasks.map(task => [Number(task.task), task]));
const cores = data.case.cpu_ids.length;
const tailTasks = data.plan.tasks.filter(task => Number(task.range_granularity) > 0);
const poolTasks = data.plan.tasks.filter(task => Number(task.placement_mode) === 1);
const predictedAvailable = data.predicted.per_core_available !== false;

function el(name, attrs = {}, text = null) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text !== null) node.textContent = text;
  return node;
}

function metric(label, value) {
  const item = document.createElement("div");
  item.className = "metric";
  const key = document.createElement("div");
  key.className = "metric-label";
  key.textContent = label;
  const number = document.createElement("div");
  number.className = "metric-value";
  number.textContent = value;
  item.append(key, number);
  return item;
}

function stageKind(stage) {
  if (stage.includes("gather") || stage === "task_overhead") return stage === "task_overhead" ? "overhead" : "gather";
  if (stage.includes("w13")) return "w13";
  if (stage.includes("w2")) return "w2";
  if (stage === "merge_ready_token") return "merge";
  if (stage.includes("merge")) return "cleanup";
  return "overhead";
}

function stageLabel(stage) {
  const kind = stageKind(stage);
  return {
    gather: "Gather / Pack A",
    w13: "W13 + SiLU",
    w2: "W2",
    merge: "Early token merge",
    cleanup: "Final merge",
    overhead: "Task overhead",
  }[kind];
}

function stageColor(stage) {
  return `var(--${stageKind(stage)})`;
}

function formatTask(task) {
  const end = Number(task.route_end);
  const coreBegin = Number(task.core_begin);
  const placement = coreBegin < 0
    ? `pool ${task.threads}T`
    : `C${String(coreBegin).padStart(2, "0")}-${String(coreBegin + task.threads - 1).padStart(2, "0")}`;
  return `T${task.task} E${task.expert} M[${task.route_begin},${end}) ${placement}`;
}

function formatShape(shape) {
  const widths = shape.map(Number);
  if (widths.length > 0 && widths.every(width => width === widths[0])) {
    return `${widths.length}×${widths[0]}T`;
  }
  return widths.map(width => `${width}T`).join("+");
}

function formatMetric(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(3)} ms` : "n/a";
}

function populateHeader() {
  const actual = data.actual;
  const earlyMerge = data.plan.early_merge === null || data.plan.early_merge === undefined
    ? "auto"
    : String(Boolean(data.plan.early_merge));
  document.getElementById("subtitle").textContent =
    `${data.case.machine} · NUMA${data.case.numa_node} cores ${data.case.cpu_ids[0]}-${data.case.cpu_ids.at(-1)} · ` +
    `${data.case.preset} · ${data.plan.execution_mode} · ${formatShape(data.plan.shape)}`;
  const layout = [];
  layout.push(`${data.plan.tasks.length} tasks`);
  layout.push(`early merge: ${earlyMerge}`);
  if (tailTasks.length > 0) {
    layout.push(`M-split tail: ${tailTasks.map(formatTask).join(" · ")}`);
  }
  if (poolTasks.length > 0) {
    layout.push(
      `dynamic pool: ${poolTasks.length} experts × ${data.plan.tail_pool_threads}T, ` +
      `M≤${data.plan.tail_pool_max_routes}`,
    );
  }
  document.getElementById("layout").textContent = layout.join(" · ");
  const metrics = document.getElementById("metrics");
  metrics.append(
    metric("Untraced median", formatMetric(actual.untraced_median_ms)),
    metric("Traced E2E", formatMetric(actual.traced_e2e_ms)),
    metric("Scheduled compute", formatMetric(actual.scheduled_compute_ms)),
    metric("Trace overhead", `${actual.trace_overhead_pct.toFixed(2)}%`),
    metric("Planner model", formatMetric(data.predicted.planner_makespan_ms)),
    metric("Phase timeline", predictedAvailable ? formatMetric(data.predicted.makespan_ms) : "aggregate only"),
  );
  if (!predictedAvailable) {
    const button = document.getElementById("predicted-button");
    button.disabled = true;
    button.title = data.predicted.scope;
  }
}

function render(mode) {
  const svg = document.getElementById("timeline");
  svg.replaceChildren();
  const view = data[mode];
  const left = 72;
  const right = 1872;
  const top = 42;
  const rowHeight = 15;
  const bottom = top + cores * rowHeight;
  const height = bottom + 28;
  const allSegments = Object.values(view.cores).flat();
  const segmentMax = allSegments.reduce((value, segment) => Math.max(value, Number(segment.end_ms)), 0);
  const maxTime = Math.max(
    segmentMax,
    mode === "actual" ? Number(data.actual.traced_e2e_ms) : Number(data.predicted.makespan_ms),
  );
  const axisMax = Math.ceil(maxTime * 2) / 2;
  const scale = (right - left) / axisMax;
  svg.setAttribute("viewBox", `0 0 1900 ${height}`);

  const tickStep = axisMax <= 12 ? 1 : 2;
  for (let time = 0; time <= axisMax + 1e-9; time += tickStep) {
    const x = left + time * scale;
    svg.append(
      el("line", {x1: x, y1: top - 10, x2: x, y2: bottom, class: "grid"}),
      el("text", {x, y: top - 16, "text-anchor": "middle", class: "axis"}, `${time} ms`),
    );
  }

  for (let core = 0; core < cores; core += 1) {
    const y = top + core * rowHeight;
    if (core % 8 === 0) svg.append(el("line", {x1: 18, y1: y, x2: right, y2: y, class: "group"}));
    svg.append(
      el("text", {x: left - 9, y: y + 11, "text-anchor": "end", class: "core"}, `C${String(core).padStart(2, "0")}`),
      el("line", {x1: left, y1: y + rowHeight, x2: right, y2: y + rowHeight, class: "row"}),
    );
    const segments = view.cores[String(core)] || [];
    for (const segment of segments) {
      const x = left + Number(segment.start_ms) * scale;
      const width = Math.max(0.8, (Number(segment.end_ms) - Number(segment.start_ms)) * scale);
      const rect = el("rect", {
        x, y: y + 1, width, height: rowHeight - 2,
        fill: stageColor(segment.stage),
        opacity: stageKind(segment.stage) === "cleanup" ? 0.55 : 0.9,
      });
      const task = tasks.get(Number(segment.task));
      const taskText = task ? ` · ${formatTask(task)} · ${task.threads}T` : "";
      rect.append(el(
        "title",
        {},
        `C${String(core).padStart(2, "0")} · ${stageLabel(segment.stage)}${taskText} · ` +
        `${Number(segment.start_ms).toFixed(3)}-${Number(segment.end_ms).toFixed(3)} ms`,
      ));
      svg.append(rect);
    }
  }
  svg.append(el("line", {x1: 18, y1: bottom, x2: right, y2: bottom, class: "group"}));

  const tailIds = new Set(tailTasks.map(task => Number(task.task)));
  const tailSegments = allSegments.filter(segment => tailIds.has(Number(segment.task)));
  if (tailSegments.length > 0) {
    const tailStart = Math.min(...tailSegments.map(segment => Number(segment.start_ms)));
    const x = left + tailStart * scale;
    svg.append(
      el("line", {x1: x, y1: top - 9, x2: x, y2: bottom, class: "wave-line"}),
      el("text", {x: x + 5, y: top - 1, class: "wave-label"}, `tail ${tailStart.toFixed(3)} ms`),
    );
  }

  for (const task of tailTasks) {
    if (Number(task.core_begin) < 0) continue;
    const matching = allSegments.filter(segment => Number(segment.task) === Number(task.task));
    if (matching.length === 0) continue;
    const start = Math.min(...matching.map(segment => Number(segment.start_ms)));
    const end = Math.max(...matching.map(segment => Number(segment.end_ms)));
    const x = left + start * scale;
    const y = top + Number(task.core_begin) * rowHeight;
    const width = Math.max(1, (end - start) * scale);
    const taskHeight = Number(task.threads) * rowHeight;
    svg.append(
      el("rect", {x, y, width, height: taskHeight, class: "tail-guide"}),
      el("text", {x: x + 4, y: y + 11, class: "tail-label"}, `T${task.task} E${task.expert} M${task.route_begin}:${task.route_end}`),
    );
  }
}

function setMode(mode) {
  if (mode === "predicted" && !predictedAvailable) return;
  document.getElementById("actual-button").setAttribute("aria-pressed", mode === "actual");
  document.getElementById("predicted-button").setAttribute("aria-pressed", mode === "predicted");
  render(mode);
}

document.getElementById("actual-button").addEventListener("click", () => setMode("actual"));
document.getElementById("predicted-button").addEventListener("click", () => setMode("predicted"));
populateHeader();
setMode("actual");
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("actual") is None:
        raise ValueError("timeline capture has no actual execution data")
    output = args.output or args.input.with_suffix(".html")
    encoded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        HTML_TEMPLATE.replace("__TIMELINE_DATA__", encoded),
        encoding="utf-8",
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
