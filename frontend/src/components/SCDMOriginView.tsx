import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Camera, Data } from "plotly.js";
import { History } from "../lib/modelAState";
import { signedSegments } from "../lib/scdmGeometry";

export function SCDMOriginView({ result, selected, showThreshold }: { result: History; selected: number; showThreshold: boolean }) {
  const [isolate, setIsolate] = useState(false);
  const [topView, setTopView] = useState(false);
  // Fit each trial, retaining actual elapsed-time coordinates and the zero origin.
  const duration = result.times[result.times.length - 1];
  const horizon = Math.max(1, duration * 1.08);
  // Plotly supports projection; the installed declaration omits this property.
  const camera: Partial<Camera> & { projection: { type: "orthographic" } } = {
    up: topView ? { x: 0, y: 0, z: 1 } : { x: 0, y: 1, z: 0 },
    eye: topView ? { x: 0, y: 2.5, z: 0 } : { x: 1.5, y: 0.9, z: 1.5 },
    projection: { type: "orthographic" },
  };
  const traces = useMemo<Data[]>(() => {
    let extent = result.threshold;
    for (const row of result.evidence) for (const e of row) extent = Math.max(extent, Math.abs(e));
    // Absolute value only determines the symmetric plot extent, never a coordinate.
    const scale = 2 / Math.max(extent, 1);
    const times = [0, ...result.times];
    const end = times[times.length - 1];
    const data: Data[] = [];
    result.angles.forEach((angle, i) => {
      if (isolate && i !== selected) return;
      const theta = angle * Math.PI / 180;
      const segments = signedSegments([0, ...result.evidence.map((row) => row[i])], times);
      const color = i === result.winner ? "#fbbf24" : `hsl(${i * 5},70%,65%)`;
      for (const sign of ["positive", "negative"] as const) {
        const points = segments[sign];
        data.push({ type: "scatter3d", mode: "lines", showlegend: false, connectgaps: false,
          name: `${angle}° · ${sign}`, opacity: i === selected || isolate ? 1 : 0.55,
          x: points.map((p) => p ? scale * p[0] * Math.cos(theta) : null),
          y: points.map((p) => p ? p[1] : null),
          z: points.map((p) => p ? scale * p[0] * Math.sin(theta) : null),
          customdata: points.map((p) => p ? [p[0], Math.max(0, p[1] - result.nondecision)] : [null, null]),
          line: { color, width: i === selected ? 6 : 2, dash: sign === "negative" ? "dash" : "solid" },
          hovertemplate: `Original candidate ${angle}° · ${sign} evidence<br>Elapsed time %{y:.1f} ms<br>Accumulation time %{customdata[1]:.1f} ms<br>Evidence %{customdata[0]:.3f}<extra></extra>` });
      }
    });
    data.push({ type: "scatter3d", mode: "lines+markers", x: [0, 0], y: [0, horizon], z: [0, 0],
      line: { color: "#94a3b8", dash: "dot", width: 3 }, marker: { size: 3, color: "#e2e8f0" }, showlegend: false,
      hovertemplate: "Zero evidence · shared origin axis<br>Elapsed time %{y:.1f} ms<extra></extra>" });
    data.push({ type: "scatter3d", mode: "text+lines+markers", showlegend: false,
      x: [0, 0], y: [0, result.nondecision], z: [0, 0],
      text: ["", `Nondecision ${result.nondecision.toFixed(1)} ms`], textposition: "middle right",
      line: { color: "#38bdf8", width: 6 }, marker: { size: 3, color: "#38bdf8" },
      textfont: { color: "#38bdf8", size: 10 },
      hovertemplate: "Nondecision interval (display convention)<br>Elapsed time %{y:.1f} ms<extra></extra>" });
    // Explicit axes intersect at the actual shared evidence origin, not box edges.
    for (const axis of ["X", "Z"] as const) data.push({ type: "scatter3d", mode: "lines", showlegend: false,
      x: axis === "X" ? [-2.3, 0, 2.3] : [0, 0, 0], y: [0, 0, 0],
      z: axis === "Z" ? [-2.3, 0, 2.3] : [0, 0, 0],
      line: { color: "#64748b", width: 2 }, hoverinfo: "skip" });
    const tickBase = 10 ** Math.floor(Math.log10(horizon / 5));
    const tickStep = [1, 2, 5, 10].find((n) => n * tickBase >= horizon / 5)! * tickBase;
    const ticks = Array.from({ length: Math.floor(horizon / tickStep) + 1 }, (_, i) => i * tickStep);
    data.push({ type: "scatter3d", mode: "text+markers", showlegend: false,
      x: ticks.map(() => 0), y: ticks, z: ticks.map(() => 0),
      text: ticks.map((t) => t === 0 ? "0 · shared start" : `${t} ms`), textposition: "middle left",
      marker: { size: 2, color: "#94a3b8" }, textfont: { size: 10, color: "#cbd5e1" }, hoverinfo: "skip" });
    const theta = Array.from({ length: 97 }, (_, i) => i * Math.PI / 48);
    const radius = scale * result.threshold;
    if (showThreshold) {
      const xs = theta.map((t) => radius * Math.cos(t)), zs = theta.map((t) => radius * Math.sin(t));
      if (!topView) {
        data.push({ type: "surface", x: [xs, xs],
          y: [theta.map(() => result.nondecision), theta.map(() => end)], z: [zs, zs],
          showscale: false, opacity: 0.06, colorscale: [[0, "#fb7185"], [1, "#fb7185"]], hoverinfo: "skip" });
        data.push({ type: "scatter3d", mode: "lines", x: xs, y: theta.map(() => result.nondecision), z: zs,
          line: { color: "#94a3b8", width: 2 }, showlegend: false, hoverinfo: "skip" });
      }
      data.push({ type: "scatter3d", mode: "lines", x: xs, y: theta.map(() => end), z: zs,
        line: { color: "#fb7185", width: 2, dash: topView ? "dash" : "solid" }, showlegend: false, hoverinfo: "skip" });
    }
    data.push({ type: "scatter3d", mode: "text", showlegend: false,
      x: [0, 90, 180, 270].map((a) => 2.2 * Math.cos(a * Math.PI / 180)), y: [0, 0, 0, 0],
      z: [0, 90, 180, 270].map((a) => 2.2 * Math.sin(a * Math.PI / 180)),
      text: ["0°", "90°", "180°", "270°"], textfont: { color: "#cbd5e1", size: 11 }, hoverinfo: "skip" });
    if (result.hit && (!isolate || selected === result.winner)) {
      const r = scale * result.evidence[result.evidence.length - 1][result.winner!];
      const angle = result.winning_angle! * Math.PI / 180;
      data.push({ type: "scatter3d", mode: "text+lines", showlegend: false,
        x: [0, r * Math.cos(angle)], y: [result.response_time!, result.response_time!], z: [0, r * Math.sin(angle)],
        line: { color: "#fbbf24", width: 2, dash: "dot" },
        text: [`RT ${result.response_time!.toFixed(1)} ms`, ""], textposition: "top left",
        textfont: { color: "#fbbf24", size: 11 },
        hovertemplate: "Response-time level %{y:.1f} ms<extra></extra>" });
      data.push({ type: "scatter3d", mode: "markers", x: [r * Math.cos(angle)], y: [result.response_time!], z: [r * Math.sin(angle)],
        marker: { color: "#fbbf24", size: 7 }, showlegend: false,
        hovertemplate: `Actual positive winner ${result.winning_angle}°<br>Response / elapsed time %{y:.1f} ms<br>Decision time ${result.decision_time} ms<extra></extra>` });
    }
    return data;
  }, [result, selected, isolate, showThreshold, horizon, topView]);
  return <>
    <div className="text-sm rounded bg-slate-50 border p-3 space-y-2">
      <p>Solid = positive evidence; dashed = negative evidence for the same original candidate. Color preserves candidate identity; the selected {result.angles[selected]}° trace is thicker. Gold dot = actual positive winning crossing.</p>
      <label className="flex gap-2"><input type="checkbox" checked={isolate} onChange={(e) => setIsolate(e.target.checked)} />Show only selected candidate ({result.angles[selected]}°)</label>
      <p className="text-xs">Threshold: +{result.threshold.toFixed(3)} evidence. Opposite-side dashed traces can meet the threshold radius without winning.</p>
    </div>
    <div className="flex items-start justify-between gap-4">
      <p className="text-xs text-slate-600">Y = total elapsed time; the endpoint equals reported response time. The blue center segment shows {result.nondecision.toFixed(1)} ms of nondecision time from the shared origin. Placing this interval before accumulation is a display convention, not a physiological assertion. Height fits each trial; compare labeled times across trials. The {topView ? "dashed" : "solid"} pink boundary is at {duration.toFixed(1)} ms.</p>
      <button type="button" aria-pressed={topView} onClick={() => setTopView((value) => !value)} className="shrink-0 rounded border px-3 py-2 text-sm text-accent">{topView ? "Return to 3D" : "Top view (2D)"}</button>
    </div>
    {topView && <p className="text-xs text-slate-500">Looking straight down the time axis: the entire history is projected onto X–Z, not a time slice.</p>}
    <Plot data={traces} layout={{ height: 570, margin: { l: 0, r: 0, t: 10, b: 0 }, paper_bgcolor: "#0f172a", font: { color: "#cbd5e1" },
      uirevision: `scdm-origin-${topView}-${horizon}`, scene: { bgcolor: "#0f172a", xaxis: { visible: false, range: [-2.6, 2.6], autorange: false },
        yaxis: { visible: false, range: [0, horizon], autorange: false }, zaxis: { visible: false, range: [-2.6, 2.6], autorange: false },
        dragmode: topView ? false : "orbit",
        aspectmode: "manual", aspectratio: { x: 1, y: 1.5, z: 1 }, camera } }}
      config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ["resetCameraDefault3d"] }} style={{ width: "100%" }} useResizeHandler />
  </>;
}
