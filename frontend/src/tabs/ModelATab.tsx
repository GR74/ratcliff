import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Data } from "plotly.js";
import { evidenceRadius } from "../lib/scdmGeometry";
import { Params, useModelAState } from "../lib/modelAState";
import { SCDMOriginView } from "../components/SCDMOriginView";


const fields: { key: keyof Params; label: string; min: number; max: number; step: number; hint?: string }[] = [
  { key: "ter", label: "Mean nondecision time (ms)", min: 0, max: 2000, step: 1 },
  { key: "st", label: "Nondecision full range (ms)", min: 0, max: 2000, step: 1, hint: "Uniform across trials: mean ± range/2." },
  { key: "cr", label: "Mean threshold (evidence units)", min: 1, max: 10000, step: 1 },
  { key: "crsd", label: "Threshold full range", min: 0, max: 1000, step: 1, hint: "One draw per trial, shared by all angles." },
  { key: "av", label: "Drift scale av", min: 0, max: 200, step: 0.5, hint: "Peak = av / (si × √(2π)); centered at 180°." },
  { key: "si", label: "Drift width si (5° bins)", min: 0.2, max: 30, step: 0.1 },
  { key: "sig", label: "Noise kernel width sig (5° bins)", min: 0.2, max: 30, step: 0.1 },
  { key: "seed", label: "Random seed", min: 0, max: 4294967295, step: 1 },
];

export function ModelATab({ origin = false }: { origin?: boolean }) {
  const { params, setParams, result, loading, error, selected, setSelected, generate: generateTrial, revision } = useModelAState();
  const [showThreshold, setShowThreshold] = useState(true);
  const traces = useMemo<Data[]>(() => {
    if (!result) return [];
    const mapping = evidenceRadius(result.evidence, result.threshold);
    const start = result.times[0], end = result.times[result.times.length - 1];
    const all: Data[] = result.angles.map((angle, i) => ({
      type: "scatter3d", mode: "lines", name: `${angle}°`, showlegend: false,
      x: result.evidence.map((row) => mapping.radius(row[i]) * Math.cos(angle * Math.PI / 180)),
      y: result.times,
      z: result.evidence.map((row) => mapping.radius(row[i]) * Math.sin(angle * Math.PI / 180)),
      customdata: result.evidence.map((row) => row[i]),
      line: { color: i === result.winner ? "#fbbf24" : `hsl(${210 + i * 1.5},65%,65%)`, width: i === result.winner ? 6 : 2 },
      hovertemplate: `Angle ${angle}°<br>Time %{y:.1f} ms<br>Signed evidence %{customdata:.3f}<extra></extra>`,
    }));
    const theta = Array.from({ length: 97 }, (_, i) => i * Math.PI / 48);
    function ring(value: number, time: number, color: string) {
      const r = mapping.radius(value);
      all.push({ type: "scatter3d", mode: "lines", showlegend: false,
        x: theta.map((t) => r * Math.cos(t)), y: theta.map(() => time), z: theta.map((t) => r * Math.sin(t)),
        line: { color, width: 2 }, hovertemplate: `Evidence ${value.toFixed(3)}<br>Time ${time.toFixed(1)} ms<extra></extra>` });
    }
    function cylinder(value: number, color: string, opacity: number) {
      const r = mapping.radius(value);
      const xs = theta.map((t) => r * Math.cos(t)), zs = theta.map((t) => r * Math.sin(t));
      all.push({ type: "surface", showscale: false, opacity,
        x: [xs, xs], y: [theta.map(() => start), theta.map(() => end)], z: [zs, zs],
        colorscale: [[0, color], [1, color]], hoverinfo: "skip" });
      ring(value, start, color); ring(value, end, color);
    }
    cylinder(0, "#94a3b8", 0.04);
    ring(0, (start + end) / 2, "#64748b");
    if (showThreshold) cylinder(result.threshold, "#fb7185", 0.07);
    ring(mapping.min, start, "#38bdf8");
    const levels = [mapping.min, 0, ...(showThreshold ? [result.threshold] : [])];
    all.push({ type: "scatter3d", mode: "text", showlegend: false,
      x: levels.map((v, i) => mapping.radius(v) * Math.cos((225 - i * 45) * Math.PI / 180)),
      y: levels.map(() => start),
      z: levels.map((v, i) => mapping.radius(v) * Math.sin((225 - i * 45) * Math.PI / 180)),
      text: levels.map((v) => v === 0 ? "0 · zero evidence" : `${v.toFixed(1)}${v === result.threshold ? " · threshold" : " · evidence"}`),
      textposition: "top left", textfont: { size: 11, color: "#e2e8f0" }, hoverinfo: "skip" });
    all.push({ type: "scatter3d", mode: "text", showlegend: false,
      x: [0, 90, 180, 270].map((a) => 2.25 * Math.cos(a * Math.PI / 180)),
      y: [start, start, start, start], z: [0, 90, 180, 270].map((a) => 2.25 * Math.sin(a * Math.PI / 180)),
      text: ["0° / 360°", "90°", "180°", "270°"], textfont: { color: "#cbd5e1", size: 12 }, hoverinfo: "skip" });
    if (result.hit) all.push({ type: "scatter3d", mode: "markers", name: "First crossing", showlegend: false,
      x: [mapping.radius(result.evidence[result.evidence.length - 1][result.winner!]) * Math.cos(result.winning_angle! * Math.PI / 180)],
      y: [result.response_time!],
      z: [mapping.radius(result.evidence[result.evidence.length - 1][result.winner!]) * Math.sin(result.winning_angle! * Math.PI / 180)],
      marker: { color: "#fbbf24", size: 6 }, hovertemplate: `Winner ${result.winning_angle}°<br>Time %{y:.1f} ms<extra></extra>` });
    return all;
  }, [result, showThreshold]);
  async function generate(event: React.FormEvent) {
    event.preventDefault(); await generateTrial();
  }
  return <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 p-4">
    <form onSubmit={generate} className="bg-white border rounded p-4 space-y-3 self-start">
      <h2 className="font-semibold">{origin ? "SCDM Origin · Model A" : "Ratcliff SCDM · Model A"}</h2>
      <p className="text-xs text-slate-500">Actual JAX simulation · one trial · 72 accumulators</p>
      {fields.map((f) => <label key={f.key} className="block text-xs text-slate-600">{f.label}
        <input required type="number" min={f.min} max={f.max} step={f.step} value={Number.isNaN(params[f.key]) ? "" : params[f.key]}
          onChange={(e) => setParams({ ...params, [f.key]: e.target.valueAsNumber })} className="block w-full mt-1 border rounded px-2 py-1 text-sm text-slate-900" />
        {f.hint && <span className="block mt-1 text-[11px] text-slate-500">{f.hint}</span>}
      </label>)}
      <button disabled={loading} className="w-full bg-accent text-white rounded py-2 disabled:opacity-50">{loading ? "Simulating…" : "Generate evidence histories"}</button>
      {error && <p role="alert" className="text-xs text-red-600 break-words">{error}</p>}
      {result && JSON.stringify(params) !== JSON.stringify(result.params) && <p className="text-xs text-amber-700">Parameters changed. Generate to apply; plots show the previous run.</p>}
    </form>
    <section className="lg:col-span-3 bg-white border rounded p-4 space-y-3 min-w-0">
      <h2 className="font-semibold">72 signed evidence histories</h2>
      <p className="text-sm text-slate-600">{origin ? "Experimental display: all 72 histories start at the same evidence origin. Y is total elapsed time from the shared origin; signed evidence is projected along each candidate’s fixed X–Z axis." : "72 fixed angles around the horizontal X–Z circle; Y is elapsed time. Evidence moves each trace radially: negative inward, positive outward from the gray zero-evidence cylinder. Gold marks the winner."}</p>
      <p className="text-xs text-slate-500">Both SCDM tabs share parameters, selected angle, and the exact stored trial. Switching tabs never resimulates. Generate in either tab updates both.{result && ` Shared trial #${revision} · seed ${result.params.seed}.`}</p>
      {result ? <>
        <p className="text-sm" aria-live="polite">{result.hit ? `Winner ${result.winning_angle}° · decision ${result.decision_time} ms · response ${result.response_time!.toFixed(1)} ms`
          : `No crossing within ${result.max_decision_time} ms of decision time. No winning angle or response time.`}</p>
        <p className="text-xs text-slate-600">Sampled threshold: {result.threshold.toFixed(3)} · sampled nondecision time: {result.nondecision.toFixed(1)} ms · drift peak before demeaning: {result.drift_peak.toFixed(3)} / step</p>
        <label className="text-sm flex gap-2 items-center"><input type="checkbox" checked={showThreshold} onChange={(e) => setShowThreshold(e.target.checked)} />{origin ? "Show decision boundary" : "Show shared trial threshold cylinder"}</label>
        {origin ? <SCDMOriginView result={result} selected={selected} showThreshold={showThreshold} /> : <Plot data={traces} layout={{ height: 570, margin: { l: 0, r: 0, t: 10, b: 0 }, paper_bgcolor: "#0f172a", font: { color: "#cbd5e1" },
          uirevision: "model-a-cylindrical-v2", scene: { bgcolor: "#0f172a",
            xaxis: { visible: false, range: [-2.6, 2.6] },
            yaxis: { title: { text: "Elapsed time (ms) ↑" }, gridcolor: "#334155" },
            zaxis: { visible: false, range: [-2.6, 2.6] },
            aspectmode: "manual", aspectratio: { x: 1, y: 1.5, z: 1 }, camera: { up: { x: 0, y: 1, z: 0 }, eye: { x: 1.5, y: 0.9, z: 1.5 } } } }}
          config={{ displaylogo: false, responsive: true, modeBarButtonsToRemove: ["resetCameraDefault3d"] }} style={{ width: "100%" }} useResizeHandler />}
        <p className="text-xs text-slate-500">Drag to orbit in 3D; scroll to zoom. Hover to read each trace. Angles are 5°, 10°, …, 360° (360° denotes 0°). {origin ? "The blue center segment displays the nondecision interval before accumulation; all time coordinates include that delay." : "Nondecision time offsets the time axis; no evidence accumulates during that delay."}</p>
        <p className="text-xs text-slate-500">{origin ? "Signed radius = scale × evidence (no offset). Negative evidence projects through the center to the opposite side: dashed lines still belong to the ORIGINAL candidate angle shown on hover, not an opposite response. The gray axis is zero evidence. Only the actual positive crossing is marked gold; negative magnitude reaching the threshold radius is not a decision. Traces begin at elapsed time 0 at the axes intersection. A zero-evidence center segment displays the nondecision interval before accumulation; this ordering is a display convention. The endpoint, hover and lower chart use total elapsed time." : "Display radius = r₀ + scale × signed evidence. The offset is calculated from all samples to keep every radius positive; gray means zero, not the center. Ring labels and hover show actual signed evidence. Radius is rescaled per trial, so compare evidence values rather than apparent cylinder size. Histories stop at the first crossing (or the time limit)."}</p>
        <label className="text-sm">Inspect one angle: <select className="border rounded p-1" value={selected} onChange={(e) => setSelected(Number(e.target.value))}>
          {result.angles.map((a, i) => <option key={a} value={i}>{a}°{i === result.winner ? " · winner" : ""}</option>)}
        </select></label>
        <Plot data={[{ type: "scatter", mode: "lines", x: result.times, y: result.evidence.map((row) => row[selected]), line: { color: "#2563eb" }, name: `${result.angles[selected]}°` }]}
          layout={{ height: 240, margin: { l: 60, r: 20, t: 15, b: 45 }, xaxis: { title: { text: "Elapsed time (ms)" } }, yaxis: { title: { text: "Signed evidence" }, zeroline: true },
            shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: result.threshold, y1: result.threshold, line: { color: "#e11d48", dash: "dash" } }] }}
          style={{ width: "100%" }} config={{ displaylogo: false, responsive: true }} useResizeHandler />
      </> : <div className="h-[420px] rounded bg-slate-900 text-slate-300 flex items-center justify-center">Generate a trial to inspect all 72 evidence traces.</div>}
      <p className="text-xs text-slate-500">Model A uses 10 ms steps: add Gaussian drift plus 5 × correlated Gaussian noise, subtract the spatial mean, then accumulate. The threshold is shared within a trial. Drift and kernel widths use 5° bins (si = 4 means 20°). The kernel uses index distance, without circular wraparound. Drift variability remains inactive, matching the existing JAX port’s documented Fortran ambiguity. This view exposes that implementation; it is not Smith’s circular diffusion and adds no new empirical validation.</p>
    </section>
  </div>;
}
