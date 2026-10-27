import { useEffect, useState } from "react";
import { CircularParams, CircularResult, CircularView } from "../components/CircularView";

const defaults: CircularParams = { nondecision: 0.2, threshold: 1, drift: 1, direction: 45, noise: 0.6, seed: 42, max_time: 5 };
const fields: { key: keyof CircularParams; label: string; min: number; max: number; step: number }[] = [
  { key: "nondecision", label: "Nondecision time (s)", min: 0, max: 3, step: 0.05 },
  { key: "threshold", label: "Threshold / radius (evidence units)", min: 0.1, max: 5, step: 0.1 },
  { key: "drift", label: "Drift strength (evidence units/s)", min: 0, max: 10, step: 0.1 },
  { key: "direction", label: "Drift direction (degrees, +X toward +Z)", min: 0, max: 360, step: 1 },
  { key: "noise", label: "Noise σ (evidence units/√s)", min: 0, max: 3, step: 0.05 },
  { key: "seed", label: "Random seed", min: 0, max: 4294967295, step: 1 },
  { key: "max_time", label: "Decision time limit (s)", min: 0.1, max: 20, step: 0.1 },
];

export function CircularTab() {
  const [params, setParams] = useState(defaults);
  const [result, setResult] = useState<CircularResult | null>(null);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!playing || !result) return;
    const start = performance.now(); const initial = index;
    const timer = window.setInterval(() => {
      const next = Math.min(result.path.length - 1, initial + Math.floor((performance.now() - start) / (1000 * result.dt)));
      setIndex(next); if (next === result.path.length - 1) setPlaying(false);
    }, 30);
    return () => clearInterval(timer);
  }, [playing, result]);
  const changed = result && JSON.stringify(result.params) !== JSON.stringify(params);
  const generate = async (event: React.FormEvent) => {
    event.preventDefault(); setLoading(true); setError(""); setPlaying(false);
    try {
      const base = (import.meta.env.VITE_API_BASE as string | undefined) || "/api";
      const response = await fetch(`${base}/circular`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params) });
      const text = await response.text();
      if (!response.ok) throw new Error(`Simulation failed (${response.status}): ${text}`);
      setResult(JSON.parse(text)); setIndex(0); setPlaying(true);
    } catch (e) { setError((e as Error).message); }
    finally { setLoading(false); }
  };
  return <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 p-4">
    <form onSubmit={generate} className="bg-white border rounded p-4 space-y-4 self-start">
      <h2 className="font-semibold">Circular Diffusion</h2>
      <p className="text-xs text-slate-500">Smith-style · one trial · starts at the center</p>
      {fields.map((f) => <label key={f.key} className="block text-xs text-slate-600">{f.label}
        <input type="number" required min={f.min} max={f.max} step={f.step} value={Number.isNaN(params[f.key]) ? "" : params[f.key]}
          onChange={(e) => setParams({ ...params, [f.key]: e.target.valueAsNumber })} className="block mt-1 w-full border rounded px-2 py-1 text-sm text-slate-900" />
      </label>)}
      <button disabled={loading} className="w-full rounded bg-accent text-white px-3 py-2 disabled:opacity-50">{loading ? "Generating…" : "Generate trajectory"}</button>
      {changed && <p className="text-xs text-amber-700">Parameters changed. Generate to apply them; the view shows the previous run.</p>}
      {error && <p role="alert" className="text-xs text-red-600 break-words">{error}</p>}
    </form>
    <section className="lg:col-span-3 bg-white border rounded p-4 space-y-3">
      <h2 className="font-semibold">Evidence through time</h2>
      <p className="text-sm text-slate-600">The circle lies in X–Z; Y is elapsed time in seconds. Gray center line: nondecision delay. Gold: accumulated evidence. Blue arrow: drift direction.</p>
      {result ? <>
        <CircularView result={result} index={index} />
        <div className="flex gap-3 items-center text-sm">
          <button className="rounded bg-slate-200 px-3 py-1" onClick={() => { if (index === result.path.length - 1) setIndex(0); setPlaying(!playing); }}>{playing ? "Pause" : index === result.path.length - 1 ? "Replay" : "Play"}</button>
          <input aria-label="Trajectory time" type="range" min={0} max={result.path.length - 1} value={index} onChange={(e) => { setPlaying(false); setIndex(Number(e.target.value)); }} className="flex-1" />
          <span>{result.path[index][1].toFixed(3)} s</span>
        </div>
        <p className="text-sm" aria-live="polite">{result.hit
          ? `Boundary hit · decision ${result.decision_time!.toFixed(3)} s · total response ${result.response_time!.toFixed(3)} s · angle ${result.angle!.toFixed(1)}°`
          : `Timed out at ${result.params.max_time} s of decision time. No boundary hit; no response time or response angle.`}</p>
        <p className="text-xs text-slate-500">Drag to rotate; scroll to zoom. Playback and rotation reuse this stored path. Time is scaled independently for display; the ring labels show seconds. Radius: {result.params.threshold} evidence units.</p>
      </> : <div className="h-[420px] flex items-center justify-center rounded bg-slate-900 text-slate-300">Choose parameters and generate a trajectory.</div>}
      <p className="text-xs text-slate-500">dX = μ dt + σ dW, with independent noise on X and Z. The radius is the absorbing decision threshold. Nondecision time shifts the trajectory upward without changing evidence. Educational Euler simulation (dt = 1 ms), stopped at the first observed exit with a line–circle intersection. Continuous Brownian crossings between samples can be missed. No fitted or validated equivalence to published predictions is claimed.</p>
      <a className="text-xs text-accent underline" href="https://doi.org/10.1007/s42113-023-00174-5" target="_blank" rel="noreferrer">Model reference: Smith, Garrett & Zhou (2023), Eq. 1</a>
    </section>
  </div>;
}
