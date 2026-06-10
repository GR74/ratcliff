import { useState } from "react";
import Plot from "react-plotly.js";

import { postPhase } from "../lib/api";
import { PhaseResponse } from "../lib/types";
import { useAppStore } from "../store";

const SWEEPABLE = ["cr", "av1", "av2", "av3", "sis", "sig", "ter", "st"];

export function PhaseTab() {
  const params = useAppStore((s) => s.params);
  const [xParam, setXParam] = useState("cr");
  const [yParam, setYParam] = useState("av1");
  const [xRange, setXRange] = useState<[number, number]>([4, 18]);
  const [yRange, setYRange] = useState<[number, number]>([4, 24]);
  const [grid, setGrid] = useState(12);
  const [nsim, setNsim] = useState(200);
  const [metric, setMetric] = useState<"accuracy" | "rt">("accuracy");
  const [result, setResult] = useState<PhaseResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await postPhase(params, {
        xParam,
        yParam,
        xRange,
        yRange,
        grid,
        nsim,
        metric,
      });
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const cells = grid * grid;

  return (
    <div className="grid grid-cols-12 gap-4 p-4">
      <div className="col-span-3 bg-white border rounded p-4 space-y-3">
        <h3 className="font-semibold text-slate-700">Phase diagram</h3>

        <div className="space-y-1">
          <label className="text-sm">x-axis param</label>
          <select
            value={xParam}
            onChange={(e) => setXParam(e.target.value)}
            className="w-full px-2 py-1 border rounded text-sm"
          >
            {SWEEPABLE.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
          <div className="flex gap-1">
            <input
              type="number"
              value={xRange[0]}
              onChange={(e) => setXRange([Number(e.target.value), xRange[1]])}
              className="w-full px-1 py-0.5 border rounded text-xs"
            />
            <input
              type="number"
              value={xRange[1]}
              onChange={(e) => setXRange([xRange[0], Number(e.target.value)])}
              className="w-full px-1 py-0.5 border rounded text-xs"
            />
          </div>
        </div>

        <div className="space-y-1">
          <label className="text-sm">y-axis param</label>
          <select
            value={yParam}
            onChange={(e) => setYParam(e.target.value)}
            className="w-full px-2 py-1 border rounded text-sm"
          >
            {SWEEPABLE.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
          <div className="flex gap-1">
            <input
              type="number"
              value={yRange[0]}
              onChange={(e) => setYRange([Number(e.target.value), yRange[1]])}
              className="w-full px-1 py-0.5 border rounded text-xs"
            />
            <input
              type="number"
              value={yRange[1]}
              onChange={(e) => setYRange([yRange[0], Number(e.target.value)])}
              className="w-full px-1 py-0.5 border rounded text-xs"
            />
          </div>
        </div>

        <div className="space-y-1">
          <label className="text-sm">color metric</label>
          <div className="flex gap-2">
            <button
              onClick={() => setMetric("accuracy")}
              className={
                "flex-1 px-2 py-1 rounded text-sm " +
                (metric === "accuracy" ? "bg-accent text-white" : "bg-slate-200")
              }
            >
              Accuracy
            </button>
            <button
              onClick={() => setMetric("rt")}
              className={
                "flex-1 px-2 py-1 rounded text-sm " +
                (metric === "rt" ? "bg-accent text-white" : "bg-slate-200")
              }
            >
              Mean RT
            </button>
          </div>
          <p className="text-[10px] text-slate-500">
            "Accuracy" = proportion of trials landing in the target region (category
            1, the innermost zone). This is a 5-category spatial model, not 2-choice.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div className="space-y-1">
            <label className="text-xs">grid ({cells} cells)</label>
            <input
              type="number"
              min={2}
              max={24}
              value={grid}
              onChange={(e) => setGrid(Number(e.target.value))}
              className="w-full px-2 py-1 border rounded text-sm"
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs">nsim / cell</label>
            <input
              type="number"
              min={32}
              max={2000}
              value={nsim}
              onChange={(e) => setNsim(Number(e.target.value))}
              className="w-full px-2 py-1 border rounded text-sm"
            />
          </div>
        </div>

        <button
          onClick={generate}
          disabled={loading}
          className="w-full px-3 py-2 rounded bg-accent text-white disabled:opacity-50"
        >
          {loading ? "Sweeping..." : "Generate phase diagram"}
        </button>
        <p className="text-[10px] text-amber-600">
          {cells} sims on CPU — expect ~{Math.ceil((cells * nsim) / 6000)} min the first
          time (much faster on a GPU).
        </p>
        {error && <p className="text-xs text-red-600">{error}</p>}
      </div>

      <div className="col-span-9 bg-white border rounded p-4">
        {result ? (
          <Plot
            data={[
              {
                z: result.z,
                x: result.x_values,
                y: result.y_values,
                type: "heatmap",
                colorscale: metric === "accuracy" ? "Viridis" : "Hot",
                colorbar: { title: { text: metric === "accuracy" ? "P(target)" : "RT (ms)" } },
              },
            ]}
            layout={{
              title: {
                text: `${metric === "accuracy" ? "Accuracy" : "Mean RT"}: ${result.x_param} × ${result.y_param}`,
              },
              xaxis: { title: { text: result.x_param } },
              yaxis: { title: { text: result.y_param } },
              height: 560,
              margin: { l: 60, r: 20, t: 50, b: 60 },
            }}
            style={{ width: "100%" }}
            config={{ displaylogo: false }}
          />
        ) : (
          <div className="h-[560px] flex items-center justify-center text-slate-400 text-sm italic">
            Pick two parameters and click "Generate phase diagram" to reveal the
            parameter regimes.
          </div>
        )}
      </div>
    </div>
  );
}
