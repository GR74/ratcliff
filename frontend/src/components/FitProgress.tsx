import Plot from "react-plotly.js";

import { FitProgressPoint } from "../lib/types";

interface Props {
  progress: FitProgressPoint[];
  status: string;
}

export function FitProgress({ progress, status }: Props) {
  if (progress.length === 0) {
    return (
      <div className="text-sm text-slate-500 italic">
        {status === "pending" ? "Queued..." : "Waiting for first evaluation..."}
      </div>
    );
  }
  return (
    <div className="space-y-2">
      <p className="text-sm">
        <span className="font-medium">Status:</span> {status} —{" "}
        <span className="font-mono">{progress.length}</span> evals,{" "}
        current loss <span className="font-mono">{progress.at(-1)?.loss.toFixed(2)}</span>
      </p>
      <Plot
        data={[
          {
            x: progress.map((p) => p.eval_n),
            y: progress.map((p) => p.loss),
            type: "scatter",
            mode: "lines+markers",
            marker: { size: 4, color: "#2563eb" },
            name: "loss",
          },
        ]}
        layout={{
          height: 300,
          margin: { l: 50, r: 20, t: 20, b: 50 },
          xaxis: { title: { text: "Evaluation #" } },
          yaxis: { title: { text: "G² loss" }, type: "log" },
        }}
        style={{ width: "100%" }}
        config={{ displaylogo: false }}
      />
    </div>
  );
}
