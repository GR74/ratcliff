import { useState } from "react";

import { RTHistogram } from "../components/RTHistogram";
import { postPredict } from "../lib/api";
import {
  DEFAULT_FULL_PARAMS,
  FULL_PARAM_LABELS,
  PredictResponse,
} from "../lib/types";

export function PredictTab() {
  const [params, setParams] = useState<number[]>([...DEFAULT_FULL_PARAMS]);
  const [nConditions, setNConditions] = useState(2);
  const [nsim, setNsim] = useState(1024);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await postPredict(params, { nConditions, nsim });
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-12 gap-4 p-4">
      <div className="col-span-4 bg-white border rounded p-4 space-y-3">
        <h3 className="font-semibold text-slate-700">Parameter vector (13)</h3>
        <div className="grid grid-cols-2 gap-2">
          {FULL_PARAM_LABELS.map((label, i) => (
            <div key={label} className="text-sm">
              <label className="font-mono text-xs text-slate-600">{label}</label>
              <input
                type="number"
                step="0.1"
                value={params[i]}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  const next = [...params];
                  next[i] = v;
                  setParams(next);
                }}
                className="w-full px-1 py-0.5 border rounded text-xs font-mono"
              />
            </div>
          ))}
        </div>
        <div className="space-y-1 pt-2 border-t">
          <label className="text-sm">n_conditions</label>
          <select
            value={nConditions}
            onChange={(e) => setNConditions(Number(e.target.value))}
            className="w-full px-2 py-1 border rounded text-sm"
          >
            <option value={1}>1</option>
            <option value={2}>2</option>
          </select>
        </div>
        <div className="space-y-1">
          <label className="text-sm">nsim per condition</label>
          <input
            type="number"
            min={64}
            max={9000}
            value={nsim}
            onChange={(e) => setNsim(Number(e.target.value))}
            className="w-full px-2 py-1 border rounded text-sm"
          />
        </div>
        <button
          onClick={run}
          disabled={loading}
          className="w-full px-3 py-2 rounded bg-accent text-white disabled:opacity-50"
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
        {error && <p className="text-xs text-red-600">{error}</p>}
      </div>

      <div className="col-span-8 space-y-4">
        {result ? (
          result.by_condition.map((cond, idx) => (
            <div key={idx} className="bg-white border rounded p-4">
              <h3 className="font-semibold text-slate-700 mb-2">Condition {idx + 1}</h3>
              <p className="text-xs text-slate-600 mb-2 font-mono">
                props = [{cond.props.map((p) => p.toFixed(3)).join(", ")}]
              </p>
              <RTHistogram rt={cond.rt} cat={cond.cat} />
            </div>
          ))
        ) : (
          <div className="bg-white border rounded p-4 text-center text-slate-500 italic">
            Set parameters on the left and click Predict.
          </div>
        )}
      </div>
    </div>
  );
}
