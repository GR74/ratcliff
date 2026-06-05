import { useEffect, useState } from "react";

import { ConfigSidebar } from "../components/ConfigSidebar";
import { ParamSliders } from "../components/ParamSliders";
import { RTHistogram } from "../components/RTHistogram";
import { postSimulate } from "../lib/api";
import { ParamSet, SimulateResponse } from "../lib/types";
import { useAppStore } from "../store";

export function ForwardSimTab() {
  const params = useAppStore((s) => s.params);
  const [result, setResult] = useState<SimulateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0); // bumped by debounce timer to trigger refetch

  // Debounced effect: ~200ms after the last param change, refetch.
  useEffect(() => {
    const id = window.setTimeout(() => setTick((t) => t + 1), 200);
    return () => window.clearTimeout(id);
  }, [params]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    postSimulate(params)
      .then((r) => {
        if (!cancelled) setResult(r);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // tick changes ⇒ refetch with the most recent params
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  const runFull = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await postSimulate(params, { full: true, nsim: 9000 });
      setResult(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  // The "onChange" optional callback is for components that want immediate notification;
  // we already react via useEffect on the `params` zustand slice above.
  const _noop = (_p: ParamSet) => {};

  return (
    <div className="grid grid-cols-12 gap-4 p-4">
      <div className="col-span-3 space-y-6">
        <div className="bg-white border rounded p-4">
          <ParamSliders onChange={_noop} />
        </div>
        <div className="bg-white border rounded p-4">
          <ConfigSidebar />
        </div>
      </div>
      <div className="col-span-9 space-y-4">
        <div className="bg-white border rounded p-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-semibold text-slate-700">Forward simulation</h2>
            <div className="flex gap-2 items-center">
              {loading && <span className="text-xs text-accent">Simulating...</span>}
              <button
                onClick={runFull}
                disabled={loading}
                className="px-3 py-1 text-sm rounded bg-accent text-white disabled:opacity-50"
              >
                Run full (nsim=9000)
              </button>
            </div>
          </div>
          {error && <p className="text-xs text-red-600 mb-2">{error}</p>}
          {result ? (
            <RTHistogram rt={result.rt} cat={result.cat} />
          ) : (
            <div className="h-96 flex items-center justify-center text-slate-400 text-sm italic">
              Waiting for first simulation...
            </div>
          )}
        </div>
        <p className="text-xs text-slate-500">
          Tip: drag any slider — the histogram refreshes ~200ms after you stop moving.
          The preview runs at nsim=256 for instant feedback; click "Run full" for
          production-scale numbers.
        </p>
      </div>
    </div>
  );
}
