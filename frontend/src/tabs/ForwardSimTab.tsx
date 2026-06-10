import { useEffect, useRef, useState } from "react";

import { ConfigSidebar } from "../components/ConfigSidebar";
import { ParamSliders } from "../components/ParamSliders";
import { RTHistogram } from "../components/RTHistogram";
import { postSimulate } from "../lib/api";
import { SimulateResponse } from "../lib/types";
import { useAppStore } from "../store";

export function ForwardSimTab() {
  const params = useAppStore((s) => s.params);
  const [result, setResult] = useState<SimulateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<number | null>(null);

  // Debounced preview: 200ms after the last param change, fire a request,
  // cancelling any in-flight one so rapid slider drags don't queue stale calls.
  useEffect(() => {
    if (timerRef.current) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;
      setLoading(true);
      setError(null);
      postSimulate(params, { signal: ac.signal })
        .then((r) => setResult(r))
        .catch((e: Error) => {
          if (e.name !== "AbortError") setError(e.message);
        })
        .finally(() => {
          // Only clear loading if this is still the active request.
          if (abortRef.current === ac) setLoading(false);
        });
    }, 200);
    return () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
    };
  }, [params]);

  const runFull = async () => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setLoading(true);
    setError(null);
    try {
      const r = await postSimulate(params, { full: true, nsim: 9000, signal: ac.signal });
      setResult(r);
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
    } finally {
      if (abortRef.current === ac) setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-12 gap-4 p-4">
      <div className="col-span-3 space-y-6">
        <div className="bg-white border rounded p-4">
          <ParamSliders />
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
              Warming up the simulator (first call compiles, ~1-2 min)...
            </div>
          )}
        </div>
        <p className="text-xs text-slate-500">
          Drag any slider — the histogram refreshes ~200ms after you stop moving.
          The preview runs at nsim=128 for instant feedback; "Run full" does
          production-scale nsim=9000.
        </p>
      </div>
    </div>
  );
}
