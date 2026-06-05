import { useEffect, useRef, useState } from "react";

import { DataUpload } from "../components/DataUpload";
import { FitProgress } from "../components/FitProgress";
import { RecoveryTable } from "../components/RecoveryTable";
import {
  getFitResult,
  getFitStatus,
  postFitStart,
} from "../lib/api";
import {
  DEFAULT_FULL_PARAMS,
  FitProgressPoint,
  FitResultResponse,
  UploadedData,
} from "../lib/types";
import { useAppStore } from "../store";

export function FitTab() {
  const gpuEndpoint = useAppStore((s) => s.gpuEndpoint);
  const setGpuEndpoint = useAppStore((s) => s.setGpuEndpoint);

  const [data, setData] = useState<UploadedData | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<FitProgressPoint[]>([]);
  const [status, setStatus] = useState<string>("idle");
  const [result, setResult] = useState<FitResultResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [maxiter, setMaxiter] = useState(100);
  const pollTimer = useRef<number | null>(null);

  // Cleanup poller on unmount.
  useEffect(() => {
    return () => {
      if (pollTimer.current) window.clearTimeout(pollTimer.current);
    };
  }, []);

  const start = async () => {
    if (!data) return;
    setProgress([]);
    setResult(null);
    setError(null);
    setStatus("starting");
    try {
      const { job_id } = await postFitStart(
        { prop: data.prop, count: data.count, quant: data.quant },
        DEFAULT_FULL_PARAMS as unknown as number[],
        { maxiter, gpuEndpoint: gpuEndpoint || undefined },
      );
      setJobId(job_id);
      setStatus("running");
      poll(job_id);
    } catch (e) {
      setError((e as Error).message);
      setStatus("error");
    }
  };

  const poll = (id: string) => {
    if (pollTimer.current) window.clearTimeout(pollTimer.current);
    pollTimer.current = window.setTimeout(async () => {
      try {
        const s = await getFitStatus(id);
        setProgress(s.progress);
        setStatus(s.status);
        if (s.status === "done") {
          const r = await getFitResult(id);
          setResult(r);
          return;
        }
        if (s.status === "error") {
          setError(s.error || "fit failed");
          return;
        }
        poll(id);
      } catch (e) {
        setError((e as Error).message);
        setStatus("error");
      }
    }, 2000);
  };

  return (
    <div className="grid grid-cols-12 gap-4 p-4">
      <div className="col-span-4 space-y-4">
        <div className="bg-white border rounded p-4 space-y-3">
          <h3 className="font-semibold text-slate-700">1. Upload data</h3>
          <DataUpload onUploaded={setData} />
          {data && (
            <p className="text-xs text-slate-600">
              Loaded: {data.n_subjects} subject{data.n_subjects !== 1 ? "s" : ""},{" "}
              {data.count.reduce((a, c) => a + c.reduce((b, n) => b + n, 0), 0)} total
              trials.
            </p>
          )}
        </div>

        <div className="bg-white border rounded p-4 space-y-3">
          <h3 className="font-semibold text-slate-700">2. Fit settings</h3>
          <div className="space-y-1">
            <label htmlFor="maxiter" className="text-sm">
              maxiter (NM iterations)
            </label>
            <input
              id="maxiter"
              type="number"
              min={10}
              max={1000}
              value={maxiter}
              onChange={(e) => setMaxiter(Number(e.target.value))}
              className="w-full px-2 py-1 border rounded text-sm"
            />
          </div>
          <div className="space-y-1">
            <label htmlFor="gpu" className="text-sm">
              BYO-GPU endpoint <span className="text-xs text-slate-500">(optional)</span>
            </label>
            <input
              id="gpu"
              type="text"
              placeholder="https://your-gpu-box:8000"
              value={gpuEndpoint}
              onChange={(e) => setGpuEndpoint(e.target.value)}
              className="w-full px-2 py-1 border rounded text-sm font-mono"
            />
            <p className="text-[10px] text-slate-500">
              Leave blank to run on this server's CPU. For a fast fit, run the same
              Docker container on a GPU machine and paste its URL here.
            </p>
          </div>
          <button
            onClick={start}
            disabled={!data || status === "running"}
            className="w-full px-3 py-2 rounded bg-accent text-white disabled:opacity-50"
          >
            {status === "running" ? "Fitting..." : "Start fit"}
          </button>
        </div>
      </div>

      <div className="col-span-8 space-y-4">
        <div className="bg-white border rounded p-4">
          <h3 className="font-semibold text-slate-700 mb-2">Progress</h3>
          {error && <p className="text-xs text-red-600">{error}</p>}
          {jobId ? (
            <FitProgress progress={progress} status={status} />
          ) : (
            <p className="text-sm text-slate-500 italic">
              Upload data and click "Start fit" to begin.
            </p>
          )}
        </div>
        {result?.result && (
          <div className="bg-white border rounded p-4 space-y-2">
            <h3 className="font-semibold text-slate-700">Recovery</h3>
            <p className="text-sm text-slate-600">
              Final loss <span className="font-mono">{result.result.loss.toFixed(2)}</span>{" "}
              after <span className="font-mono">{result.result.n_iters}</span> iterations.
            </p>
            <RecoveryTable fittedParams={result.result.params} />
          </div>
        )}
      </div>
    </div>
  );
}
