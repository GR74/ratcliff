import {
  FitResultResponse,
  FitStatusResponse,
  ParamSet,
  PredictResponse,
  SimulateResponse,
  UploadedData,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) || "/api";

async function jsonOrThrow<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = "";
    try {
      const body = await r.json();
      detail = body?.detail || JSON.stringify(body);
    } catch {
      detail = await r.text();
    }
    throw new Error(`${r.status} ${r.statusText}: ${detail}`);
  }
  return (await r.json()) as T;
}

export async function postSimulate(
  params: ParamSet,
  opts: { keySeed?: number; full?: boolean; nsim?: number } = {},
): Promise<SimulateResponse> {
  const r = await fetch(`${BASE}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      params,
      key_seed: opts.keySeed ?? 0,
      full: opts.full ?? false,
      nsim: opts.nsim,
    }),
  });
  return jsonOrThrow<SimulateResponse>(r);
}

export async function postUpload(file: File): Promise<UploadedData> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
  return jsonOrThrow<UploadedData>(r);
}

export async function postFitStart(
  data: { prop: number[][]; count: number[][]; quant: number[][][] },
  x0: number[],
  opts: { maxiter?: number; gpuEndpoint?: string } = {},
): Promise<{ job_id: string }> {
  const r = await fetch(`${BASE}/fit/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      data,
      x0,
      maxiter: opts.maxiter ?? 100,
      gpu_endpoint: opts.gpuEndpoint || undefined,
    }),
  });
  return jsonOrThrow<{ job_id: string }>(r);
}

export async function getFitStatus(jobId: string): Promise<FitStatusResponse> {
  const r = await fetch(`${BASE}/fit/status/${jobId}`);
  return jsonOrThrow<FitStatusResponse>(r);
}

export async function getFitResult(jobId: string): Promise<FitResultResponse> {
  const r = await fetch(`${BASE}/fit/result/${jobId}`);
  return jsonOrThrow<FitResultResponse>(r);
}

export async function postPredict(
  paramsFull: number[],
  opts: { nConditions?: number; nsim?: number; keySeed?: number } = {},
): Promise<PredictResponse> {
  const r = await fetch(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      params_full: paramsFull,
      n_conditions: opts.nConditions ?? 2,
      nsim: opts.nsim ?? 1024,
      key_seed: opts.keySeed ?? 0,
    }),
  });
  return jsonOrThrow<PredictResponse>(r);
}
