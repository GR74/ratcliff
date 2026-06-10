export interface ParamSet {
  ter: number;
  st: number;
  cr: number;
  crsd: number;
  sis: number;
  sig: number;
  av1: number;
  av2: number;
  av3: number;
}

export const DEFAULT_PARAMS: ParamSet = {
  ter: 200,
  st: 50,
  cr: 10,
  crsd: 2,
  sis: 12,
  sig: 10,
  av1: 15,
  av2: 10,
  av3: 8,
};

export const PARAM_DEFS: {
  key: keyof ParamSet;
  label: string;
  min: number;
  max: number;
  step: number;
  desc: string;
}[] = [
  { key: "ter", label: "ter", min: 100, max: 400, step: 1, desc: "Non-decision time (ms)" },
  { key: "st", label: "st", min: 0, max: 200, step: 1, desc: "Variability in non-decision time" },
  { key: "cr", label: "cr", min: 1, max: 30, step: 0.1, desc: "Decision threshold" },
  { key: "crsd", label: "crsd", min: 0, max: 10, step: 0.05, desc: "Threshold variability" },
  { key: "sis", label: "sis", min: 1, max: 30, step: 0.5, desc: "Drift bump spatial width" },
  { key: "sig", label: "sig", min: 0.5, max: 17, step: 0.1, desc: "GRF correlation length" },
  { key: "av1", label: "av1", min: 0, max: 30, step: 0.5, desc: "Drift bump 1 amplitude" },
  { key: "av2", label: "av2", min: 0, max: 30, step: 0.5, desc: "Drift bump 2 amplitude" },
  { key: "av3", label: "av3", min: 0, max: 30, step: 0.5, desc: "Drift bump 3 amplitude" },
];

export interface SimulateResponse {
  rt: number[];
  cat: number[];
}

export interface UploadedData {
  prop: number[][];
  count: number[][];
  quant: number[][][];
  n_subjects: number;
}

export interface FitProgressPoint {
  eval_n: number;
  loss: number;
}

export interface FitStatusResponse {
  status: "pending" | "running" | "done" | "error";
  progress: FitProgressPoint[];
  error: string | null;
}

export interface FitResultResponse {
  status: "pending" | "running" | "done" | "error";
  result?: {
    params: number[];
    loss: number;
    n_iters: number;
    converged: boolean;
  };
  error?: string;
}

export interface PredictResponse {
  by_condition: {
    rt: number[];
    cat: number[];
    props: number[];
  }[];
}

export interface FieldResponse {
  frames: number[][][]; // [frame][row][col]
  steps: number[];
  threshold: number;
  n: number;
  m: number;
  nstep: number;
}

/** A 13-parameter vector used for fits + predictions. Indices match Ratcliff:
 *  [ter, st, cr, crsd, sis, sig, sv, av1c1, av2c1, av3c1, av1c2, av2c2, av3c2]
 */
export type FullParams = [
  number, number, number, number, number, number, number,
  number, number, number, number, number, number,
];

export const DEFAULT_FULL_PARAMS: FullParams = [
  200, 50, 10, 2, 12, 10, 0.5,
  15, 10, 8, 14, 11, 9,
];

export const FULL_PARAM_LABELS = [
  "ter", "st", "cr", "crsd", "sis", "sig", "sv",
  "av1c1", "av2c1", "av3c1", "av1c2", "av2c2", "av3c2",
] as const;
