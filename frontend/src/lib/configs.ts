import { FullParams, ParamSet } from "./types";

export interface SavedConfig {
  name: string;
  timestamp: number;
  params: ParamSet;
  /** Optional fitted 13-param vector. Present after a successful fit. */
  paramsFull?: FullParams;
  notes?: string;
}

const KEY = "ratcliff_configs_v1";

export function loadAll(): SavedConfig[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed as SavedConfig[];
  } catch {
    return [];
  }
}

export function save(cfg: SavedConfig): void {
  const all = loadAll().filter((c) => c.name !== cfg.name);
  all.push(cfg);
  localStorage.setItem(KEY, JSON.stringify(all));
}

export function remove(name: string): void {
  const all = loadAll().filter((c) => c.name !== name);
  localStorage.setItem(KEY, JSON.stringify(all));
}

export function exportAllAsJson(): string {
  return JSON.stringify(loadAll(), null, 2);
}

export function importFromJson(json: string): SavedConfig[] {
  const arr = JSON.parse(json);
  if (!Array.isArray(arr)) throw new Error("import must be a JSON array of configs");
  localStorage.setItem(KEY, JSON.stringify(arr));
  return arr as SavedConfig[];
}
