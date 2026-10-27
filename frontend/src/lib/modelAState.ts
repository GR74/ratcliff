import { create } from "zustand";

export const defaults = { ter: 200, st: 50, cr: 50, crsd: 10, av: 20, si: 4, sig: 5, seed: 42 };
export type Params = typeof defaults;
export interface History {
  params: Params; angles: number[]; times: number[]; evidence: number[][];
  threshold: number; nondecision: number; hit: boolean; winner: number | null;
  winning_angle: number | null; decision_time: number | null; response_time: number | null;
  dt: number; max_decision_time: number; drift_peak: number;
}
interface State {
  params: Params; result: History | null; selected: number; loading: boolean; error: string; revision: number;
  setParams: (p: Params) => void; setSelected: (i: number) => void; generate: () => Promise<void>;
}
export const useModelAState = create<State>((set, get) => ({
  params: defaults, result: null, selected: 35, loading: false, error: "", revision: 0,
  setParams: (params) => set({ params }), setSelected: (selected) => set({ selected }),
  generate: async () => {
    if (get().loading) return;
    const params = { ...get().params };
    set({ loading: true, error: "" });
    try {
      const base = (import.meta.env.VITE_API_BASE as string | undefined) || "/api";
      const r = await fetch(`${base}/model-a/history`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params) });
      const text = await r.text();
      if (!r.ok) { let message = text; try { const body = JSON.parse(text); message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch {} throw new Error(message); }
      const result: History = JSON.parse(text);
      set((s) => ({ result, selected: result.winner ?? 35, revision: s.revision + 1 }));
    } catch (e) { set({ error: (e as Error).message }); }
    finally { set({ loading: false }); }
  },
}));
