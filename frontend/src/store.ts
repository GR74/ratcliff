import { create } from "zustand";

import { DEFAULT_PARAMS, ParamSet } from "./lib/types";

interface AppState {
  currentTab: "sim" | "fit" | "predict" | "compare";
  setTab: (t: AppState["currentTab"]) => void;

  params: ParamSet;
  setParam: <K extends keyof ParamSet>(key: K, value: ParamSet[K]) => void;
  setParams: (p: ParamSet) => void;
  resetParams: () => void;

  gpuEndpoint: string;
  setGpuEndpoint: (s: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  currentTab: "sim",
  setTab: (t) => set({ currentTab: t }),

  params: { ...DEFAULT_PARAMS },
  setParam: (key, value) => set((s) => ({ params: { ...s.params, [key]: value } })),
  setParams: (p) => set({ params: { ...p } }),
  resetParams: () => set({ params: { ...DEFAULT_PARAMS } }),

  gpuEndpoint: "",
  setGpuEndpoint: (s) => set({ gpuEndpoint: s }),
}));
