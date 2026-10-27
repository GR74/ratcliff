import { lazy, Suspense } from "react";

import { useAppStore } from "./store";

// Lazy-load each tab so the heavy deps (Plotly ~3 MB, three.js ~1 MB) only
// download when a tab that uses them is first opened. Keeps the initial load
// light — the Forward Sim tab is interactive before Phase/Field code arrives.
const ForwardSimTab = lazy(() =>
  import("./tabs/ForwardSimTab").then((m) => ({ default: m.ForwardSimTab })),
);
const FieldTab = lazy(() =>
  import("./tabs/FieldTab").then((m) => ({ default: m.FieldTab })),
);
const PhaseTab = lazy(() =>
  import("./tabs/PhaseTab").then((m) => ({ default: m.PhaseTab })),
);
const FitTab = lazy(() => import("./tabs/FitTab").then((m) => ({ default: m.FitTab })));
const PredictTab = lazy(() =>
  import("./tabs/PredictTab").then((m) => ({ default: m.PredictTab })),
);
const CompareTab = lazy(() =>
  import("./tabs/CompareTab").then((m) => ({ default: m.CompareTab })),
);

const TABS = [
  { id: "sim", label: "Forward Sim" },
  { id: "field", label: "Field (3D)" },
  { id: "phase", label: "Phase Diagram" },
  { id: "fit", label: "Fit" },
  { id: "predict", label: "Predict" },
  { id: "compare", label: "Compare" },
  { id: "model-a", label: "Ratcliff SCDM" },
  { id: "scdm-origin", label: "SCDM Origin" },
  { id: "circular", label: "Circular Diffusion" },
] as const;

const CircularTab = lazy(() => import("./tabs/CircularTab").then((m) => ({ default: m.CircularTab })));
const ModelATab = lazy(() => import("./tabs/ModelATab").then((m) => ({ default: m.ModelATab })));

export function App() {
  const currentTab = useAppStore((s) => s.currentTab);
  const setTab = useAppStore((s) => s.setTab);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white border-b">
        <div className="flex items-baseline gap-6 px-4 py-3">
          <h1 className="text-lg font-semibold">Ratcliff DDM</h1>
          <span className="text-xs text-slate-500">
            2D spatial diffusion decision model
          </span>
        </div>
        <nav className="px-4 flex gap-2 border-t overflow-x-auto" aria-label="Models and tools">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              aria-current={currentTab === t.id ? "page" : undefined}
              className={
                "px-4 py-2 text-sm border-b-2 -mb-px whitespace-nowrap " +
                (currentTab === t.id
                  ? "border-accent text-accent font-medium"
                  : "border-transparent text-slate-600 hover:text-slate-900")
              }
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="flex-1">
        <Suspense
          fallback={
            <div className="p-8 text-sm text-slate-400 italic">Loading…</div>
          }
        >
          {currentTab === "sim" && <ForwardSimTab />}
          {currentTab === "field" && <FieldTab />}
          {currentTab === "phase" && <PhaseTab />}
          {currentTab === "fit" && <FitTab />}
          {currentTab === "predict" && <PredictTab />}
          {currentTab === "compare" && <CompareTab />}
          {currentTab === "model-a" && <ModelATab />}
          {currentTab === "scdm-origin" && <ModelATab origin />}
          {currentTab === "circular" && <CircularTab />}
        </Suspense>
      </main>
      <footer className="border-t bg-white py-2 px-4 text-xs text-slate-500">
        Ratcliff models: JAX · Circular demo: NumPy. Source:{" "}
        <a
          className="underline"
          href="https://github.com/GR74/ratcliff"
          target="_blank"
          rel="noopener noreferrer"
        >
          GR74/ratcliff
        </a>
      </footer>
    </div>
  );
}
