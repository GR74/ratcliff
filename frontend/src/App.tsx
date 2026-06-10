import { CompareTab } from "./tabs/CompareTab";
import { FieldTab } from "./tabs/FieldTab";
import { FitTab } from "./tabs/FitTab";
import { ForwardSimTab } from "./tabs/ForwardSimTab";
import { PhaseTab } from "./tabs/PhaseTab";
import { PredictTab } from "./tabs/PredictTab";
import { useAppStore } from "./store";

const TABS = [
  { id: "sim", label: "Forward Sim" },
  { id: "field", label: "Field (3D)" },
  { id: "phase", label: "Phase Diagram" },
  { id: "fit", label: "Fit" },
  { id: "predict", label: "Predict" },
  { id: "compare", label: "Compare" },
] as const;

export function App() {
  const currentTab = useAppStore((s) => s.currentTab);
  const setTab = useAppStore((s) => s.setTab);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="bg-white border-b">
        <div className="flex items-baseline gap-6 px-4 py-3">
          <h1 className="text-lg font-semibold">Ratcliff DDM</h1>
          <span className="text-xs text-slate-500">
            2D spatial diffusion decision model — Stage 7 UI
          </span>
        </div>
        <nav className="px-4 flex gap-2 border-t">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={
                "px-4 py-2 text-sm border-b-2 -mb-px " +
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
        {currentTab === "sim" && <ForwardSimTab />}
        {currentTab === "field" && <FieldTab />}
        {currentTab === "phase" && <PhaseTab />}
        {currentTab === "fit" && <FitTab />}
        {currentTab === "predict" && <PredictTab />}
        {currentTab === "compare" && <CompareTab />}
      </main>
      <footer className="border-t bg-white py-2 px-4 text-xs text-slate-500">
        Powered by JAX + K-L low-rank GRF. Source:{" "}
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
