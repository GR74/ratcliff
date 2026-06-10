import { useEffect, useState } from "react";

import { RTHistogram } from "../components/RTHistogram";
import { postSimulate } from "../lib/api";
import * as configs from "../lib/configs";
import { SimulateResponse } from "../lib/types";

export function CompareTab() {
  const [list, setList] = useState<configs.SavedConfig[]>(configs.loadAll());
  const [nameA, setNameA] = useState<string>("");
  const [nameB, setNameB] = useState<string>("");
  const [resA, setResA] = useState<SimulateResponse | null>(null);
  const [resB, setResB] = useState<SimulateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refresh saved-configs list whenever component mounts.
  useEffect(() => {
    setList(configs.loadAll());
  }, []);

  const run = async () => {
    const a = list.find((c) => c.name === nameA);
    const b = list.find((c) => c.name === nameB);
    if (!a || !b) {
      setError("Pick two saved configs (Save them on the Forward Sim tab first).");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const [rA, rB] = await Promise.all([
        postSimulate(a.params, { full: true, nsim: 512 }),
        postSimulate(b.params, { full: true, nsim: 512 }),
      ]);
      setResA(rA);
      setResB(rB);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-12 gap-4 p-4">
      <div className="col-span-3 bg-white border rounded p-4 space-y-3">
        <h3 className="font-semibold text-slate-700">Compare two configs</h3>
        {list.length < 2 && (
          <p className="text-xs text-amber-600">
            Save at least two named configs (on the Forward Sim tab) to compare.
          </p>
        )}
        <div className="space-y-1">
          <label className="text-sm">Config A</label>
          <select
            value={nameA}
            onChange={(e) => setNameA(e.target.value)}
            className="w-full px-2 py-1 border rounded text-sm"
          >
            <option value="">(pick one)</option>
            {list.map((c) => (
              <option key={c.name}>{c.name}</option>
            ))}
          </select>
        </div>
        <div className="space-y-1">
          <label className="text-sm">Config B</label>
          <select
            value={nameB}
            onChange={(e) => setNameB(e.target.value)}
            className="w-full px-2 py-1 border rounded text-sm"
          >
            <option value="">(pick one)</option>
            {list.map((c) => (
              <option key={c.name}>{c.name}</option>
            ))}
          </select>
        </div>
        <button
          onClick={run}
          disabled={loading || !nameA || !nameB || nameA === nameB}
          className="w-full px-3 py-2 rounded bg-accent text-white disabled:opacity-50"
        >
          {loading ? "Simulating both..." : "Compare"}
        </button>
        {error && <p className="text-xs text-red-600">{error}</p>}
      </div>

      <div className="col-span-9 space-y-4">
        {resA && (
          <div className="bg-white border rounded p-4">
            <h3 className="font-semibold text-slate-700 mb-2">A: {nameA}</h3>
            <RTHistogram rt={resA.rt} cat={resA.cat} />
          </div>
        )}
        {resB && (
          <div className="bg-white border rounded p-4">
            <h3 className="font-semibold text-slate-700 mb-2">B: {nameB}</h3>
            <RTHistogram rt={resB.rt} cat={resB.cat} />
          </div>
        )}
      </div>
    </div>
  );
}
