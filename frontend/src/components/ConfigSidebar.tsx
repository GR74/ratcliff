import { useState } from "react";

import * as configs from "../lib/configs";
import { ParamSet } from "../lib/types";
import { useAppStore } from "../store";

export function ConfigSidebar() {
  const params = useAppStore((s) => s.params);
  const setParams = useAppStore((s) => s.setParams);
  const [list, setList] = useState<configs.SavedConfig[]>(configs.loadAll());
  const [newName, setNewName] = useState("");

  const refresh = () => setList(configs.loadAll());

  const handleSave = () => {
    const name = newName.trim();
    if (!name) return;
    configs.save({ name, timestamp: Date.now(), params: { ...params } });
    setNewName("");
    refresh();
  };

  const handleLoad = (cfg: configs.SavedConfig) => {
    setParams(cfg.params);
  };

  const handleRemove = (name: string) => {
    configs.remove(name);
    refresh();
  };

  const handleExport = () => {
    const json = configs.exportAllAsJson();
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ratcliff-configs.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (file: File) => {
    file.text().then((txt) => {
      try {
        configs.importFromJson(txt);
        refresh();
      } catch (e) {
        alert("Import failed: " + (e as Error).message);
      }
    });
  };

  return (
    <div className="space-y-3">
      <h3 className="font-semibold text-slate-700">Saved configs</h3>

      <div className="flex gap-2">
        <input
          type="text"
          placeholder="config name"
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          className="flex-1 px-2 py-1 border rounded text-sm"
        />
        <button
          onClick={handleSave}
          disabled={!newName.trim()}
          className="px-2 py-1 text-sm rounded bg-accent text-white disabled:opacity-50"
        >
          Save
        </button>
      </div>

      <div className="space-y-1 max-h-80 overflow-y-auto">
        {list.length === 0 ? (
          <p className="text-xs text-slate-500 italic">No saved configs yet.</p>
        ) : (
          list.map((cfg) => (
            <div
              key={cfg.name}
              className="flex items-center justify-between text-sm border rounded px-2 py-1 bg-white"
            >
              <button
                onClick={() => handleLoad(cfg)}
                className="text-left truncate flex-1 hover:underline"
                title={new Date(cfg.timestamp).toLocaleString()}
              >
                {cfg.name}
              </button>
              <button
                onClick={() => handleRemove(cfg.name)}
                className="text-xs text-red-600 ml-2"
                title="Delete"
              >
                ✕
              </button>
            </div>
          ))
        )}
      </div>

      <div className="flex gap-2 pt-2 border-t">
        <button
          onClick={handleExport}
          className="flex-1 text-xs px-2 py-1 rounded bg-slate-200 hover:bg-slate-300"
        >
          Export
        </button>
        <label className="flex-1 text-xs px-2 py-1 rounded bg-slate-200 hover:bg-slate-300 cursor-pointer text-center">
          Import
          <input
            type="file"
            accept=".json,application/json"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleImport(f);
            }}
            className="hidden"
          />
        </label>
      </div>
    </div>
  );
}
