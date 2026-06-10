import { useEffect, useRef, useState } from "react";

import { FieldView } from "../components/FieldView";
import { ParamSliders } from "../components/ParamSliders";
import { postField } from "../lib/api";
import { FieldResponse } from "../lib/types";
import { useAppStore } from "../store";

export function FieldTab() {
  const params = useAppStore((s) => s.params);
  const [field, setField] = useState<FieldResponse | null>(null);
  const [mode, setMode] = useState<"single" | "mean">("single");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showThreshold, setShowThreshold] = useState(false);
  const playRef = useRef<number | null>(null);

  const generate = async () => {
    setLoading(true);
    setError(null);
    setPlaying(false);
    try {
      const f = await postField(params, { mode, nFrames: 48 });
      setField(f);
      setFrameIndex(0);
      setPlaying(true);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  // Animation loop: advance frameIndex ~12 fps while playing.
  useEffect(() => {
    if (!playing || !field) return;
    const id = window.setInterval(() => {
      setFrameIndex((i) => {
        const next = i + 1;
        if (next >= field.frames.length) {
          return 0; // loop
        }
        return next;
      });
    }, 80);
    playRef.current = id;
    return () => window.clearInterval(id);
  }, [playing, field]);

  return (
    <div className="grid grid-cols-12 gap-4 p-4">
      <div className="col-span-3 space-y-4">
        <div className="bg-white border rounded p-4 space-y-3">
          <h3 className="font-semibold text-slate-700">Evidence field</h3>
          <div className="flex gap-2 text-sm">
            <button
              onClick={() => setMode("single")}
              className={
                "flex-1 px-2 py-1 rounded " +
                (mode === "single" ? "bg-accent text-white" : "bg-slate-200")
              }
            >
              Single trial
            </button>
            <button
              onClick={() => setMode("mean")}
              className={
                "flex-1 px-2 py-1 rounded " +
                (mode === "mean" ? "bg-accent text-white" : "bg-slate-200")
              }
            >
              Mean field
            </button>
          </div>
          <button
            onClick={generate}
            disabled={loading}
            className="w-full px-3 py-2 rounded bg-accent text-white disabled:opacity-50"
          >
            {loading ? "Generating..." : "Generate field"}
          </button>
          {mode === "mean" && (
            <p className="text-[10px] text-amber-600">
              Mean field averages 64 trials — slower to generate on CPU.
            </p>
          )}
          {error && <p className="text-xs text-red-600">{error}</p>}
        </div>

        <div className="bg-white border rounded p-4">
          <ParamSliders />
        </div>
      </div>

      <div className="col-span-9 space-y-3">
        <div className="bg-white border rounded p-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="font-semibold text-slate-700">
              3D accumulator surface{" "}
              {field && (
                <span className="text-xs font-normal text-slate-500">
                  — frame {frameIndex + 1}/{field.frames.length} (step{" "}
                  {field.steps[frameIndex]}/{field.nstep})
                </span>
              )}
            </h2>
            {field && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPlaying((p) => !p)}
                  className="px-3 py-1 text-sm rounded bg-slate-200 hover:bg-slate-300"
                >
                  {playing ? "Pause" : "Play"}
                </button>
                <label className="flex items-center gap-1 text-xs text-slate-600 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showThreshold}
                    onChange={(e) => setShowThreshold(e.target.checked)}
                  />
                  threshold plane
                </label>
              </div>
            )}
          </div>

          {field ? (
            <>
              <FieldView field={field} frameIndex={frameIndex} showThreshold={showThreshold} />
              <input
                type="range"
                min={0}
                max={field.frames.length - 1}
                value={frameIndex}
                onChange={(e) => {
                  setPlaying(false);
                  setFrameIndex(Number(e.target.value));
                }}
                className="w-full mt-2 accent-accent"
              />
              <p className="text-xs text-slate-500 mt-1">
                Drag to orbit, scroll to zoom. In single-trial mode the gold trail
                tracks the winning region — the marker flashes at the moment the
                decision commits. Toggle "threshold plane" to overlay the threshold.
              </p>
            </>
          ) : (
            <div className="h-[500px] flex items-center justify-center text-slate-400 text-sm italic">
              Click "Generate field" to render the evidence accumulator surface.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
