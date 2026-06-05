import { ParamSet, PARAM_DEFS } from "../lib/types";
import { useAppStore } from "../store";

interface Props {
  onChange?: (params: ParamSet) => void;
}

export function ParamSliders({ onChange }: Props) {
  const params = useAppStore((s) => s.params);
  const setParam = useAppStore((s) => s.setParam);
  const resetParams = useAppStore((s) => s.resetParams);

  const handleChange = (key: keyof ParamSet, value: number) => {
    setParam(key, value);
    if (onChange) {
      onChange({ ...params, [key]: value });
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-slate-700">Parameters</h3>
        <button
          onClick={() => {
            resetParams();
            if (onChange) onChange(useAppStore.getState().params);
          }}
          className="text-xs px-2 py-1 rounded bg-slate-200 hover:bg-slate-300"
        >
          Reset
        </button>
      </div>
      {PARAM_DEFS.map((def) => (
        <div key={def.key} className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <label
              htmlFor={`slider-${def.key}`}
              title={def.desc}
              className="font-mono cursor-help"
            >
              {def.label}
            </label>
            <input
              type="number"
              step={def.step}
              min={def.min}
              max={def.max}
              value={params[def.key]}
              onChange={(e) => handleChange(def.key, Number(e.target.value))}
              className="w-20 px-1 py-0.5 border rounded text-right text-xs"
            />
          </div>
          <input
            id={`slider-${def.key}`}
            type="range"
            min={def.min}
            max={def.max}
            step={def.step}
            value={params[def.key]}
            onChange={(e) => handleChange(def.key, Number(e.target.value))}
            className="w-full accent-accent"
          />
          <p className="text-[10px] text-slate-500">{def.desc}</p>
        </div>
      ))}
    </div>
  );
}
