import { DEFAULT_FULL_PARAMS, FULL_PARAM_LABELS } from "../lib/types";

interface Props {
  fittedParams: number[];
  trueParams?: number[];
}

const ACTIVE_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12];

export function RecoveryTable({ fittedParams, trueParams }: Props) {
  const truth = trueParams ?? DEFAULT_FULL_PARAMS;
  return (
    <table className="w-full text-sm border-collapse">
      <thead>
        <tr className="border-b text-slate-600">
          <th className="text-left py-1 px-2">Param</th>
          <th className="text-right py-1 px-2">True</th>
          <th className="text-right py-1 px-2">Fitted</th>
          <th className="text-right py-1 px-2">% error</th>
        </tr>
      </thead>
      <tbody>
        {ACTIVE_INDICES.map((i) => {
          const trueVal = truth[i];
          const got = fittedParams[i];
          const err = trueVal !== 0 ? (Math.abs(got - trueVal) / Math.abs(trueVal)) * 100 : 0;
          const color = err < 5 ? "text-green-700" : err < 10 ? "text-amber-600" : "text-red-600";
          return (
            <tr key={i} className="border-b">
              <td className="py-1 px-2 font-mono">{FULL_PARAM_LABELS[i]}</td>
              <td className="py-1 px-2 text-right font-mono">{trueVal.toFixed(3)}</td>
              <td className="py-1 px-2 text-right font-mono">{got.toFixed(3)}</td>
              <td className={`py-1 px-2 text-right font-mono ${color}`}>{err.toFixed(1)}%</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
