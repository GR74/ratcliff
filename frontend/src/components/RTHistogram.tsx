import Plot from "react-plotly.js";

interface Props {
  rt: number[];
  cat: number[];
  title?: string;
}

const CAT_COLORS = ["#2563eb", "#16a34a", "#dc2626", "#ea580c", "#7c3aed"];

export function RTHistogram({ rt, cat, title }: Props) {
  const traces = [1, 2, 3, 4, 5].map((c, i) => ({
    name: `Cat ${c}`,
    type: "histogram" as const,
    x: rt.filter((_, idx) => cat[idx] === c),
    opacity: 0.65,
    nbinsx: 30,
    marker: { color: CAT_COLORS[i] },
  }));
  return (
    <Plot
      data={traces}
      layout={{
        barmode: "overlay",
        title: title ?? "Reaction time by category",
        xaxis: { title: { text: "RT (ms)" } },
        yaxis: { title: { text: "Count" } },
        height: 400,
        margin: { l: 50, r: 20, t: 50, b: 50 },
        legend: { orientation: "h", y: -0.2 },
      }}
      style={{ width: "100%" }}
      config={{ displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"] }}
    />
  );
}
