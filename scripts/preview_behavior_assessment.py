"""Descriptive saved-data preview; never a registered look or stop decision."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts import behavior_assessment_stats as stats
from scripts.run_behavior_assessment import read_batch, write


def main(rounds):
    root = Path(__file__).resolve().parents[1]
    raw = root/".benchmark-results/2026-10-26-behavioral-assessment-20-runs"
    protocol = json.loads((raw/"manifest.json").read_text())["protocol"]
    for r in range(rounds):
        if not (raw/"rounds"/f"round-{r+1:02d}.json").exists():
            raise ValueError("Requested prefix is not complete")
    geometry_file = np.load(raw/"geometry.npz")
    geometry = stats.base.projection_geometry()
    rows = []
    for ci, case in enumerate(protocol["cases"]):
        if case["name"] not in protocol["core_case_names"]: continue
        samples = {}
        for mi, method in enumerate(protocol["methods"]):
            pieces = [read_batch(raw/"trials"/f"round-{r+1:02d}"/case["name"]/f"{method}.npz",
                                 stats.seed_for(protocol, r, mi, ci), protocol["trials_per_condition_per_round"]) for r in range(rounds)]
            if any(p is None for p in pieces): raise ValueError("Missing committed batch")
            samples[method] = {k: np.concatenate([p[k] for p in pieces], axis=1) for k in ("rt", "cat", "positions", "crossed")}
        for condition in range(2):
            a = {k:v[condition] for k, v in samples["fft199"].items()}
            masks = geometry_file[f"{ci}_{condition}_masks"]
            av = stats.base.probability_vectors(a, masks)
            for method in protocol["methods"][1:]:
                b = {k:v[condition] for k, v in samples[method].items()}
                bv = stats.base.probability_vectors(b, masks)
                probabilities = [100*(bv[f"region_{k}"].mean()-av[f"region_{k}"].mean()) for k in range(1, 6)]
                quantile_errors = []
                for category in range(1, 6):
                    ar = a["rt"][(a["cat"] == category) & a["crossed"]]
                    br = b["rt"][(b["cat"] == category) & b["crossed"]]
                    if min(len(ar), len(br)) >= 10:
                        for q, error in zip(stats.base.QUANTILES, np.quantile(br, stats.base.QUANTILES)-np.quantile(ar, stats.base.QUANTILES)):
                            quantile_errors.append(dict(region=category, quantile=q, difference_ms=float(error), counts=[len(ar), len(br)]))
                worst = max(quantile_errors, key=lambda x: abs(x["difference_ms"])) if quantile_errors else None
                rows.append(dict(case=case["name"], condition=condition+1, method=method,
                    n=len(a["rt"]), reference_low_support_percent=100*av["low_support_0.1"].mean(),
                    candidate_low_support_percent=100*bv["low_support_0.1"].mean(),
                    low_support_difference_pp=100*(bv["low_support_0.1"].mean()-av["low_support_0.1"].mean()),
                    region_differences_pp=probabilities, maximum_region_difference_pp=max(map(abs, probabilities)),
                    mean_crossing_rt_difference_ms=float(b["rt"][b["crossed"]].mean()-a["rt"][a["crossed"]].mean()),
                    worst_conditional_rt_quantile=worst, conditional_rt_quantile_differences=quantile_errors,
                    empirical_spatial_sw1_pixels=stats.base.spatial_distance(a, b, geometry),
                    reference_non_crossings=int((~a["crossed"]).sum()), candidate_non_crossings=int((~b["crossed"]).sum())))
    result = dict(scope="Unscheduled descriptive preview of core cases; no confidence bounds, tests, pass/fail labels or stopping decisions.",
                  rounds=rounds, rows=rows)
    write(root/"docs/results/2026-10-26-behavioral-assessment-20-runs/descriptive-previews"/f"round-{rounds:02d}.json", result)
    compact = [{k:r[k] for k in ("case", "condition", "method", "low_support_difference_pp", "maximum_region_difference_pp", "mean_crossing_rt_difference_ms", "empirical_spatial_sw1_pixels")} for r in rows]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    main(parser.parse_args().rounds)
