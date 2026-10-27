"""Finite-look behavioral inference, independent of the simulator and GPU."""
import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("behavior_base_metrics", Path(__file__).with_name("fft200_comparison_metrics.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
SENSITIVITY = {"low_support_0.05", "low_support_0.2"}


def look_alpha(protocol, completed):
    if completed not in protocol["look_after_rounds"]:
        raise ValueError("Inference is allowed only at registered complete looks")
    i = protocol["look_after_rounds"].index(completed)
    return protocol["family_alpha"]*protocol["look_weights"][i]/protocol["core_registered_endpoints"]


def seed_for(protocol, round_index, method_index, case_index):
    if not 0 <= round_index < protocol["max_independent_rounds"]:
        raise ValueError("Invalid round")
    return protocol["seed_base"]+100000*method_index+1000*case_index+round_index


def spatial_lower(a, b, alpha_lower):
    """Average valid lower bounds on 64 absolute projected mean differences."""
    aa, bb = [s["positions"][s["crossed"]] for s in (a, b)]
    if not len(aa) or not len(bb):
        return 0.
    xy = lambda p: 2*np.column_stack([p % 160, p // 160])
    delta = xy(bb).mean(axis=0)-xy(aa).mean(axis=0)
    theta = np.arange(64)*np.pi/64
    directions = np.column_stack([np.cos(theta), np.sin(theta)])
    ranges = 318*np.abs(directions[:, 0])+198*np.abs(directions[:, 1])
    # 128 two-sided mean intervals, allocated from the lower-bound budget.
    alpha_mean = alpha_lower/(2*64)
    radii = ranges*np.sqrt(np.log(2/alpha_mean)/2)*(1/np.sqrt(len(aa))+1/np.sqrt(len(bb)))
    return float(np.maximum(0, np.abs(directions @ delta)-radii).mean())


def compare(a, b, masks, alpha, geometry, core=True):
    rows = base.compare(a, b, masks, alpha, geometry=geometry, spatial_alpha=alpha/2)
    av, bv = base.probability_vectors(a, masks), base.probability_vectors(b, masks)
    for row in rows:
        name = row["metric"]
        row["registered"] = core and name not in SENSITIVITY
        row["interval_scope"] = "sequential_family_adjusted" if row["registered"] else "pointwise_diagnostic"
        if name in av:
            row.update(reference=float(av[name].mean()), candidate=float(bv[name].mean()),
                       counts=[int(av[name].sum()), int(bv[name].sum())], totals=[len(av[name]), len(bv[name])])
            if name in SENSITIVITY:
                ai = base.binary_interval(row["counts"][0], len(av[name]), .025)
                bi = base.binary_interval(row["counts"][1], len(bv[name]), .025)
                row["interval"] = list(base.interval_difference(ai, bi))
                row["status"] = base.interval_status(row["interval"], row["margin"])
        elif name.startswith("rt_region_"):
            category = int(name.split("_")[2]); q = float(name.split("_q")[1])
            values = [s["rt"][(s["cat"] == category) & s["crossed"]] for s in (a, b)]
            row.update(reference=float(np.quantile(values[0], q)) if len(values[0]) else np.nan,
                       candidate=float(np.quantile(values[1], q)) if len(values[1]) else np.nan)
        else:
            low = spatial_lower(a, b, alpha/2)
            high = row["interval"][1]
            row["interval"] = [low, high]
            row["status"] = "inconclusive" if low > high else "pass" if high <= 2 else "fail" if low > 2 else "inconclusive"
            row["counts"] = [int(a["crossed"].sum()), int(b["crossed"].sum())]
            row["note"] = "Upper/lower bounds each use half the endpoint alpha; fixed 64 directions, independent samples."
        if not row["registered"]:
            row["diagnostic_relation"] = row.pop("status")
            row["status"] = "diagnostic_only"
    return rows


def cell_status(endpoints):
    rows = [r for r in endpoints if r["registered"]]
    if len(rows) != 33:
        raise ValueError("Expected all 33 registered endpoints")
    if any(r["status"] == "fail" for r in rows):
        return "material_mismatch"
    if all(r["status"] == "pass" for r in rows):
        return "equivalent_within_tolerances"
    return "inconclusive"


def stop_decision(records, core_cases):
    cells = [r for r in records if r["case"] in core_cases]
    expected = {(c, i) for c in core_cases for i in range(2)}
    if len(cells) != 6 or {(r["case"], r["condition"]) for r in cells} != expected:
        raise ValueError("All six unique core cells are required before stopping")
    statuses = [cell_status(r["endpoints"]) for r in cells]
    decisive = all(s != "inconclusive" for s in statuses)
    return dict(stop=decisive, status=("equivalent_in_core_scope" if all(s == "equivalent_within_tolerances" for s in statuses)
                else "material_mismatch_in_core_scope" if any(s == "material_mismatch" for s in statuses) else "inconclusive"),
                core_cells=[dict(case=r["case"], condition=r["condition"], status=s) for r, s in zip(cells, statuses)])


def validate_sample(values, n):
    for name in ("rt", "cat", "steps", "positions", "crossed"):
        if values[name].shape != (2, n):
            raise ValueError(f"Invalid shape: {name}")
    if not np.isfinite(values["rt"]).all() or not np.isin(values["cat"], np.arange(1, 6)).all():
        raise ValueError("Invalid RT or response category")
    if not ((values["positions"] >= 0) & (values["positions"] < 16000)).all():
        raise ValueError("Invalid winning-position index")
    if not ((values["steps"] >= 1) & (values["steps"] <= 400)).all():
        raise ValueError("Invalid first-passage step")


def condition(values, ci):
    return {k: values[k][ci] for k in ("rt", "cat", "steps", "positions", "crossed")}
