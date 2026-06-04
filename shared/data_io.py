"""Parsers for Ratcliff observed RT/proportion data files."""
from pathlib import Path

import numpy as np


def load_twod24data(path):
    """
    Parse `twod24data` as 64 lines × 27 fields = 3 categories × (prop, count, q1..q5, x1, x2).

    Returns dict with:
      prop  : (64, 3) float — observed response proportion per category
      count : (64, 3) int   — observed trial count per category
      quant : (64, 5, 3) float — 5 RT quantiles per category
    """
    lines = [
        ln for ln in Path(path).read_text().splitlines() if ln.strip()
    ]
    n_lines = len(lines)

    prop = np.zeros((n_lines, 3), dtype=np.float64)
    count = np.zeros((n_lines, 3), dtype=np.int64)
    quant = np.zeros((n_lines, 5, 3), dtype=np.float64)

    for i, ln in enumerate(lines):
        fields = ln.split()
        if len(fields) != 27:
            raise ValueError(
                f"line {i}: expected 27 fields, got {len(fields)} — line: {ln!r}"
            )
        for cat in range(3):
            base = cat * 9
            prop[i, cat] = float(fields[base])
            count[i, cat] = int(float(fields[base + 1]))
            for q in range(5):
                quant[i, q, cat] = float(fields[base + 2 + q])
            # fields base+7 (x1) and base+8 (x2) ignored — RT extremes, not used by fofs

    return {"prop": prop, "count": count, "quant": quant}


def group_by_subject(data, conditions_per_subject: int = 4):
    """Reshape (64, ...) flat arrays into (n_subjects, conditions_per_subject, ...)."""
    n_records = data["prop"].shape[0]
    if n_records % conditions_per_subject != 0:
        raise ValueError(
            f"{n_records} records is not divisible by {conditions_per_subject}"
        )
    n_subjects = n_records // conditions_per_subject
    return {
        "prop": data["prop"].reshape(n_subjects, conditions_per_subject, 3),
        "count": data["count"].reshape(n_subjects, conditions_per_subject, 3),
        "quant": data["quant"].reshape(n_subjects, conditions_per_subject, 5, 3),
    }
