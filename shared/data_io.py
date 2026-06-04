"""Parsers for Ratcliff observed RT/proportion data files."""
from pathlib import Path

import numpy as np


def load_twod24data(path: str | Path) -> dict[str, np.ndarray]:
    """
    Parse a Ratcliff RT/proportion data file: N non-empty lines × 27 fields
    = 3 categories × (prop, count, q1..q5, x1, x2). The x1/x2 RT-extreme
    fields are ignored.

    Returns dict with:
      prop  : (N, 3) float — observed response proportion per category
      count : (N, 3) int   — observed trial count per category
      quant : (N, 5, 3) float — 5 RT quantiles per category
    """
    raw_lines = Path(path).read_text().splitlines()
    # Keep (file_line_number_1_based, content) for non-blank lines.
    indexed = [(i, ln) for i, ln in enumerate(raw_lines, start=1) if ln.strip()]
    n_lines = len(indexed)

    prop = np.zeros((n_lines, 3), dtype=np.float64)
    count = np.zeros((n_lines, 3), dtype=np.int64)
    quant = np.zeros((n_lines, 5, 3), dtype=np.float64)

    for record_idx, (file_lineno, ln) in enumerate(indexed):
        fields = ln.split()
        if len(fields) != 27:
            raise ValueError(
                f"{path}:{file_lineno}: expected 27 fields, got {len(fields)} — line: {ln!r}"
            )
        for cat in range(3):
            base = cat * 9
            prop[record_idx, cat] = float(fields[base])
            count[record_idx, cat] = int(float(fields[base + 1]))
            for q in range(5):
                quant[record_idx, q, cat] = float(fields[base + 2 + q])
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
