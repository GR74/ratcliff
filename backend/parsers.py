"""Data file parsers for the upload endpoint.

Two formats supported:
1. twod3datanew — Ratcliff's existing aggregate file format. Each pair of
   non-blank lines is one subject's 2 conditions. Fields per line:
   5 categories × (prop, count, q1..q5, x1, x2) = 45.
2. Generic CSV with columns: RT, response_cat, condition[, subject_id].

The output shape returned by either parser is suitable for direct ingestion
by fofs_b_new: dict with prop (2, 5), count (2, 5), quant (2, 5, 5) — for a
single subject. If the upload contains multiple subjects, the caller picks
one (typically subject index 0 for the demo).
"""
import csv
import io
from typing import Optional

import numpy as np


N_CONDITIONS = 2
N_CATEGORIES = 5
N_QUANTILES = 5
FIELDS_PER_CAT = 2 + N_QUANTILES + 2   # prop, count, q1..q5, x1, x2 = 9
FIELDS_PER_LINE = N_CATEGORIES * FIELDS_PER_CAT  # 45

CSV_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


def parse_twod3datanew(text: str) -> dict:
    """Parse a twod3datanew-format string. Returns one subject's worth of data.

    text : full file contents (multiline string)
    Returns: {
        "prop":   list[list[float]] shape (2, 5),
        "count":  list[list[int]]   shape (2, 5),
        "quant":  list[list[list[float]]] shape (2, 5, 5),
        "n_subjects": int  # for caller info; we return subject 0's slice
    }
    Raises ValueError on malformed input.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("twod3datanew: file has no non-blank lines")
    if len(lines) % N_CONDITIONS != 0:
        raise ValueError(
            f"twod3datanew: {len(lines)} non-blank lines is not divisible by "
            f"n_conditions={N_CONDITIONS}"
        )
    n_subjects = len(lines) // N_CONDITIONS

    prop = np.zeros((n_subjects, N_CONDITIONS, N_CATEGORIES), dtype=np.float64)
    count = np.zeros((n_subjects, N_CONDITIONS, N_CATEGORIES), dtype=np.int64)
    quant = np.zeros(
        (n_subjects, N_CONDITIONS, N_QUANTILES, N_CATEGORIES), dtype=np.float64
    )

    for line_idx, ln in enumerate(lines):
        fields = ln.split()
        if len(fields) != FIELDS_PER_LINE:
            raise ValueError(
                f"twod3datanew line {line_idx+1}: expected {FIELDS_PER_LINE} "
                f"fields, got {len(fields)}"
            )
        s, c = divmod(line_idx, N_CONDITIONS)
        for k in range(N_CATEGORIES):
            base = k * FIELDS_PER_CAT
            prop[s, c, k] = float(fields[base])
            count[s, c, k] = int(float(fields[base + 1]))
            for q in range(N_QUANTILES):
                quant[s, c, q, k] = float(fields[base + 2 + q])

    # Return subject 0's slice. Multi-subject support is a UI concern: the
    # frontend should expose a subject picker if n_subjects > 1.
    return {
        "prop": prop[0].tolist(),
        "count": count[0].astype(int).tolist(),
        "quant": quant[0].tolist(),
        "n_subjects": int(n_subjects),
    }


def _expect_columns(reader: csv.DictReader, required: tuple[str, ...]) -> None:
    fns = reader.fieldnames or []
    missing = [c for c in required if c not in fns]
    if missing:
        raise ValueError(
            f"CSV: missing required columns {missing}; got fields {fns}"
        )


def parse_csv(text: str) -> dict:
    """Parse a trial-level CSV. Required columns: rt, cat, condition.

    Optional column: subject_id (we aggregate over subjects unless caller
    pre-filters).

    Aggregation produces, per condition × category:
      - prop  = proportion of trials
      - count = trial count
      - quant = 5 RT quantiles at (0.1, 0.3, 0.5, 0.7, 0.9)

    Returns same shape as parse_twod3datanew.
    Raises ValueError on malformed input or missing columns.
    """
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("CSV: file appears to be empty")
    # Be forgiving on case
    reader.fieldnames = [fn.strip().lower() for fn in reader.fieldnames]
    _expect_columns(reader, ("rt", "cat", "condition"))

    rows: list[tuple[float, int, int]] = []
    for line_no, raw in enumerate(reader, start=2):  # data starts on line 2
        try:
            rt = float(raw["rt"])
            cat = int(raw["cat"])
            cond = int(raw["condition"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"CSV line {line_no}: {e}") from e
        if cond not in (1, 2):
            raise ValueError(
                f"CSV line {line_no}: condition must be 1 or 2, got {cond}"
            )
        if cat not in (1, 2, 3, 4, 5):
            raise ValueError(
                f"CSV line {line_no}: cat must be in 1..5, got {cat}"
            )
        rows.append((rt, cat, cond))

    if not rows:
        raise ValueError("CSV: no data rows")

    prop = np.zeros((N_CONDITIONS, N_CATEGORIES), dtype=np.float64)
    count = np.zeros((N_CONDITIONS, N_CATEGORIES), dtype=np.int64)
    quant = np.zeros((N_CONDITIONS, N_QUANTILES, N_CATEGORIES), dtype=np.float64)

    arr = np.array(rows, dtype=[("rt", "f8"), ("cat", "i4"), ("cond", "i4")])
    for ci, cond in enumerate((1, 2)):
        mask_cond = arr["cond"] == cond
        n_cond = int(mask_cond.sum())
        if n_cond == 0:
            continue
        for ki, cat in enumerate((1, 2, 3, 4, 5)):
            mask = mask_cond & (arr["cat"] == cat)
            n_cat = int(mask.sum())
            count[ci, ki] = n_cat
            prop[ci, ki] = n_cat / max(n_cond, 1)
            if n_cat >= N_QUANTILES:
                rts_cat = arr["rt"][mask]
                quant[ci, :, ki] = np.quantile(rts_cat, CSV_QUANTILES)

    return {
        "prop": prop.tolist(),
        "count": count.astype(int).tolist(),
        "quant": quant.transpose(0, 1, 2).tolist(),  # (cond, q, cat)
        "n_subjects": 1,
    }


def parse_auto(text: str) -> dict:
    """Try twod3datanew first, fall back to CSV. Raise the CSV error if both fail."""
    try:
        return parse_twod3datanew(text)
    except ValueError:
        pass
    return parse_csv(text)
