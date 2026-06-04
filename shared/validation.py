"""
Aggregate-statistical comparison between Fortran reference and JAX port.

Conventions:
  - Proportions use absolute tolerance (they live in [0, 1]).
  - RT quantiles use relative tolerance (they span ~300-1500 ms, wide range).
  - All functions enforce matching input shapes (raise ValueError on mismatch).
  - Return values are JSON-serializable dicts (Python bools/floats, no NumPy scalars).
"""
__all__ = ["proportions_match", "quantiles_match", "aggregate_match"]
import numpy as np

# Denominator floor for relative-tolerance comparison. Prevents 0/0 -> NaN
# when both quantiles are zero (e.g. an empty-category placeholder).
_QUANT_DENOM_FLOOR = 1e-9


def proportions_match(a, b, abs_tol: float = 0.005):
    """
    Compare response proportions elementwise within an absolute tolerance.
    Returns (ok, report_dict).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"proportions_match shape mismatch: {a.shape} vs {b.shape}")
    diffs = np.abs(a - b)
    return bool(np.all(diffs <= abs_tol)), {
        "max_abs_diff": float(diffs.max()),
        "tol": abs_tol,
    }


def quantiles_match(a, b, rel_tol: float = 0.01):
    """
    Compare RT quantiles within a relative tolerance |a-b| / max(|a|, |b|, eps).
    Returns (ok, report_dict).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"quantiles_match shape mismatch: {a.shape} vs {b.shape}")
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), _QUANT_DENOM_FLOOR)
    rels = np.abs(a - b) / denom
    return bool(np.all(rels <= rel_tol)), {
        "max_rel_diff": float(rels.max()),
        "tol": rel_tol,
    }


def aggregate_match(
    obs_prop, sim_prop, obs_quant, sim_quant,
    prop_abs_tol: float = 0.005, quant_rel_tol: float = 0.01,
):
    """Combined gate: proportions AND quantiles within their respective tolerances."""
    prop_ok, prop_report = proportions_match(obs_prop, sim_prop, prop_abs_tol)
    quant_ok, quant_report = quantiles_match(obs_quant, sim_quant, quant_rel_tol)
    return {
        "passed": prop_ok and quant_ok,
        "prop_passed": prop_ok,
        "quant_passed": quant_ok,
        "prop": prop_report,
        "quant": quant_report,
    }


# TODO(stage-5): add summarize(result) -> str for one-line benchmark report rows
#   format: "FAILED: prop max_diff=0.012 (tol=0.005), quant max_rel=0.003 (tol=0.01, OK)"
