from pathlib import Path

import numpy as np
import pytest

from shared import data_io

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "twod24data"


def test_data_file_exists():
    assert DATA_PATH.exists(), f"missing {DATA_PATH}"


def test_load_twod24data_returns_expected_shape():
    d = data_io.load_twod24data(DATA_PATH)
    # 16 subjects × 4 conditions = 64 condition-records
    assert d["prop"].shape == (64, 3)
    assert d["count"].shape == (64, 3)
    assert d["quant"].shape == (64, 5, 3)


def test_proportions_sum_close_to_one_per_line():
    d = data_io.load_twod24data(DATA_PATH)
    sums = d["prop"].sum(axis=1)
    # Some lines have empty categories — tolerate ±0.05
    assert np.all(np.abs(sums - 1.0) < 0.05), f"max deviation {np.abs(sums - 1.0).max()}"


def test_counts_are_nonneg_integers():
    d = data_io.load_twod24data(DATA_PATH)
    assert d["count"].dtype.kind in ("i", "u")
    assert np.all(d["count"] >= 0)


def test_quantiles_are_monotone_per_category():
    d = data_io.load_twod24data(DATA_PATH)
    # For categories with nonzero count, quantiles must be non-decreasing along axis 1
    for ci in range(64):
        for cat in range(3):
            if d["count"][ci, cat] > 0:
                q = d["quant"][ci, :, cat]
                assert np.all(np.diff(q) >= 0), f"line {ci} cat {cat} not monotone: {q}"


def test_grouped_by_subject_returns_4_conditions_per_subject():
    d = data_io.load_twod24data(DATA_PATH)
    g = data_io.group_by_subject(d, conditions_per_subject=4)
    assert g["prop"].shape == (16, 4, 3)
    assert g["count"].shape == (16, 4, 3)
    assert g["quant"].shape == (16, 4, 5, 3)


def test_load_twod3datanew_returns_expected_shape():
    from shared import data_io
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "data" / "twod3datanew"
    d = data_io.load_twod3datanew(path)
    # 2 conditions per subject, 5 categories, 5 quantiles
    assert d["prop"].ndim == 3 and d["prop"].shape[1:] == (2, 5)
    assert d["count"].ndim == 3 and d["count"].shape[1:] == (2, 5)
    assert d["quant"].ndim == 4 and d["quant"].shape[1:] == (2, 5, 5)
    # Proportions per condition sum close to 1
    prop_sums = d["prop"].sum(axis=2)
    assert np.all(np.abs(prop_sums - 1.0) < 0.05)


def test_load_twod3datanew_first_record_values():
    from shared import data_io
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "data" / "twod3datanew"
    d = data_io.load_twod3datanew(path)
    # First subject, first condition — sanity check
    assert d["prop"][0, 0, 0] >= 0.0
    # At least one category should have non-zero count
    assert d["count"][0, 0].sum() > 0
