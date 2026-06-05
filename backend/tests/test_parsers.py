"""Tests for backend/parsers.py — twod3datanew + CSV + auto."""
import textwrap

import pytest

from backend import parsers


# ---- twod3datanew ------------------------------------------------------

def _make_twod3datanew_line(*, prop=0.2, count=20, qs=(300, 400, 500, 600, 700),
                            extras=(0.0, 0.0)) -> str:
    """One line = 5 categories × (prop, count, q1..q5, x1, x2)."""
    fields = []
    for _ in range(parsers.N_CATEGORIES):
        fields.append(str(prop))
        fields.append(str(count))
        fields.extend(str(q) for q in qs)
        fields.extend(str(x) for x in extras)
    return " ".join(fields)


def test_parse_twod3datanew_single_subject():
    """One subject = 2 condition lines."""
    line = _make_twod3datanew_line()
    text = f"{line}\n{line}\n"
    out = parsers.parse_twod3datanew(text)
    assert out["n_subjects"] == 1
    assert len(out["prop"]) == 2
    assert len(out["prop"][0]) == 5
    assert len(out["count"]) == 2
    assert len(out["quant"]) == 2
    assert len(out["quant"][0]) == 5
    assert len(out["quant"][0][0]) == 5  # 5 quantiles


def test_parse_twod3datanew_returns_first_subject():
    """Multi-subject file → we return subject 0."""
    line_a = _make_twod3datanew_line(prop=0.2, count=20)
    line_b = _make_twod3datanew_line(prop=0.3, count=30)
    text = "\n".join([line_a, line_a, line_b, line_b]) + "\n"
    out = parsers.parse_twod3datanew(text)
    assert out["n_subjects"] == 2
    # First subject's prop should be 0.2 across the board
    assert out["prop"][0][0] == pytest.approx(0.2)


def test_parse_twod3datanew_rejects_blank_file():
    with pytest.raises(ValueError, match="no non-blank lines"):
        parsers.parse_twod3datanew("   \n\n")


def test_parse_twod3datanew_rejects_odd_line_count():
    text = _make_twod3datanew_line() + "\n"  # only one line
    with pytest.raises(ValueError, match="divisible"):
        parsers.parse_twod3datanew(text)


def test_parse_twod3datanew_rejects_wrong_field_count():
    text = "1 2 3\n4 5 6\n"
    with pytest.raises(ValueError, match="expected"):
        parsers.parse_twod3datanew(text)


# ---- CSV ---------------------------------------------------------------

def test_parse_csv_minimal():
    """Two conditions, all 5 categories present in each."""
    rows = ["rt,cat,condition"]
    for cond in (1, 2):
        for cat in (1, 2, 3, 4, 5):
            for rt in (300, 350, 400, 450, 500):
                rows.append(f"{rt},{cat},{cond}")
    text = "\n".join(rows) + "\n"
    out = parsers.parse_csv(text)
    assert out["n_subjects"] == 1
    assert len(out["prop"]) == 2
    assert sum(out["count"][0]) == 25
    # Each category has 5 trials per condition, so proportions are 0.2 across the board
    for k in range(5):
        assert out["prop"][0][k] == pytest.approx(0.2)


def test_parse_csv_aggregates_proportions():
    """Skewed distribution shows up in proportions."""
    rows = ["rt,cat,condition"]
    for _ in range(80):
        rows.append("300,1,1")  # 80% of cond-1 trials are cat 1
    for _ in range(20):
        rows.append("400,2,1")
    for _ in range(50):
        rows.append("350,3,2")
    text = "\n".join(rows) + "\n"
    out = parsers.parse_csv(text)
    assert out["prop"][0][0] == pytest.approx(0.8)
    assert out["prop"][0][1] == pytest.approx(0.2)
    assert sum(out["prop"][1]) == pytest.approx(1.0)


def test_parse_csv_rejects_missing_columns():
    text = "rt,response\n300,1\n"
    with pytest.raises(ValueError, match="missing required columns"):
        parsers.parse_csv(text)


def test_parse_csv_rejects_bad_category():
    text = "rt,cat,condition\n300,7,1\n"
    with pytest.raises(ValueError, match="cat must be in 1..5"):
        parsers.parse_csv(text)


def test_parse_csv_rejects_bad_condition():
    text = "rt,cat,condition\n300,1,3\n"
    with pytest.raises(ValueError, match="condition must be 1 or 2"):
        parsers.parse_csv(text)


def test_parse_csv_rejects_empty():
    with pytest.raises(ValueError):
        parsers.parse_csv("")


# ---- parse_auto --------------------------------------------------------

def test_parse_auto_picks_twod3datanew_first():
    line = _make_twod3datanew_line()
    text = f"{line}\n{line}\n"
    out = parsers.parse_auto(text)
    assert out["n_subjects"] == 1   # twod3datanew picked


def test_parse_auto_falls_back_to_csv():
    text = textwrap.dedent("""
        rt,cat,condition
        300,1,1
        400,1,1
        500,1,1
        600,1,1
        700,1,1
        300,2,1
        400,2,1
        500,2,1
        600,2,1
        700,2,1
        300,3,1
        400,3,1
        500,3,1
        600,3,1
        700,3,1
        300,4,1
        400,4,1
        500,4,1
        600,4,1
        700,4,1
        300,5,1
        400,5,1
        500,5,1
        600,5,1
        700,5,1
        300,1,2
        400,1,2
        500,1,2
        600,1,2
        700,1,2
        300,2,2
        400,2,2
        500,2,2
        600,2,2
        700,2,2
        300,3,2
        400,3,2
        500,3,2
        600,3,2
        700,3,2
        300,4,2
        400,4,2
        500,4,2
        600,4,2
        700,4,2
        300,5,2
        400,5,2
        500,5,2
        600,5,2
        700,5,2
    """).strip()
    out = parsers.parse_auto(text)
    assert out["n_subjects"] == 1
    assert sum(out["count"][0]) == 25
    assert sum(out["count"][1]) == 25
