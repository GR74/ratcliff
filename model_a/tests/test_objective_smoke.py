"""Smoke tests for model_a/objective.py."""
import jax.numpy as jnp
import pytest


def test_objective_module_imports():
    from model_a import objective
    # Confirm public API
    assert hasattr(objective, "condition_g2_vectorized")
    assert hasattr(objective, "fofs_new")
    assert hasattr(objective, "COND_MAP")


def test_cond_map_has_4_conditions():
    from model_a import objective
    assert len(objective.COND_MAP) == 4
    for (di, bi) in objective.COND_MAP:
        assert isinstance(di, int)
        assert isinstance(bi, int)
