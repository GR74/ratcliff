import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shared.staged_fit import TrialStage


@pytest.mark.parametrize("model", ["a", "b"])
def test_fit_drivers_forward_stage_budgets_and_simulation_options(monkeypatch, model):
    if model == "a":
        from model_a import fit, objective
        name = "fofs_new"
        driver = fit.fit_simplex
        count = 10
    else:
        from model_b import fit, objective
        name = "fofs_b_new"
        driver = fit.fit_simplex_b
        count = 13
    calls = []
    data = {"sentinel": True}

    def fake_objective(p, received_data, key, **kwargs):
        assert received_data is data
        calls.append(kwargs)
        return jnp.sum((p - 1.) ** 2)

    monkeypatch.setattr(objective, name, fake_objective)
    result = driver(
        data, jax.random.key(0), np.ones(count) * 1.1,
        nsim=32, maxiter=6, chunk_size=8, time_block_size=4,
        trial_schedule=[TrialStage(8, 3), TrialStage(32, 3)],
    )
    assert {c["nsim"] for c in calls} == {8, 32}
    assert all(c["time_block_size"] == 4 and c["stop_early"] for c in calls)
    assert result.n_evals == len(calls)
    assert result.stages[-1]["nsim"] == 32
