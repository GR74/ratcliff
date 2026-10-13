"""Stage transitions recompute losses and finish with full-budget refinement."""
import numpy as np
import pytest

from shared.staged_fit import TrialStage, minimize_staged, validate_schedule


def test_final_stage_refines_on_full_budget():
    calls = []
    updates = []
    stages = []

    def loss(p, nsim):
        calls.append(nsim)
        target = 10. if nsim == 16 else 2.
        return np.sum((p - target) ** 2)

    result = minimize_staged(
        loss, [0.], nsim=64, maxiter=100, tol=1e-6,
        schedule=[TrialStage(16, 30), TrialStage(64, 70)],
        on_update=lambda n, value, p: updates.append(n),
        on_stage=stages.append,
    )
    assert result.params[0] == pytest.approx(2., abs=1e-3)
    assert result.loss < 1e-6
    assert result.n_evals == len(calls)
    assert updates == list(range(1, len(calls) + 1))
    assert [s["nsim"] for s in stages] == [16, 64]
    assert sum(s["n_evals"] for s in stages) == result.n_evals
    assert calls[-1] == 64


@pytest.mark.parametrize("schedule,match", [
    ([], "positive"),
    ([TrialStage(64, 10), TrialStage(16, 10)], "increase"),
    ([TrialStage(16, 20)], "full"),
    ([TrialStage(16, 3), TrialStage(64, 3)], "sum"),
])
def test_invalid_schedules_rejected(schedule, match):
    with pytest.raises(ValueError, match=match):
        validate_schedule(64, 20, schedule)
