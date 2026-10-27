"""Guard the scientific study's event definition and uncertainty accounting."""
import jax.numpy as jnp
import numpy as np

from scripts.check_kl_winner_fidelity import observe, spectrum
from scripts.report_kl_winner_fidelity import paired_binary_interval


def test_mode_cap_reports_achieved_variance_instead_of_target():
    mask, count, retained, _ = spectrum(jnp.ones((4, 4)), .99, 2)
    assert count == 2
    assert retained == .125
    assert mask.sum() == 2


def test_no_discordances_still_have_nonzero_uncertainty():
    events = np.zeros(100, dtype=bool)
    low, high = paired_binary_interval(events, events)
    assert low < 0 < high
    assert np.isclose(low, -high)


def test_paired_interval_direction_and_reversal():
    a = np.ones(100, dtype=bool)
    b = np.zeros(100, dtype=bool)
    low, high = paired_binary_interval(a, b)
    assert 90 < low < high == 100
    np.testing.assert_allclose(paired_binary_interval(b, a), [-high, -low])


def test_observer_distinguishes_a_win_from_horizon_censoring():
    drift = jnp.array([[1., 0.], [0., 0.]], dtype=jnp.float32)
    fields = jnp.zeros((2, 400, 2, 2), dtype=jnp.float32)
    fields = fields.at[1].set(-drift)
    u = jnp.full((2, 10), .5)
    zones = jnp.array([[2, 3], [4, 5]])
    result = np.asarray(observe(fields, u, drift, zones))
    np.testing.assert_array_equal(result[0], [340, 2, 0, 1, 14])
    np.testing.assert_array_equal(result[1], [4200, 5, 0, 0, 400])
