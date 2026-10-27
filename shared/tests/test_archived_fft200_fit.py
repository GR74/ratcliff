from scripts.run_archived_fft200_fit import fft_objective


def test_noise_override_preserves_training_and_optimizer_inputs():
    calls = []
    marker = object()
    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return marker
    p, data, key = object(), object(), object()
    result = fft_objective(original)(p, data, key, nsim=9000, chunk_size=8,
        use_kl=True, stable_kl_noise=True, time_block_size=32,
        stop_early=True, compact=True)
    assert result is marker
    assert calls == [((p, data, key), dict(nsim=9000, chunk_size=8, use_kl=False,
        stable_kl_noise=False, time_block_size=32, stop_early=True, compact=True,
        kl_bucketed=False, cumulative_kl=False))]
