"""Top-level pytest config: enable JAX x64 mode before any test imports.

This runs at pytest collection time, before any `from model_a import jax_port`
statement. Idempotent with the duplicate call inside jax_port.py — both set
the same flag, second call is a no-op.

JAX is an optional dependency: the `cognitive_society` package is pure NumPy and
its tests must collect and pass without JAX installed. So the import is guarded —
if JAX is absent we simply skip the x64 setup; the model_a / model_b tests that
genuinely need JAX will fail at their own import with a clear error.
"""
try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass
