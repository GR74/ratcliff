"""Top-level pytest config: enable JAX x64 mode before any test imports.

This runs at pytest collection time, before any `from model_a import jax_port`
statement. Idempotent with the duplicate call inside jax_port.py — both set
the same flag, second call is a no-op.

The import is guarded so parser and job-management tests can collect without
the scientific runtime installed. Model A / Model B tests require JAX and will
report a missing dependency at their own import.
"""
try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass
