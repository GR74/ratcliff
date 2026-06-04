"""Top-level pytest config: enable JAX x64 mode before any test imports.

This runs at pytest collection time, before any `from model_a import jax_port`
statement. Idempotent with the duplicate call inside jax_port.py — both set
the same flag, second call is a no-op.
"""
import jax

jax.config.update("jax_enable_x64", True)
