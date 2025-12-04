"""Ensure that the reset model is equivalent to a non-reset batched model"""
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from memax.equinox.train_utils import add_batch_dim, get_residual_memory_models

@pytest.mark.parametrize("name, model", get_residual_memory_models(
        input=1, hidden=8, output=10, num_layers=2, key=jax.random.key(0)
    ).items()
)
def test_reset(name, model):
    print(f"Testing {name}")
    L = 2048
    B = 32
    x_long = jax.random.uniform(jax.random.key(1), (L, 1))
    x_short = x_long.reshape(B, -1, 1)
    start_long = jnp.zeros(L, dtype=bool).at[jnp.arange(0, L, B)].set(True)
    start_short = start_long.reshape(B, -1)

    h0_long = model.initialize_carry()
    h_long, out_long = model(h0_long, (x_long, start_long))
    h0_short = add_batch_dim(h0_long, B)
    h_short, out_short = eqx.filter_vmap(model)(h0_short, (x_short, start_short))

    assert jnp.allclose(
        out_long,
        out_short.reshape(L, *out_short.shape[2:]),
        atol=1e-5,
        rtol=1e-5,
    ), f"Reset failed for {name}"

if __name__ == "__main__":
    test_reset()