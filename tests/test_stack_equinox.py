"""Ensure that the reset model is equivalent to a non-reset batched model"""
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from memorax.equinox.semigroups.stack import Stack


def test_stack():
    T = 4
    F = 2
    stack_size = 3

    def make_input(x):
        y = jnp.concatenate([
            jnp.zeros((stack_size - 1, F)),
            x[None]
        ])
        mask = jnp.concatenate([
            jnp.zeros((stack_size - 1,)),
            jnp.array([1.0])
        ])
        return y, mask


    m = Stack(recurrent_size=F, stack_size=stack_size, key=jax.random.key(0))
    inputs = (
        # x
        jnp.arange(T * F, dtype=jnp.float32).reshape((T, F)),
        # starts
        jnp.zeros((T,), dtype=bool)
    )
    h = m.initialize_carry()
    hs, ys = m(h, inputs)
    masks = hs[0][1]
    # mask == [[0,0,1], [0, 1, 1], [1, 1, 1] ... ]
    assert jnp.allclose(
        masks,
        jnp.array([
            [False, False, True],
            [False, True, True],
            [True, True, True],
            [True, True, True],
        ])
    )

if __name__ == "__main__":
    test_stack()