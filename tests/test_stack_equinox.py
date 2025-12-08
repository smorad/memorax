"""Tests the framestacking method works with associative scan"""
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from memax.equinox.semigroups.stack import Stack


def test_stack():
    T = 4
    F = 2
    stack_size = 3

    m = Stack(recurrent_size=F, window_size=stack_size, key=jax.random.key(0))
    x = jnp.arange(T * F, dtype=jnp.float32).reshape((T, F))
    starts = jnp.zeros((T,), dtype=bool)
    inputs = (x, starts)

    h = m.initialize_carry()
    hs, ys = m(h, inputs)
    masks = hs[0][1]
    states = hs[0][0]
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
    # states = [ [0, 0, x[0]], [0, x[0], x[1]], ...]
    zero = jnp.zeros_like(x[0])
    tgt = jnp.array([
        [zero, zero, x[0]],
        [zero, x[0], x[1]],
        [x[0], x[1], x[2]],
        [x[1], x[2], x[3]],
    ])
    assert jnp.allclose(
        states,
        tgt
    ), f"{tgt}!=\n{states}"

if __name__ == "__main__":
    test_stack()