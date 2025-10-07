"""Proves the correctness (empirically) of associative updates"""
from functools import partial

import jax
import jax.numpy as jnp

from memorax.flax.groups import Semigroup
from memorax.flax.train_utils import get_semigroups


def random_state(state, key):
    if state.dtype in [jnp.float32, jnp.complex64]:
        return jax.random.normal(key, state.shape, dtype=state.dtype)
    elif state.dtype in [jnp.int32]:
        return jax.random.randint(key, state.shape, 0, 5, dtype=state.dtype)
    elif state.dtype in [jnp.bool_]:
        return jax.random.bernoulli(key, 0.5, state.shape, dtype=state.dtype)
    else:
        raise NotImplementedError(
            f"Random state not implemented for dtype {state.dtype}"
        )


def map_assert(monoid, a, b):
    is_equal = jnp.allclose(a, b)
    error = jnp.abs(a - b)
    if not is_equal:
        raise Exception(
            f"Monoid {type(monoid).__name__} failed associativity test:\n{a} != \n{b}, \nerror: {error}"
        )

def prove_semigroup_correctness(sg: Semigroup):
    initial_state = sg.initialize_carry()
    x1 = jax.tree.map(partial(random_state, key=jax.random.PRNGKey(1)), initial_state)
    x2 = jax.tree.map(partial(random_state, key=jax.random.PRNGKey(2)), initial_state)
    x3 = jax.tree.map(partial(random_state, key=jax.random.PRNGKey(3)), initial_state)

    a = sg(sg(x1, x2), x3)
    b = sg(x1, sg(x2, x3))

    is_equal = jax.tree.map(jnp.allclose, a, b)
    if isinstance(is_equal, tuple):
        is_equal = all(is_equal)
    else:
        is_equal = jnp.all(is_equal)

    jax.tree.map(partial(map_assert, sg), a, b)


def test_semigroup_correctness():
    for name, sg in get_semigroups(recurrent_size=3).items():
        prove_semigroup_correctness(sg)

if __name__ == '__main__':
    test_semigroup_correctness()
