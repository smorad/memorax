
"""This module implements scans for executing recurrent models in Equinox.

They wrap JAX's scan and associative_scan functions, creating a unified interface
for executing recurrent models defined as binary algebras (set actions and semigroups).
It also provides some useful error checking to ensure the scans are used correctly.
"""
from beartype.typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from memax.mtypes import RecurrentState
from memax.utils import debug_shape


def set_action_scan(
    set_action_op: Callable[[RecurrentState, RecurrentState], RecurrentState],
    state: RecurrentState,
    input: RecurrentState,
):
    """Update the recurrent state using a sequential scan.

    This works with virtually any recurrent model. You can even use it with semigroups,
    it will just be less efficient than an associative scan.
    """
    assert jax.tree.structure(state) == jax.tree.structure(
        input
    ), f"Mismatched structures passed to scan, {jax.tree.structure(state)} and {jax.tree.structure(input)}"

    def wrapped_set_action_op(set_action, carry, xs):
        out = set_action(carry, xs)
        return out, out  # Ensure the output carry matches the input structure

    _, new_state = nn.scan(
        target=wrapped_set_action_op,
        in_axes=0,
        variable_broadcast="params",
        split_rngs={"params": False},
    )(set_action_op, state, input)
    assert jax.tree.structure(new_state) == jax.tree.structure(
        input
    ), f"Mismatched structures returned from scan, {jax.tree.structure(input)} and {jax.tree.structure(new_state)}"
    return new_state


def semigroup_scan(
    semigroup_op: Callable[[RecurrentState, RecurrentState], RecurrentState],
    state: RecurrentState,
    input: RecurrentState,
    axis: int = 0,
) -> RecurrentState:
    r"""Update the recurrent state using an associative scan.

    The semigroup_op MUST be associative, i.e.,

    $f(a, f(b,c)) = f(f(a,b), c)$

    See https://en.wikipedia.org/wiki/Semigroup for information.
    """
    assert jax.tree.structure(state) == jax.tree.structure(
        input
    ), f"Mismatched structures passed to scan, {jax.tree.structure(state)} and {jax.tree.structure(input)}"

    # Both scan initial state and inputs must have the same numbers of dims
    # Usually, we have h0 = (features,) and x = (time, features)
    # Expand h0 = (1, features)
    singleton_state = jax.tree.map(lambda s: jnp.expand_dims(s, axis=axis), state)
    # Concatenate the previous state to the inputs and scan over the result
    # This ensures the previous recurrent state contributes to the current batch
    scan_inputs = jax.tree.map(
        lambda s, x: jnp.concatenate([s, x], axis=axis), singleton_state, input
    )

    new_state = jax.lax.associative_scan(
        fn=jax.vmap(semigroup_op), elems=scan_inputs, axis=axis
    )

    # The zeroth index corresponds to the previous recurrent state
    # We just use it to ensure continuity
    # We do not actually want to use these values, so slice them away
    new_state = jax.tree.map(
        lambda x: jax.lax.slice_in_dim(x, start_index=1, limit_index=None, axis=axis),
        new_state,
    )

    assert jax.tree.structure(new_state) == jax.tree.structure(
        input
    ), f"Mismatched structures returned from scan, {jax.tree.structure(input)} and {jax.tree.structure(new_state)}"
    assert all(
        jax.tree.leaves(jax.tree.map(lambda x, y: x.shape == y.shape, input, new_state))
    ), f"Shapes do not match {debug_shape(input)} and {debug_shape(new_state)}"

    return new_state
