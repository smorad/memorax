from typing import Callable
import jax
import jax.numpy as jnp
from memorax.mtypes import RecurrentState


def magma_scan(
    magma_op: Callable[[RecurrentState, RecurrentState], RecurrentState],
    state: RecurrentState,
    input: RecurrentState,
):
    _, new_state = jax.lax.scan(
        f=magma_op,
        init=state,
        xs=input,
    )
    return new_state


def monoid_scan(
    monoid_op: Callable[[RecurrentState, RecurrentState], RecurrentState],
    state: RecurrentState,
    input: RecurrentState,
    axis: int = 0,
) -> RecurrentState:
    """Update the recurrent state using an associative scan.

    Executes a monoidal scan. The monoid_op MUST be associative, i.e.,
        f(f(e_I, a), f(b,c)) == f(f(a,b), f(c, e_I))
        where e_I is the identity element
    See https://en.wikipedia.org/wiki/Monoid for information about monoids.
    """

    # Concatenate the previous state to the inputs and scan over the result
    # This ensures the previous recurrent state contributes to the current batch
    scan_inputs = jax.tree.map(lambda s, x: jnp.concatenate([s, x], axis=axis), state, input)

    new_state = jax.lax.associative_scan(
        fn=monoid_op,
        elems=scan_inputs,
        axis=axis,
    )

    # The zeroth index corresponds to the previous recurrent state
    # We just use it to ensure continuity
    # We do not actually want to use these values, so slice them away
    return jax.tree.map(
        lambda x: jax.lax.slice_in_dim(x, start_index=1, limit_index=None, axis=axis), new_state
    )
