from functools import partial
from beartype.typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Shaped

from memax.mtypes import Input, RecurrentState, ResetRecurrentState, StartFlag
from memax.utils import debug_shape


class Module(eqx.Module):
    r"""
    The base module for memory/sequence models.

    A module $m$ maps a recurrent state and inputs to an output recurrent state and outputs.
    We always include a binary start flag in the inputs.

    $ m: H \times X^n \times \\{0, 1\\}^n \mapsto H^n \times Y^n $


    The start flag signifies the beginning of a new sequence. For example,

        [1, 2, 3, 4, 5]
        [0, 0, 1, 0, 1]

    Denotes three distinct input sequences: [1, 2], [3, 4], and [5]. With `Resettable`, these sequences
    will not interfere with each other. As such, you can concatenate multiple sequences
    into a single sequence for efficient processing.
    """

    def __call__(self, h: RecurrentState, x: Input) -> RecurrentState:
        r"""Applies $\bullet$ to the inputs to return an updated recurrent state."""
        raise NotImplementedError

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> RecurrentState:
        r"""Returns the initial recurrent state $h_0$."""
        raise NotImplementedError


class BinaryAlgebra(Module):
    r"""An binary algebraic structure (e.g., semigroup, set action) that maps two inputs to an output.

    You must define an initial state $h_0$ and a binary operator $\bullet$.
    """

    def __call__(self, carry: RecurrentState, input: RecurrentState) -> RecurrentState:
        pass

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> RecurrentState:
        raise NotImplementedError


class SetAction(BinaryAlgebra):
    r"""
    A set action, a form of binary algebra that we execute using scans.

    $ \bullet: H \times Z \mapsto H $.
    """

    def __call__(self, carry: RecurrentState, input: RecurrentState) -> RecurrentState:
        pass

    def initialize_carry(self,  key: Optional[Shaped[PRNGKeyArray, ""]] = None) -> RecurrentState:
        raise NotImplementedError


class Semigroup(BinaryAlgebra):
    r"""A semigroup, as defined in https://en.wikipedia.org/wiki/Semigroup.

    A semigroup is a constrained form of set action, where:
    1. The action and recurrent state spaces are identical, i.e., $Z = H$.
    2. The binary operator $\bullet$ is associative.

    $ \bullet: H \times H \mapsto H $

    $ (a \bullet b) \bullet c = a \bullet (b \bullet c) $.
    """

    def __call__(self, carry: RecurrentState, input: RecurrentState) -> RecurrentState:
        raise NotImplementedError

    def initialize_carry(self, key: Optional[Shaped[PRNGKeyArray, ""]] = None) -> RecurrentState:
        raise NotImplementedError


class Resettable(BinaryAlgebra):
    r"""A wrapper that resets the recurrent state upon beginning a new sequence.

    This is a binary algebra defined on another binary algebra.

    For set actions this is

    $ \circ: (H \times \\{0, 1\\}) \times (Z \times \\{0, 1\\}) \mapsto (H \times \\{0, 1\\}) $

    while for semigroups this is

    $ \circ: (H \times \\{0, 1\\}) \times (H \times \\{0, 1\\}) \mapsto (H \times \\{0, 1\\}) $.
    """

    algebra: BinaryAlgebra

    def __init__(self, algebra: BinaryAlgebra):
        self.algebra = algebra

    def __call__(self, carry: ResetRecurrentState, input: ResetRecurrentState):
        assert jax.tree.structure(carry) == jax.tree.structure(
            input
        ), f"Mismatched structures passed to algebra, {jax.tree.structure(carry)} and {jax.tree.structure(input)}"
        states, prev_carry_reset_flag = carry
        xs, start = input

        def reset_state(
            start: StartFlag,
            current_state: RecurrentState,
            initial_state: RecurrentState,
        ):
            assert initial_state.ndim == current_state.ndim
            out = current_state * jnp.logical_not(start) + initial_state * start
            return out

        # TODO: Plumb key thru
        initial_states = self.algebra.initialize_carry(None) 
        states = jax.tree.map(
            partial(reset_state, start), states, initial_states)
        out = self.algebra(states, xs)
        carry_reset_flag = jnp.logical_or(start, prev_carry_reset_flag)
        to_return = out, carry_reset_flag
        assert jax.tree.structure(carry) == jax.tree.structure(
            to_return
        ), f"Mismatched structures passed from algebra,\n{jax.tree.structure(carry)} and\n{jax.tree.structure(out)}"
        assert all(
            jax.tree.leaves(jax.tree.map(lambda x, y: x.shape == y.shape, states, out))
        ), f"Shapes do not match\n{debug_shape(states)} and\n{debug_shape(out)}"

        return to_return

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> RecurrentState:
        return self.algebra.initialize_carry(key), jnp.zeros((), dtype=bool)
