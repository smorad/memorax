from functools import partial
from beartype.typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Shaped

from memorax.mtypes import Input, RecurrentState, ResetRecurrentState, StartFlag
from memorax.utils import debug_shape


class Module(eqx.Module):
    r"""
    The base module for memory/sequence models.

    A module :math:`f` maps a recurrent state and inputs to an output recurrent state and outputs.
    We always include a binary start flag in the inputs.

    .. math::

        f: H \times X^n \times \{0, 1\}^n \mapsto H^n \times Y^n


    The start flag signifies the beginning of a new sequence. For example,
    .. code::

        [1, 2, 3, 4, 5]
        [0, 0, 1, 0, 1]

    Denotes that the input at 3 and 5 begin new sequences.

    """

    def __call__(self, s: RecurrentState, x: Input) -> RecurrentState:
        raise NotImplementedError

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> RecurrentState:
        raise NotImplementedError


class BinaryAlgebra(Module):
    r"""An binary algebraic structure (e.g., semigroup, set action, etc) that maps two inputs to an output.

    The inputs and output must belong to the same set

    .. math::

            f: H \times H \mapsto H
    """

    def __call__(self, carry: RecurrentState, input: RecurrentState) -> RecurrentState:
        pass

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> RecurrentState:
        raise NotImplementedError


class SetAction(BinaryAlgebra):
    r"""A set action, as defined in https://en.wikipedia.org/wiki/Magma_(algebra)

    A set action is a set :math:`H`, an action :math:`X` and an operator :math:`\bullet` that maps two inputs to an output

    .. math::

        \bullet: H \times X \mapsto H
    """

    def __call__(self, carry: RecurrentState, input: RecurrentState) -> RecurrentState:
        pass

    def initialize_carry(self,  key: Optional[Shaped[PRNGKeyArray, ""]] = None) -> RecurrentState:
        raise NotImplementedError


class Semigroup(BinaryAlgebra):
    r"""A semigroup, as defined in https://en.wikipedia.org/wiki/Semigroup.

    A semigroup is a set :math:`H` and an operator :math:`\bullet`. Unlike
    the set action, the semigroup operator must be associative.

    .. math::

        \bullet: H \times H \mapsto H

        (a \bullet b) \bullet c = a \bullet (b \bullet c)
    """

    def __call__(self, carry: RecurrentState, input: RecurrentState) -> RecurrentState:
        raise NotImplementedError

    def initialize_carry(self, key: Optional[Shaped[PRNGKeyArray, ""]] = None) -> RecurrentState:
        raise NotImplementedError


class Resettable(BinaryAlgebra):
    """A wrapper that resets the recurrent state upon beginning a new sequence.

    You can apply this to semigroups or set actions to reset the recurrent state upon a start flag.
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
