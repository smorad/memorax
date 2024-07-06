from typing import Tuple
import jax
from functools import partial

from jaxtyping import jaxtyped
from memorax.mtypes import ResetRecurrentState, StartFlag, Input, RecurrentState
import equinox as eqx
import jax.numpy as jnp


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


class Magma(Module):
    r"""A magma, as defined in https://en.wikipedia.org/wiki/Magma_(algebra)

    A Magma is a set :math:`H` and an operator :math:`\bullet` that maps two inputs to an output

    .. math::

        \bullet: H \times H \mapsto H
    """

    def __call__(self, carry: RecurrentState, input: RecurrentState) -> RecurrentState:
        pass

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> RecurrentState:
        raise NotImplementedError


class ResettableMagma(Magma):
    """A magma wrapper that resets the recurrent state upon beginning a new sequence"""

    magma: Magma

    def __init__(self, magma: Magma):
        self.magma = magma

    def __call__(self, carry: ResetRecurrentState, input: ResetRecurrentState):
        states, prev_carry_reset_flag = carry
        xs, start = input

        def reset_state(
            start: StartFlag, current_state: RecurrentState, initial_state: RecurrentState
        ):
            # Expand to reset all dims of state: [1, B, 1, ...]
            assert initial_state.ndim == current_state.ndim
            expanded_start = start.reshape(-1, start.shape[1], *([1] * (current_state.ndim - 2)))
            out = current_state * jnp.logical_not(expanded_start) + initial_state
            return out

        # Add an extra dim, as start will be [Batch] while intialize carry expects [Batch, Feature]
        initial_states = self.magma.initialize_carry(batch_shape=start.shape[:-1])
        states = jax.tree.map(partial(reset_state, start), states, initial_states)
        out = self.magma(states, xs)
        carry_reset_flag = jnp.logical_or(start, prev_carry_reset_flag)

        return out, carry_reset_flag

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> RecurrentState:
        return self.magma.initialize_carry(batch_shape), jnp.zeros((1, batch_shape), dtype=bool)


class Monoid(Module):
    r"""A monoid, as defined in https://en.wikipedia.org/wiki/Monoid

    A monoid is a set :math:`H`, an operator :math:`\bullet`, and an identity element :math:`e_I`. Unlike
    the Magma, the monoid operator must be associative.

    .. math::

        \bullet: H \times H \mapsto H

        e_I \in H

        (a \bullet b) \bullet c = a \bullet (b \bullet c)

        a \bullet e_I = e_I \bullet a = a
    """

    def __call__(self, carry: RecurrentState, input: RecurrentState) -> RecurrentState:
        raise NotImplementedError

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> RecurrentState:
        raise NotImplementedError


class ResettableMonoid(Monoid):
    """A monoid wrapper that resets the recurrent state upon beginning a new sequence"""

    monoid: Monoid

    def __init__(self, monoid: Monoid):
        self.monoid = monoid

    def __call__(self, carry: ResetRecurrentState, input: ResetRecurrentState):
        states, prev_carry_reset_flag = carry
        xs, start = input

        def reset_state(
            start: StartFlag, current_state: RecurrentState, initial_state: RecurrentState
        ):
            # Expand to reset all dims of state: [1, B, 1, ...]
            assert initial_state.ndim == current_state.ndim
            expanded_start = start.reshape(-1, *([1] * (current_state.ndim - 1)))
            out = current_state * jnp.logical_not(expanded_start) + initial_state
            return out

        # Add an extra dim, as start will be [Batch] while intialize carry expects [Batch, Feature]
        initial_states = self.monoid.initialize_carry(batch_shape=start.shape[:-1])
        states = jax.tree.map(partial(reset_state, start), states, initial_states)
        out = self.monoid(states, xs)
        carry_reset_flag = jnp.logical_or(start, prev_carry_reset_flag)

        return out, carry_reset_flag

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> RecurrentState:
        return self.monoid.initialize_carry(batch_shape), jnp.zeros((*batch_shape, 1), dtype=bool)
