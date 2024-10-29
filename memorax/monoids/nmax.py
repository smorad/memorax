from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import Array, Float

from memorax.groups import BinaryAlgebra, Monoid, Resettable
from memorax.memoroid import Memoroid
from memorax.mtypes import Input, StartFlag
from memorax.scans import monoid_scan

NMaxRecurrentState = Float[Array, "Time Hidden"]
NMaxRecurrentStateWithReset = Tuple[NMaxRecurrentState, StartFlag]


class NMaxMonoid(Monoid):
    recurrent_size: int

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> NMaxRecurrentState:
        return jnp.zeros((1, self.recurrent_size))

    def __call__(
        self, carry: NMaxRecurrentState, input: NMaxRecurrentState
    ) -> NMaxRecurrentState:
        return jnp.maximum(carry, input)


class NMax(Memoroid):
    """A Gated Impulse Linear Recurrent layer.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [NMaxRecurrentStateWithReset, NMaxRecurrentStateWithReset],
                NMaxRecurrentStateWithReset,
            ],
            NMaxRecurrentStateWithReset,
            NMaxRecurrentStateWithReset,
        ],
        NMaxRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    g: nn.Sequential
    i: nn.Sequential

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(NMaxMonoid(recurrent_size))
        self.scan = monoid_scan

        keys = jax.random.split(key)

        self.g = nn.Sequential(
            [
                nn.Linear(recurrent_size, recurrent_size, key=keys[1]),
                nn.Lambda(jax.nn.sigmoid),
            ]
        )
        self.i = nn.Sequential(
            [
                nn.Linear(recurrent_size, recurrent_size, key=keys[1]),
                nn.Lambda(jax.nn.leaky_relu),
            ]
        )

    def forward_map(self, x: Input) -> NMaxRecurrentStateWithReset:
        emb, start = x
        z = emb
        return z, start

    def backward_map(
        self, h: NMaxRecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        z = state / jnp.linalg.norm(state, ord=1)
        g = self.g(emb)
        return g * state + (1 - g) * emb

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> NMaxRecurrentStateWithReset:
        return self.algebra.initialize_carry(batch_shape)
