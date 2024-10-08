from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from equinox import filter_vmap, nn
from jaxtyping import Array, Float

from memorax.groups import BinaryAlgebra, Module, Monoid, Resettable
from memorax.memoroid import Memoroid
from memorax.mtypes import Input, RecurrentState, StartFlag
from memorax.scans import monoid_scan
from memorax.utils import leaky_relu, relu

LinearRNNRecurrentState = Float[Array, "Time Hidden"]
LinearRNNRecurrentStateWithReset = Tuple[LinearRNNRecurrentState, StartFlag]


class LinearRNNMonoid(Monoid):
    recurrent_size: int

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> LinearRNNRecurrentState:
        return jnp.zeros((1, self.recurrent_size))

    def __call__(
        self, carry: LinearRNNRecurrentState, input: LinearRNNRecurrentState
    ) -> LinearRNNRecurrentState:
        return carry + input


class LinearRecurrent(Memoroid):
    """A simple Linear Recurrent layer.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [LinearRNNRecurrentStateWithReset, LinearRNNRecurrentStateWithReset],
                LinearRNNRecurrentStateWithReset,
            ],
            LinearRNNRecurrentStateWithReset,
            LinearRNNRecurrentStateWithReset,
        ],
        LinearRNNRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    project: nn.Linear

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(LinearRNNMonoid(recurrent_size))
        self.scan = monoid_scan

        keys = jax.random.split(key)

        self.project = nn.Sequential(
            [nn.Linear(recurrent_size, recurrent_size, key=keys[1]), leaky_relu]
        )

    def forward_map(self, x: Input) -> LinearRNNRecurrentStateWithReset:
        emb, start = x
        z = emb
        return z, start

    def backward_map(
        self, h: LinearRNNRecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        return self.project(state)

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> LinearRNNRecurrentStateWithReset:
        return self.algebra.initialize_carry(batch_shape)
