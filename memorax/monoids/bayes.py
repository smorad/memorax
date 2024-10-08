from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import Array, Float

from memorax.groups import BinaryAlgebra, Monoid, Resettable
from memorax.memoroid import Memoroid
from memorax.mtypes import Input, StartFlag
from memorax.scans import monoid_scan

LogBayesRecurrentState = Float[Array, "Time Hidden"]
LogBayesRecurrentStateWithReset = Tuple[LogBayesRecurrentState, StartFlag]


class LogBayesMonoid(Monoid):
    recurrent_size: int

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> LogBayesRecurrentState:
        return jnp.ones((*batch_shape, 1, self.recurrent_size)) * -jnp.log(
            self.recurrent_size
        )

    def __call__(
        self, carry: LogBayesRecurrentState, input: LogBayesRecurrentState
    ) -> LogBayesRecurrentState:
        return carry + input  # In log space, equivalent to carry @ input


class LogBayes(Memoroid):
    """A simple Bayesian memory layer.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [LogBayesRecurrentStateWithReset, LogBayesRecurrentStateWithReset],
                LogBayesRecurrentStateWithReset,
            ],
            LogBayesRecurrentStateWithReset,
            LogBayesRecurrentStateWithReset,
        ],
        LogBayesRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    project: nn.Linear

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(LogBayesMonoid(recurrent_size))
        self.scan = monoid_scan

        keys = jax.random.split(key)

        self.project = nn.Linear(recurrent_size, recurrent_size, key=keys[0])

    def forward_map(self, x: Input) -> LogBayesRecurrentStateWithReset:
        emb, start = x
        z = self.project(emb)
        return z, start

    def backward_map(
        self, h: LogBayesRecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        out = jax.nn.softmax(state, axis=-1)
        return out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> LogBayesRecurrentStateWithReset:
        return self.algebra.initialize_carry(batch_shape)
