from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import Array, Float

from memorax.groups import BinaryAlgebra, Magma, Resettable
from memorax.memoroid import Memoroid
from memorax.mtypes import Input, StartFlag
from memorax.scans import magma_scan

MGURecurrentState = Float[Array, "Time Recurrent"]
MGURecurrentStateWithReset = Tuple[MGURecurrentState, StartFlag]


class MGUMagma(Magma):
    """
    The Minimal Gated Unit Magma (recurrent update) from
    https://arxiv.org/abs/1701.03452
    """

    recurrent_size: int
    U_h: nn.Linear
    U_f: nn.Linear
    W_h: nn.Linear
    W_f: nn.Linear

    def __init__(self, recurrent_size: int, key):
        self.recurrent_size = recurrent_size
        keys = jax.random.split(key, 4)
        self.U_h = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[0]
        )
        self.U_f = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[1]
        )
        self.W_h = nn.Linear(recurrent_size, recurrent_size, key=keys[2])
        self.W_f = nn.Linear(recurrent_size, recurrent_size, key=keys[3])

    def __call__(
        self, carry: MGURecurrentState, input: MGURecurrentState
    ) -> MGURecurrentState:
        f = jax.nn.sigmoid(self.W_f(input) + self.U_f(carry))
        h_hat = jax.nn.tanh(self.W_h(input) + self.U_h(f * carry))
        h = (1 - f) * carry + f * h_hat
        return h

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> MGURecurrentState:
        return jnp.zeros((*batch_shape, self.recurrent_size))


class MGU(Memoroid):
    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [MGURecurrentStateWithReset, MGURecurrentStateWithReset],
                MGURecurrentStateWithReset,
            ],
            MGURecurrentStateWithReset,
            MGURecurrentStateWithReset,
        ],
        MGURecurrentStateWithReset,
    ]

    def __init__(self, recurrent_size, key):
        keys = jax.random.split(key, 3)
        self.algebra = Resettable(MGUMagma(recurrent_size, key=keys[0]))
        self.scan = magma_scan

    def forward_map(self, x: Input) -> MGURecurrentStateWithReset:
        emb, start = x
        return emb, start

    def backward_map(
        self, h: MGURecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        z, reset_flag = h
        emb, start = x
        return z

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> MGURecurrentState:
        return self.algebra.initialize_carry(batch_shape)
