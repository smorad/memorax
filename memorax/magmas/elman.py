import jax
import jax.numpy as jnp
from equinox import nn

from typing import Callable, Tuple
from memorax.groups import BinaryAlgebra, Magma, Resettable
from memorax.memoroid import Memoroid
from jaxtyping import Array, Float

from memorax.mtypes import Input, StartFlag
from memorax.scans import magma_scan


ElmanRecurrentState = Float[Array, "Time Recurrent"]
ElmanRecurrentStateWithReset = Tuple[ElmanRecurrentState, StartFlag]


class ElmanMagma(Magma):
    recurrent_size: int
    U_h: nn.Linear

    def __init__(self, recurrent_size: int, key):
        self.recurrent_size = recurrent_size
        self.U_h = nn.Linear(recurrent_size, recurrent_size, key=key)

    def __call__(
        self, carry: ElmanRecurrentState, input: ElmanRecurrentState
    ) -> ElmanRecurrentState:
        return jax.nn.sigmoid(self.U_h(carry) + input)

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> ElmanRecurrentState:
        return jnp.zeros((*batch_shape, self.recurrent_size))


# TODO: Make this inherit RNN or memoroid
class Elman(Memoroid):
    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [ElmanRecurrentStateWithReset, ElmanRecurrentStateWithReset],
                ElmanRecurrentStateWithReset,
            ],
            ElmanRecurrentStateWithReset,
            ElmanRecurrentStateWithReset,
        ],
        ElmanRecurrentStateWithReset,
    ]
    W_h: nn.Linear
    W_y: nn.Linear

    def __init__(self, recurrent_size, hidden_size, key):
        keys = jax.random.split(key, 3)
        self.algebra = Resettable(ElmanMagma(recurrent_size, key=keys[0]))
        self.scan = magma_scan
        self.W_h = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[1])
        self.W_y = nn.Linear(recurrent_size, hidden_size, key=keys[2])

    def forward_map(self, x: Input) -> ElmanRecurrentStateWithReset:
        emb, start = x
        return self.W_h(emb), start

    def backward_map(
        self, h: ElmanRecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        z, reset_flag = h
        emb, start = x
        return self.W_y(z)

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> ElmanRecurrentState:
        return self.algebra.initialize_carry(batch_shape)
