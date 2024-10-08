from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import Array, Float

from memorax.groups import BinaryAlgebra, Monoid, Resettable
from memorax.memoroid import Memoroid
from memorax.mtypes import Input, StartFlag
from memorax.scans import monoid_scan

FARTRecurrentState = Tuple[Float[Array, "Time Key Value"], Float[Array, "Time Key"]]
FARTRecurrentStateWithReset = Tuple[FARTRecurrentState, StartFlag]


def phi(x, key=None):
    return 1 + jax.nn.elu(x)


class FARTMonoid(Monoid):
    """The Fast AutoRegressive Transformer monoid (recurrent update) from https://arxiv.org/abs/2006.16236"""

    recurrent_size: int
    value_size: int

    def __init__(self, recurrent_size, value_size):
        self.recurrent_size = recurrent_size
        self.value_size = value_size

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> FARTRecurrentState:
        return (
            jnp.zeros((*batch_shape, 1, self.recurrent_size, self.value_size)),
            jnp.zeros((*batch_shape, 1, self.recurrent_size)),
        )

    def __call__(
        self, carry: FARTRecurrentState, input: FARTRecurrentState
    ) -> FARTRecurrentState:
        (
            kv_sum,
            k_sum,
        ) = carry
        kv, k = input
        kv_sum = kv_sum + kv
        k_sum = k_sum + k
        return kv_sum, k


class FART(Memoroid):
    """The Fast AutoRegressive Transformer from https://arxiv.org/abs/2006.16236.

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int
    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [FARTRecurrentStateWithReset, FARTRecurrentStateWithReset],
                FARTRecurrentStateWithReset,
            ],
            FARTRecurrentStateWithReset,
            FARTRecurrentStateWithReset,
        ],
        FARTRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    K: nn.Linear
    Q: nn.Linear
    V: nn.Linear

    def __init__(self, hidden_size, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.algebra = Resettable(FARTMonoid(recurrent_size, hidden_size))
        self.scan = monoid_scan

        keys = jax.random.split(key, 6)

        self.K = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[0])
        self.Q = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[1])
        self.V = nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[2])

    def forward_map(self, x: Input) -> FARTRecurrentStateWithReset:
        emb, start = x
        k = phi(self.K(emb))
        v: Float[Array, "Time Feat"] = self.V(emb)
        kv = jnp.outer(k, v)
        return (kv, k), start

    def backward_map(
        self, h: FARTRecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        (kv_sum, k_sum), reset_flag = h
        q = phi(self.Q(emb))
        out = q @ kv_sum / (1e-6 + jnp.dot(k_sum, q))
        return out + emb
        # return self.ff(out + emb)

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> FARTRecurrentStateWithReset:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return self.algebra.initialize_carry(batch_shape)
