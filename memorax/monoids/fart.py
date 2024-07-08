from typing import Callable, Tuple
from memorax.groups import BinaryAlgebra, Monoid, Resettable
from memorax.memoroid import Memoroid
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
from equinox import nn
from memorax.scans import monoid_scan
from memorax.utils import relu

from memorax.mtypes import Input, StartFlag


FARTRecurrentState = Tuple[Float[Array, "Time Key Value"], Float[Array, "Time Key"]]
FARTRecurrentStateWithReset = Tuple[FARTRecurrentState, StartFlag]


def phi(x, key=None):
    return 1 + jax.nn.elu(x)


class FARTMonoid(Monoid):
    key_size: int
    value_size: int

    def __init__(self, key_size, value_size):
        self.key_size = key_size
        self.value_size = value_size

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> FARTRecurrentState:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return (
            jnp.zeros((*batch_shape, 1, self.key_size, self.value_size)),
            jnp.zeros((*batch_shape, 1, self.key_size)),
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
    hidden_size: int
    key_size: int
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
    ff: nn.Sequential

    def __init__(self, hidden_size, key_size, key):
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.algebra = Resettable(FARTMonoid(key_size, hidden_size))
        self.scan = monoid_scan

        keys = jax.random.split(key, 6)

        self.K = nn.Linear(hidden_size, key_size, use_bias=False, key=keys[0])
        self.Q = nn.Linear(hidden_size, key_size, use_bias=False, key=keys[1])
        self.V = nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[2])
        self.ff = nn.Sequential(
            [
                nn.Linear(hidden_size, hidden_size, key=keys[3]),
                relu,
                nn.Linear(hidden_size, hidden_size, key=keys[4]),
                relu,
                nn.Linear(hidden_size, hidden_size, key=keys[5]),
            ]
        )

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
        (kv_sum, k_sum), start = h
        q = phi(self.Q(emb))
        out = kv_sum / (1e-6 + jnp.dot(k_sum, q))
        return out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> FARTRecurrentStateWithReset:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return self.algebra.initialize_carry(batch_shape)
