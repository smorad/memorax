from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from equinox import filter_vmap, nn
from jaxtyping import Array, Float

from memorax.groups import BinaryAlgebra, Module, Monoid, Resettable
from memorax.memoroid import Memoroid
from memorax.mtypes import Input, StartFlag
from memorax.scans import monoid_scan
from memorax.utils import leaky_relu, relu

FARTRecurrentState = Tuple[Float[Array, "Time Key Value"], Float[Array, "Time Key"]]
FARTRecurrentStateWithReset = Tuple[FARTRecurrentState, StartFlag]


def phi(x, key=None):
    return 1 + jax.nn.elu(x)


class FARTMonoid(Monoid):
    """The Fast AutoRegressive Transformer monoid (recurrent update) from https://arxiv.org/abs/2006.16236"""

    key_size: int
    value_size: int

    def __init__(self, key_size, value_size):
        self.key_size = key_size
        self.value_size = value_size

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> FARTRecurrentState:
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


class FARTLayer(Memoroid):
    """The Fast AutoRegressive Transformer from https://arxiv.org/abs/2006.16236.

    You might want to use this as a building block for a more complex model.
    """

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

    def __init__(self, hidden_size, key_size, key):
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.algebra = Resettable(FARTMonoid(key_size, hidden_size))
        self.scan = monoid_scan

        keys = jax.random.split(key, 6)

        self.K = nn.Linear(hidden_size, key_size, use_bias=False, key=keys[0])
        self.Q = nn.Linear(hidden_size, key_size, use_bias=False, key=keys[1])
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


class FART(Module):
    layers: List[FARTLayer]
    ff: List[nn.Sequential]
    map_in: nn.Linear
    map_out: nn.Linear

    def __init__(self, input_size, output_size, hidden_size, num_layers, key):
        self.layers = []
        self.ff = []
        self.map_in = nn.Linear(input_size, hidden_size, key=key)
        self.map_out = nn.Linear(hidden_size, output_size, key=key)
        for _ in range(num_layers):
            key, ff_key = jax.random.split(key)
            self.layers.append(FARTLayer(hidden_size, hidden_size, key))
            self.ff.append(
                nn.Sequential(
                    [
                        nn.Linear(hidden_size, hidden_size, key=ff_key),
                        leaky_relu,
                    ]
                )
            )

    def __call__(
        self, h: FARTRecurrentStateWithReset, x: Input
    ) -> FARTRecurrentStateWithReset:
        emb, start = x
        emb = filter_vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, FART_layer, h_i in zip(self.ff, self.layers, h):
            tmp, z = FART_layer(h_i, layer_in)
            h_out.append(tmp)
            z = filter_vmap(ff)(z)
            layer_in = (z, start)
        out = filter_vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> Tuple[FARTRecurrentStateWithReset, ...]:
        return tuple(l.initialize_carry(batch_shape) for l in self.layers)
