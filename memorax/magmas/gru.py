from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from equinox import filter_vmap, nn
from jaxtyping import Array, Float

from memorax.groups import BinaryAlgebra, Magma, Module, Resettable
from memorax.memoroid import Memoroid
from memorax.mtypes import Input, StartFlag
from memorax.scans import magma_scan
from memorax.utils import leaky_relu

GRURecurrentState = Float[Array, "Time Recurrent"]
GRURecurrentStateWithReset = Tuple[GRURecurrentState, StartFlag]


class GRUMagma(Magma):
    """
    The Gated Recurrent Unit Magma (recurrent update) from
    https://arxiv.org/abs/1406.1078
    """

    recurrent_size: int
    U_z: nn.Linear
    U_r: nn.Linear
    U_h: nn.Linear
    W_z: nn.Linear
    W_r: nn.Linear
    W_h: nn.Linear

    def __init__(self, recurrent_size: int, key):
        self.recurrent_size = recurrent_size
        keys = jax.random.split(key, 6)
        self.U_z = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[0]
        )
        self.U_r = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[1]
        )
        self.U_h = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[2]
        )

        self.W_z = nn.Linear(recurrent_size, recurrent_size, key=keys[3])
        self.W_r = nn.Linear(recurrent_size, recurrent_size, key=keys[4])
        self.W_h = nn.Linear(recurrent_size, recurrent_size, key=keys[5])

    def __call__(
        self, carry: GRURecurrentState, input: GRURecurrentState
    ) -> GRURecurrentState:
        z = jax.nn.sigmoid(self.W_z(input) + self.U_z(carry))
        r = jax.nn.sigmoid(self.W_r(input) + self.U_r(carry))
        h_hat = jax.nn.tanh(self.W_h(input) + self.U_h(r * carry))
        h = (1 - z) * carry + z * h_hat
        return h

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> GRURecurrentState:
        return jnp.zeros((*batch_shape, self.recurrent_size))


class GRULayer(Memoroid):
    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [GRURecurrentStateWithReset, GRURecurrentStateWithReset],
                GRURecurrentStateWithReset,
            ],
            GRURecurrentStateWithReset,
            GRURecurrentStateWithReset,
        ],
        GRURecurrentStateWithReset,
    ]

    def __init__(self, recurrent_size, hidden_size, key):
        keys = jax.random.split(key, 3)
        self.algebra = Resettable(GRUMagma(recurrent_size, key=keys[0]))
        self.scan = magma_scan

    def forward_map(self, x: Input) -> GRURecurrentStateWithReset:
        emb, start = x
        return emb, start

    def backward_map(
        self, h: GRURecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        z, reset_flag = h
        emb, start = x
        return z

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> GRURecurrentState:
        return self.algebra.initialize_carry(batch_shape)


class GRU(Module):
    layers: List[GRULayer]
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
            self.layers.append(GRULayer(hidden_size, hidden_size, key))
            self.ff.append(
                nn.Sequential(
                    [
                        nn.Linear(hidden_size, hidden_size, key=ff_key),
                        leaky_relu,
                    ]
                )
            )

    def __call__(
        self, h: GRURecurrentStateWithReset, x: Input
    ) -> Tuple[GRURecurrentStateWithReset, ...]:
        emb, start = x
        emb = filter_vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, GRU_layer, h_i in zip(self.ff, self.layers, h):
            tmp, z = GRU_layer(h_i, layer_in)
            h_out.append(tmp)
            z = filter_vmap(ff)(z)
            layer_in = (z, start)
        out = filter_vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> Tuple[GRURecurrentStateWithReset, ...]:
        return tuple(l.initialize_carry(batch_shape) for l in self.layers)
