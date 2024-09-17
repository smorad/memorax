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


class MGULayer(Memoroid):
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

    def __init__(self, recurrent_size, hidden_size, key):
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


class MGU(Module):
    layers: List[MGULayer]
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
            self.layers.append(MGULayer(hidden_size, hidden_size, key))
            self.ff.append(
                nn.Sequential(
                    [
                        nn.Linear(hidden_size, hidden_size, key=ff_key),
                        leaky_relu,
                    ]
                )
            )

    def __call__(
        self, h: MGURecurrentStateWithReset, x: Input
    ) -> Tuple[MGURecurrentStateWithReset, ...]:
        emb, start = x
        emb = filter_vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, MGU_layer, h_i in zip(self.ff, self.layers, h):
            tmp, z = MGU_layer(h_i, layer_in)
            h_out.append(tmp)
            z = filter_vmap(ff)(z)
            layer_in = (z, start)
        out = filter_vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> Tuple[MGURecurrentStateWithReset, ...]:
        return tuple(l.initialize_carry(batch_shape) for l in self.layers)
