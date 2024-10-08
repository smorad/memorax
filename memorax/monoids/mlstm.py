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

MLSTMRecurrentState = Tuple[
    Float[Array, "Time Hidden Hidden"],
    Float[Array, "Time Hidden"],
    Float[Array, "Time Hidden"],
]
MLSTMRecurrentStateWithReset = Tuple[MLSTMRecurrentState, StartFlag]


class MLSTMMonoid(Monoid):
    recurrent_size: int
    f: nn.Sequential

    def __init__(self, recurrent_size: int, key):
        self.recurrent_size = recurrent_size
        self.f = filter_vmap(
            nn.Sequential(
                [
                    nn.Linear(recurrent_size, 1, key=key),
                    nn.Lambda(jax.nn.sigmoid),
                ]
            )
        )

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> MLSTMRecurrentState:
        return (
            jnp.zeros((*batch_shape, 1, self.recurrent_size, self.recurrent_size)),
            jnp.zeros((*batch_shape, 1, self.recurrent_size)),
            jnp.zeros((*batch_shape, 1, self.recurrent_size)),
        )

    def __call__(
        self, carry: MLSTMRecurrentState, input: MLSTMRecurrentState
    ) -> MLSTMRecurrentState:
        prev_C, prev_n, prev_x = carry
        C, n, x = input
        return (
            self.f(x).reshape(-1, 1, 1) * prev_C + C,
            self.f(x).reshape(-1, 1) * prev_n + n,
            x,
        )


class MLSTMLayer(Memoroid):
    """The mLSTM layer from the xLSTM paper

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int
    scan: Callable[
        [
            Callable[
                [MLSTMRecurrentStateWithReset, MLSTMRecurrentStateWithReset],
                MLSTMRecurrentStateWithReset,
            ],
            MLSTMRecurrentStateWithReset,
            MLSTMRecurrentStateWithReset,
        ],
        MLSTMRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    i: nn.Sequential
    v: nn.Linear
    q: nn.Linear
    k: nn.Sequential
    o: nn.Sequential

    def __init__(self, hidden_size, key):
        self.hidden_size = hidden_size
        keys = jax.random.split(key, 6)
        self.algebra = Resettable(MLSTMMonoid(hidden_size, key=keys[0]))
        self.scan = monoid_scan

        self.i = nn.Sequential(
            [nn.Linear(hidden_size, 1, key=keys[1]), nn.Lambda(lambda x: jnp.exp(x))]
        )
        self.v = nn.Linear(hidden_size, hidden_size, key=keys[2])
        self.q = nn.Linear(hidden_size, hidden_size, key=keys[3])
        self.k = nn.Sequential(
            [
                nn.Lambda(lambda x: 1 / jnp.sqrt(hidden_size) * x),
                nn.Linear(hidden_size, hidden_size, key=keys[4]),
            ]
        )
        self.o = nn.Sequential(
            [
                nn.Linear(hidden_size, hidden_size, key=keys[5]),
                nn.Lambda(jax.nn.sigmoid),
            ]
        )

    def forward_map(self, x: Input) -> MLSTMRecurrentStateWithReset:
        emb, start = x

        i = self.i(emb)
        v = self.v(emb)
        k = self.k(emb)

        C = i * jnp.outer(v, k)
        n = i * k

        return (C, n, emb), start

    def backward_map(
        self, h: MLSTMRecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        state, reset_carry = h
        C, n, _ = state
        q = self.q(emb)
        o = self.o(emb)
        h_tilde = C @ q / jnp.maximum(jnp.abs(jnp.dot(n, q)), 1.0)
        h = o * h_tilde
        return h

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> MLSTMRecurrentStateWithReset:
        return self.algebra.initialize_carry(batch_shape)


class MLSTM(Module):
    layers: List[MLSTMLayer]
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
            self.layers.append(MLSTMLayer(hidden_size, key))
            self.ff.append(
                nn.Sequential(
                    [nn.Linear(2 * hidden_size, hidden_size, key=ff_key), leaky_relu]
                )
            )

    def __call__(
        self, h: MLSTMRecurrentStateWithReset, x: Input
    ) -> Tuple[MLSTMRecurrentStateWithReset, ...]:
        emb, start = x
        emb = filter_vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, MLSTM_layer, h_i in zip(self.ff, self.layers, h):
            tmp, z = MLSTM_layer(h_i, layer_in)
            h_out.append(tmp)
            z = filter_vmap(ff)(jnp.concatenate([z, emb], axis=-1))
            layer_in = (z, start)
        out = filter_vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> Tuple[MLSTMRecurrentStateWithReset, ...]:
        return tuple(l.initialize_carry(batch_shape) for l in self.layers)
