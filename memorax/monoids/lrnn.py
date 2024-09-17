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
        # return jnp.zeros((*batch_shape, 1, self.recurrent_size, self.recurrent_size))
        return jnp.zeros((1, self.recurrent_size))

    def __call__(
        self, carry: LinearRNNRecurrentState, input: LinearRNNRecurrentState
    ) -> LinearRNNRecurrentState:
        return carry + input  # In log space, equivalent to carry @ input


class LinearRNNLayer(Memoroid):
    """A simple Linear Recurrent Model.

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int
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

    def __init__(self, hidden_size, key):
        self.hidden_size = hidden_size
        self.algebra = Resettable(LinearRNNMonoid(hidden_size))
        self.scan = monoid_scan

        keys = jax.random.split(key)

        self.project = nn.Sequential(
            [nn.Linear(hidden_size, hidden_size, key=keys[1]), leaky_relu]
        )

    def forward_map(self, x: Input) -> LinearRNNRecurrentStateWithReset:
        emb, start = x
        z = emb
        return z, start

    def backward_map(
        self, h: LinearRNNRecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        state, reset_carry = h
        return self.project(state)

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> LinearRNNRecurrentStateWithReset:
        return self.algebra.initialize_carry(batch_shape)


class LinearRNN(Module):
    layers: List[LinearRNNLayer]
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
            self.layers.append(LinearRNNLayer(hidden_size, key))
            self.ff.append(
                nn.Sequential(
                    [nn.Linear(2 * hidden_size, hidden_size, key=ff_key), leaky_relu]
                )
            )

    def __call__(
        self, h: LinearRNNRecurrentStateWithReset, x: Input
    ) -> Tuple[LinearRNNRecurrentStateWithReset, ...]:
        emb, start = x
        emb = filter_vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, LinearRNN_layer, h_i in zip(self.ff, self.layers, h):
            tmp, z = LinearRNN_layer(h_i, layer_in)
            h_out.append(tmp)
            z = filter_vmap(ff)(jnp.concatenate([z, emb], axis=-1))
            layer_in = (z, start)
        out = filter_vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> Tuple[LinearRNNRecurrentStateWithReset, ...]:
        return tuple(l.initialize_carry(batch_shape) for l in self.layers)
