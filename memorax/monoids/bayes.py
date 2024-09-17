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

LogBayesRecurrentState = Float[Array, "Time Hidden"]
LogBayesRecurrentStateWithReset = Tuple[LogBayesRecurrentState, StartFlag]


class LogBayesMonoid(Monoid):
    recurrent_size: int
    gamma: Float[Array, "Hidden Hidden"]

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size
        self.gamma = jnp.eye(recurrent_size)

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> LogBayesRecurrentState:
        # return jnp.zeros((*batch_shape, 1, self.recurrent_size, self.recurrent_size))
        return jnp.ones((1, self.recurrent_size)) * -jnp.log(self.recurrent_size)

    def __call__(
        self, carry: LogBayesRecurrentState, input: LogBayesRecurrentState
    ) -> LogBayesRecurrentState:
        return carry + input  # In log space, equivalent to carry @ input


class LogBayesLayer(Memoroid):
    """The Decaying LogSumExp memory model.

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int
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

    def __init__(self, hidden_size, key):
        self.hidden_size = hidden_size
        self.algebra = Resettable(LogBayesMonoid(hidden_size))
        self.scan = monoid_scan

        keys = jax.random.split(key)

        self.project = nn.Linear(hidden_size, hidden_size, key=keys[0])

    def forward_map(self, x: Input) -> LogBayesRecurrentStateWithReset:
        emb, start = x
        z = self.project(emb)
        return z, start

    def backward_map(
        self, h: LogBayesRecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        state, reset_carry = h
        out = jax.nn.softmax(state)
        return out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> LogBayesRecurrentStateWithReset:
        return self.algebra.initialize_carry(batch_shape)


class LogBayes(Module):
    layers: List[LogBayesLayer]
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
            self.layers.append(LogBayesLayer(hidden_size, key))
            self.ff.append(
                nn.Sequential(
                    [nn.Linear(2 * hidden_size, hidden_size, key=ff_key), leaky_relu]
                )
            )

    def __call__(
        self, h: LogBayesRecurrentStateWithReset, x: Input
    ) -> Tuple[LogBayesRecurrentStateWithReset, ...]:
        emb, start = x
        emb = filter_vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, LogBayes_layer, h_i in zip(self.ff, self.layers, h):
            tmp, z = LogBayes_layer(h_i, layer_in)
            h_out.append(tmp)
            z = filter_vmap(ff)(jnp.concatenate([z, emb], axis=-1))
            layer_in = (z, start)
        out = filter_vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> Tuple[LogBayesRecurrentStateWithReset, ...]:
        return tuple(l.initialize_carry(batch_shape) for l in self.layers)
