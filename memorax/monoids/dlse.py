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

DLSERecurrentState = Float[Array, "Time Hidden Hidden"]
DLSERecurrentStateWithReset = Tuple[DLSERecurrentState, StartFlag]


class DLSEMonoid(Monoid):
    recurrent_size: int
    gamma: Float[Array, "Hidden Hidden"]

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size
        self.gamma = jnp.eye(recurrent_size)

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> DLSERecurrentState:
        return jnp.zeros((1, self.recurrent_size, self.recurrent_size))

    def __call__(
        self, carry: DLSERecurrentState, input: DLSERecurrentState
    ) -> DLSERecurrentState:
        max_val = jnp.maximum(self.gamma @ carry, input)
        return max_val + jnp.log1p(jnp.exp(-jnp.abs(self.gamma @ carry - input)))


class DLSELayer(Memoroid):
    """The Decaying LogSumExp memory model.

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int
    key_size: int
    scan: Callable[
        [
            Callable[
                [DLSERecurrentStateWithReset, DLSERecurrentStateWithReset],
                DLSERecurrentStateWithReset,
            ],
            DLSERecurrentStateWithReset,
            DLSERecurrentStateWithReset,
        ],
        DLSERecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    K: nn.Linear
    Q: nn.Linear
    V: nn.Linear

    def __init__(self, hidden_size, key_size, key):
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.algebra = Resettable(DLSEMonoid(hidden_size))
        self.scan = monoid_scan

        keys = jax.random.split(key, 3)

        self.K = nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[0])
        self.Q = nn.Linear(hidden_size, key_size, use_bias=False, key=keys[1])
        self.V = nn.Linear(hidden_size, hidden_size, use_bias=False, key=keys[2])

    def forward_map(self, x: Input) -> DLSERecurrentStateWithReset:
        emb, start = x
        k = self.K(emb)
        v = self.V(emb)
        kv = jnp.outer(k, v)
        return kv, start

    def backward_map(
        self, h: DLSERecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        state, reset_carry = h
        q = self.Q(emb)
        out = q @ (state / jnp.linalg.norm(state))
        return out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> DLSERecurrentStateWithReset:
        return self.algebra.initialize_carry(batch_shape)


class DLSE(Module):
    layers: List[DLSELayer]
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
            self.layers.append(DLSELayer(hidden_size, hidden_size, key))
            self.ff.append(
                nn.Sequential(
                    [nn.Linear(2 * hidden_size, hidden_size, key=ff_key), leaky_relu]
                )
            )

    def __call__(
        self, h: DLSERecurrentStateWithReset, x: Input
    ) -> Tuple[DLSERecurrentStateWithReset, ...]:
        emb, start = x
        emb = filter_vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, dlse_layer, h_i in zip(self.ff, self.layers, h):
            tmp, z = dlse_layer(h_i, layer_in)
            h_out.append(tmp)
            z = filter_vmap(ff)(jnp.concatenate([z, emb], axis=-1))
            layer_in = (z, start)
        out = filter_vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> Tuple[DLSERecurrentStateWithReset, ...]:
        return tuple(l.initialize_carry(batch_shape) for l in self.layers)
