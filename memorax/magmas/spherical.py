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

SphericalRecurrentState = Float[Array, "Time Recurrent"]
SphericalRecurrentStateWithReset = Tuple[SphericalRecurrentState, StartFlag]


class SphericalMagma(Magma):
    """
    The Spherical Magma (recurrent update) from
    https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1.
    """

    recurrent_size: int
    project: nn.Linear

    def __init__(self, recurrent_size: int, key):
        self.recurrent_size = recurrent_size
        proj_size = int(self.recurrent_size * (self.recurrent_size - 1) / 2)
        self.project = nn.Linear(recurrent_size, proj_size, key=key)

    def __call__(
        self, carry: SphericalRecurrentState, input: SphericalRecurrentState
    ) -> SphericalRecurrentState:
        q = self.project(input)
        A = jnp.zeros((self.recurrent_size, self.recurrent_size))
        tri_idx = jnp.triu_indices_from(A, 1)
        A = A.at[tri_idx].set(q)
        A = A - A.T
        R = jax.scipy.linalg.expm(A)
        return R @ carry
        # return jax.nn.tanh(self.U_h(carry) + input)

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> SphericalRecurrentState:
        # return jnp.zeros((*batch_shape, self.recurrent_size))
        v = jnp.ones((*batch_shape, self.recurrent_size))
        return v / jnp.linalg.norm(v, axis=-1, ord=2)
        # return jnp.ones((*batch_shape, self.recurrent_size)) / jnp.linalg.norm


class SphericalLayer(Memoroid):
    """The Spherical RNN from
    https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1."""

    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [SphericalRecurrentStateWithReset, SphericalRecurrentStateWithReset],
                SphericalRecurrentStateWithReset,
            ],
            SphericalRecurrentStateWithReset,
            SphericalRecurrentStateWithReset,
        ],
        SphericalRecurrentStateWithReset,
    ]
    W_h: nn.Linear
    W_y: nn.Linear

    def __init__(self, recurrent_size, hidden_size, key):
        keys = jax.random.split(key, 3)
        self.algebra = Resettable(SphericalMagma(recurrent_size, key=keys[0]))
        self.scan = magma_scan
        self.W_h = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[1])
        self.W_y = nn.Linear(recurrent_size, hidden_size, key=keys[2])

    def forward_map(self, x: Input) -> SphericalRecurrentStateWithReset:
        emb, start = x
        return self.W_h(emb), start

    def backward_map(
        self, h: SphericalRecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        z, reset_flag = h
        emb, start = x
        return self.W_y(z)

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> SphericalRecurrentState:
        return self.algebra.initialize_carry(batch_shape)


class Spherical(Module):
    layers: List[SphericalLayer]
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
            self.layers.append(SphericalLayer(hidden_size, hidden_size, key=key))
            self.ff.append(
                nn.Sequential(
                    [
                        nn.Linear(hidden_size, hidden_size, key=ff_key),
                        leaky_relu,
                    ]
                )
            )

    def __call__(
        self, h: SphericalRecurrentStateWithReset, x: Input
    ) -> Tuple[SphericalRecurrentStateWithReset, ...]:
        emb, start = x
        emb = filter_vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, Spherical_layer, h_i in zip(self.ff, self.layers, h):
            tmp, z = Spherical_layer(h_i, layer_in)
            h_out.append(tmp)
            z = filter_vmap(ff)(z)
            layer_in = (z, start)
        out = filter_vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> Tuple[SphericalRecurrentStateWithReset, ...]:
        return tuple(l.initialize_carry(batch_shape) for l in self.layers)
