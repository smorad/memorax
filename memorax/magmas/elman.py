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

ElmanRecurrentState = Float[Array, "Time Recurrent"]
ElmanRecurrentStateWithReset = Tuple[ElmanRecurrentState, StartFlag]


class ElmanMagma(Magma):
    """
    The Elman Magma (recurrent update) from
    https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1.
    """

    recurrent_size: int
    U_h: nn.Linear

    def __init__(self, recurrent_size: int, key):
        self.recurrent_size = recurrent_size
        self.U_h = nn.Linear(recurrent_size, recurrent_size, key=key)

    def __call__(
        self, carry: ElmanRecurrentState, input: ElmanRecurrentState
    ) -> ElmanRecurrentState:
        return jax.nn.tanh(self.U_h(carry) + input)

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> ElmanRecurrentState:
        return jnp.zeros((*batch_shape, self.recurrent_size))


class LNElmanMagma(Magma):
    """
    The Elman Magma (recurrent update) from
    https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1.
    The tanh is replaced with layernorm.
    """

    recurrent_size: int
    U_h: nn.Linear
    ln: nn.LayerNorm

    def __init__(self, recurrent_size: int, key):
        self.recurrent_size = recurrent_size
        self.U_h = nn.Linear(recurrent_size, recurrent_size, key=key)
        self.ln = nn.LayerNorm((recurrent_size,), use_bias=False, use_weight=False)

    def __call__(
        self, carry: ElmanRecurrentState, input: ElmanRecurrentState
    ) -> ElmanRecurrentState:
        return self.ln(self.U_h(carry) + input)

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> ElmanRecurrentState:
        return jnp.zeros((*batch_shape, self.recurrent_size))


class ElmanLayer(Memoroid):
    """The Elman RNN from
    https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog1402_1."""

    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [ElmanRecurrentStateWithReset, ElmanRecurrentStateWithReset],
                ElmanRecurrentStateWithReset,
            ],
            ElmanRecurrentStateWithReset,
            ElmanRecurrentStateWithReset,
        ],
        ElmanRecurrentStateWithReset,
    ]
    W_h: nn.Linear
    W_y: nn.Linear

    def __init__(self, recurrent_size, hidden_size, ln_variant=False, *, key):
        keys = jax.random.split(key, 3)
        if ln_variant:
            self.algebra = Resettable(LNElmanMagma(recurrent_size, key=keys[0]))
        else:
            self.algebra = Resettable(ElmanMagma(recurrent_size, key=keys[0]))
        self.scan = magma_scan
        self.W_h = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[1])
        self.W_y = nn.Linear(recurrent_size, hidden_size, key=keys[2])

    def forward_map(self, x: Input) -> ElmanRecurrentStateWithReset:
        emb, start = x
        return self.W_h(emb), start

    def backward_map(
        self, h: ElmanRecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        z, reset_flag = h
        emb, start = x
        return self.W_y(z)

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> ElmanRecurrentState:
        return self.algebra.initialize_carry(batch_shape)


class Elman(Module):
    layers: List[ElmanLayer]
    ff: List[nn.Sequential]
    map_in: nn.Linear
    map_out: nn.Linear

    def __init__(
        self, input_size, output_size, hidden_size, num_layers, ln_variant=False, *, key
    ):
        self.layers = []
        self.ff = []
        self.map_in = nn.Linear(input_size, hidden_size, key=key)
        self.map_out = nn.Linear(hidden_size, output_size, key=key)
        for _ in range(num_layers):
            key, ff_key = jax.random.split(key)
            self.layers.append(
                ElmanLayer(hidden_size, hidden_size, ln_variant=ln_variant, key=key)
            )
            self.ff.append(
                nn.Sequential(
                    [
                        nn.Linear(hidden_size, hidden_size, key=ff_key),
                        leaky_relu,
                    ]
                )
            )

    def __call__(
        self, h: ElmanRecurrentStateWithReset, x: Input
    ) -> Tuple[ElmanRecurrentStateWithReset, ...]:
        emb, start = x
        emb = filter_vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, Elman_layer, h_i in zip(self.ff, self.layers, h):
            tmp, z = Elman_layer(h_i, layer_in)
            h_out.append(tmp)
            z = filter_vmap(ff)(z)
            layer_in = (z, start)
        out = filter_vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> Tuple[ElmanRecurrentStateWithReset, ...]:
        return tuple(l.initialize_carry(batch_shape) for l in self.layers)
