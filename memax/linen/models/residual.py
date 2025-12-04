from beartype.typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
from jaxtyping import PRNGKeyArray, Shaped

from memax.mtypes import Input, ResetRecurrentState
from memax.linen.groups import Module


class ResidualModel(Module):
    """A model consisting of multiple recurrent layers, with a
    residual connection from the original input into each layer.

    There is a nonlinearity between network layers."""

    make_layer_fn: Callable[..., nn.Module]
    output_size: int
    recurrent_size: int
    num_layers: int = 2
    activation: Callable[[jax.Array], jax.Array] = jax.nn.leaky_relu

    def setup(self):
        layers = []
        ff = []
        self.map_in = nn.Dense(self.recurrent_size)
        self.map_out = nn.Dense(self.output_size)
        for _ in range(self.num_layers):
            layers.append(self.make_layer_fn(recurrent_size=self.recurrent_size))
            ff.append(
                nn.Sequential(
                    [
                        nn.Dense(self.recurrent_size),
                        nn.LayerNorm(use_scale=False, use_bias=False),
                        self.activation,
                    ]
                )
            )
        self.layers = layers
        self.ff = ff

    def __call__(
        self, h: ResetRecurrentState, x: Input
    ) -> Tuple[ResetRecurrentState, ...]:
        emb, start = x
        emb = jax.vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, recurrent_layer, h_i in zip(self.ff, self.layers, h):
            tmp, z = recurrent_layer(h_i, layer_in)
            h_out.append(tmp)
            z = jax.vmap(ff)(z)
            layer_in = (z, start)
        out = jax.vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> Tuple[ResetRecurrentState, ...]:
        if key is None:
            keys = tuple(None for _ in range(self.num_layers))
        else:
            keys = jax.random.split(key, self.num_layers)

        return tuple(l.initialize_carry(k) for l, k in zip(self.layers, keys))

    @nn.nowrap
    def zero_carry(self) -> Tuple[ResetRecurrentState, ...]:
        layers = [
            self.make_layer_fn(recurrent_size=self.recurrent_size)
            for _ in range(self.num_layers)
        ]
        return tuple(l.zero_carry() for l in layers)

    @nn.nowrap
    def latest_recurrent_state(self, h: ResetRecurrentState) -> ResetRecurrentState:
        layers = [
            self.make_layer_fn(recurrent_size=self.recurrent_size)
            for _ in range(self.num_layers)
        ]
        return tuple(l.latest_recurrent_state(h_i) for l, h_i in zip(layers, h))
