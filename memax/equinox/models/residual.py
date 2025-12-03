from beartype.typing import List, Optional, Tuple

import jax
from equinox import filter_vmap, nn
from jaxtyping import PRNGKeyArray, Shaped

from memax.equinox.groups import Module
from memax.mtypes import Input, ResetRecurrentState


class ResidualModel(Module):
    """A model consisting of multiple recurrent layers, with a
    residual connection from the original input into each layer.

    There is a nonlinearity between network layers."""

    layers: List[Module]
    ff: List[nn.Sequential]
    map_in: nn.Linear
    map_out: nn.Linear

    def __init__(
        self,
        make_layer_fn,
        input_size,
        output_size,
        recurrent_size,
        num_layers=2,
        activation=jax.nn.leaky_relu,
        *,
        key
    ):
        self.layers = []
        self.ff = []
        keys = jax.random.split(key, 3)
        self.map_in = nn.Linear(input_size, recurrent_size, key=keys[0])
        self.map_out = nn.Linear(recurrent_size, output_size, key=keys[1])
        key = keys[2]
        for _ in range(num_layers):
            key, ff_key = jax.random.split(key)
            self.layers.append(make_layer_fn(recurrent_size=recurrent_size, key=key))
            self.ff.append(
                nn.Sequential(
                    [
                        nn.Linear(recurrent_size, recurrent_size, key=ff_key),
                        nn.LayerNorm(
                            (recurrent_size,), use_weight=False, use_bias=False
                        ),
                        nn.Lambda(activation),
                    ]
                )
            )

    def __call__(
        self, h: ResetRecurrentState, x: Input, key: Optional[PRNGKeyArray] = None
    ) -> Tuple[ResetRecurrentState, ...]:
        emb, start = x
        emb = filter_vmap(self.map_in)(emb)
        layer_in = (emb, start)
        h_out = []
        for ff, recurrent_layer, h_i in zip(self.ff, self.layers, h):
            if key is None:
                key, rkey = None, None
            else:
                key, rkey = jax.random.split(key)
            tmp, z = recurrent_layer(h_i, layer_in, key=rkey)
            h_out.append(tmp)
            z = filter_vmap(ff)(z)
            layer_in = (z, start)
        out = filter_vmap(self.map_out)(layer_in[0])
        return tuple(h_out), out

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> Tuple[ResetRecurrentState, ...]:
        if key is None:
            keys = tuple(None for _ in range(len(self.layers)))
        else:
            keys = jax.random.split(key, len(self.layers))
        return tuple(l.initialize_carry(k) for l, k in zip(self.layers, keys))

    def latest_recurrent_state(self, h: ResetRecurrentState) -> ResetRecurrentState:
        return tuple(l.latest_recurrent_state(h_i) for l, h_i in zip(self.layers, h))
