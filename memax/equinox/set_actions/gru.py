from beartype.typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, SetAction, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import set_action_scan

GRURecurrentState = Float[Array, "Recurrent"]
GRURecurrentStateWithReset = Tuple[GRURecurrentState, StartFlag]


class GRUMagma(SetAction):
    """
    The Gated Recurrent Unit set action

    Paper: https://arxiv.org/abs/1406.1078
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

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: GRURecurrentState, input: Float[Array, "Recurrent"]
    ) -> GRURecurrentState:
        z = jax.nn.sigmoid(self.W_z(input) + self.U_z(carry))
        r = jax.nn.sigmoid(self.W_r(input) + self.U_r(carry))
        h_hat = jax.nn.tanh(self.W_h(input) + self.U_h(r * carry))
        out = (1 - z) * carry + z * h_hat
        return out

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GRURecurrentState:
        return jnp.zeros((self.recurrent_size,))


class GRU(GRAS):
    """
    The Gated Recurrent Unit layer

    Paper: https://arxiv.org/abs/1406.1078
    """

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
    recurrent_size: int

    def __init__(self, recurrent_size, key):
        keys = jax.random.split(key, 3)
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(GRUMagma(recurrent_size, key=keys[0]))
        self.scan = set_action_scan

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GRURecurrentStateWithReset:
        emb, start = x
        return emb, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: GRURecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        z, reset_flag = h
        emb, start = x
        return z

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GRURecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
