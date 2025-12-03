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

MGURecurrentState = Float[Array, "Recurrent"]
MGURecurrentStateWithReset = Tuple[MGURecurrentState, StartFlag]


class MGUSetAction(SetAction):
    """
    The Minimal Gated Unit set action

    Paper: https://arxiv.org/abs/1701.03452
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

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: MGURecurrentState, input: Float[Array, "Recurrent"]
    ) -> MGURecurrentState:
        f = jax.nn.sigmoid(self.W_f(input) + self.U_f(carry))
        h_hat = jax.nn.tanh(self.W_h(input) + self.U_h(f * carry))
        h = (1 - f) * carry + f * h_hat
        return h

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MGURecurrentState:
        return jnp.zeros((self.recurrent_size,))


class MGU(GRAS):
    """The Minimal Gated Unit layer

    Paper: https://arxiv.org/abs/1701.03452
    """

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
    recurrent_size: int

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        keys = jax.random.split(key, 3)
        self.algebra = Resettable(MGUSetAction(recurrent_size, key=keys[0]))
        self.scan = set_action_scan

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MGURecurrentStateWithReset:
        emb, start = x
        return emb, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: MGURecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        z, reset_flag = h
        emb, start = x
        return z

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MGURecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
