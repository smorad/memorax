from beartype.typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, SetAction, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, InputEmbedding, StartFlag
from memax.equinox.scans import set_action_scan

LSTMRecurrentState = Tuple[Float[Array, "Recurrent"], Float[Array, "Recurrent"]]
LSTMRecurrentStateWithReset = Tuple[LSTMRecurrentState, StartFlag]


class LSTMMagma(SetAction):
    """
    The Long Short-Term Memory set action

    Paper: https://www.bioinf.jku.at/publications/older/2604.pdf
    """

    recurrent_size: int
    U_f: nn.Linear
    U_i: nn.Linear
    U_o: nn.Linear
    U_c: nn.Linear
    W_f: nn.Linear
    W_i: nn.Linear
    W_o: nn.Linear
    W_c: nn.Linear

    def __init__(self, recurrent_size: int, key):
        self.recurrent_size = recurrent_size
        keys = jax.random.split(key, 8)
        self.U_f = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[0]
        )
        self.U_i = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[1]
        )
        self.U_o = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[2]
        )
        self.U_c = nn.Linear(
            recurrent_size, recurrent_size, use_bias=False, key=keys[3]
        )

        self.W_f = nn.Linear(recurrent_size, recurrent_size, key=keys[4])
        self.W_i = nn.Linear(recurrent_size, recurrent_size, key=keys[5])
        self.W_o = nn.Linear(recurrent_size, recurrent_size, key=keys[6])
        self.W_c = nn.Linear(recurrent_size, recurrent_size, key=keys[7])

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: LSTMRecurrentState, input: LSTMRecurrentState
    ) -> LSTMRecurrentState:
        x_t, _ = input
        c, h = carry
        f_f = jax.nn.sigmoid(self.W_f(x_t) + self.U_f(h))
        f_i = jax.nn.sigmoid(self.W_i(x_t) + self.U_i(h))
        f_o = jax.nn.sigmoid(self.W_o(x_t) + self.U_o(h))
        f_c = jax.nn.tanh(self.W_c(x_t) + self.U_c(h))

        c = f_f * c + f_i * f_c
        h = f_o * c

        return (c, h)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LSTMRecurrentState:
        return (
            jnp.zeros((self.recurrent_size,)),
            jnp.zeros((self.recurrent_size,)),
        )


class LSTM(GRAS):
    """
    The Long Short-Term Memory layer

    Paper: https://www.bioinf.jku.at/publications/older/2604.pdf
    """

    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [LSTMRecurrentStateWithReset, LSTMRecurrentStateWithReset],
                LSTMRecurrentStateWithReset,
            ],
            LSTMRecurrentStateWithReset,
            LSTMRecurrentStateWithReset,
        ],
        LSTMRecurrentStateWithReset,
    ]

    def __init__(self, recurrent_size, key):
        keys = jax.random.split(key, 3)
        self.algebra = Resettable(LSTMMagma(recurrent_size, key=keys[0]))
        self.scan = set_action_scan

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LSTMRecurrentStateWithReset:
        emb, start = x
        c = jnp.zeros_like(emb)
        return (emb, c), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: LSTMRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) ->  Float[Array, "Recurrent"]:
        (c_t, h_t), reset_flag = h
        emb, start = x
        return h_t

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LSTMRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
