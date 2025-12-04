# https://github.com/NicolasZucchet/minimal-S6/blob/main/lru/model.py
from functools import partial
from beartype.typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.mtypes import Input, StartFlag
from memax.linen.groups import Semigroup, Resettable
from memax.linen.gras import GRAS
from memax.linen.scans import semigroup_scan

S6RecurrentState = Tuple[Float[Array, "Recurrent"], Float[Array, "Recurrent"]]
S6RecurrentStateWithReset = Tuple[S6RecurrentState, StartFlag]


class S6Semigroup(Semigroup):
    """The diagonal S6 semigroup (recurrent update) from https://arxiv.org/abs/2312.00752.
    
    This is a diagonal S5/LRU recurrent update with a learnable timestep parameter. """

    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> S6RecurrentState:
        # Represent a diagonal matrix as a vector
        return (
            jnp.ones((self.recurrent_size,)),
            jnp.zeros((self.recurrent_size)),
        )

    @nn.nowrap
    def zero_carry(self) -> S6RecurrentState:
        return (
            jnp.zeros((self.recurrent_size,)),
            jnp.zeros((self.recurrent_size)),
        )

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self, carry: S6RecurrentState, input: S6RecurrentState
    ) -> S6RecurrentState:
        # Ax + Bu, but A is diagonal, and we treat it as a vector
        # So we can be more efficient by writing Ax as vec(A) * x
        A_i, bu_i = carry
        A_j, bu_j = input
        return A_j * A_i, A_j * bu_i + bu_j


class S6(GRAS):
    """
    The diagonal S6 SSM, an SSM with a trainable dt.

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int  # output dimensions
    recurrent_size: int  # hidden state dimension

    def setup(self):
        self.A_log = self.param(
            "A_log",
            jax.random.normal,
            (self.recurrent_size,),
        )
        self.B = nn.Dense(self.recurrent_size)
        self.C = nn.Dense(self.hidden_size)
        self.dt = nn.Sequential([
            nn.Dense(self.recurrent_size),
            jax.nn.softplus
        ])

    @jaxtyped(typechecker=typechecker)
    def forward_map(self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None):
        emb, start = x
        dt = self.dt(emb)
        A = -jnp.exp(self.A_log)
        A_bar = jnp.exp(dt * A)
        B = self.B(emb)
        # NOTE: A and B are diagonal so we can compute B_bar more simply than the mamba paper
        # Thankfully, inv(A) is just 1 / A if A is diagonal
        # Furthermore the dt's cancel: 1 / (dt A) with dt B
        B_bar = 1 / A * (A_bar - 1.0) * B
        Bu = B_bar * emb
        return (A_bar, Bu), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: S6RecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        state, reset_flag = h
        emb, start = x
        lambdas, lambda_x_Bu = state
        C = self.C(emb)
        out = C * lambda_x_Bu 
        return out

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> S6RecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> S6RecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(S6Semigroup(**kwargs))

    @staticmethod
    def default_scan():
        return semigroup_scan
