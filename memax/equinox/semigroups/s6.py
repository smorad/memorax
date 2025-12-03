# https://github.com/NicolasZucchet/minimal-S6/blob/main/S6/model.py
from beartype.typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from equinox import nn
from beartype import beartype as typechecker
from jaxtyping import Array, Complex, Float, PRNGKeyArray, Scalar, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, Semigroup, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import semigroup_scan

S6RecurrentState = Tuple[Float[Array, "Recurrent"], Float[Array, "Recurrent"]]
S6RecurrentStateWithReset = Tuple[S6RecurrentState, StartFlag]


class S6Semigroup(Semigroup):
    """The S6 semigroup (recurrent update) from https://arxiv.org/abs/2312.00752.
    
    This is an S5/LRU recurrent update with a learnable timestep parameter. """

    recurrent_size: int

    def __init__(
        self,
        recurrent_size: int,
    ):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> S6RecurrentState:
        # Represent a diagonal matrix as a vector
        return (
            jnp.ones((self.recurrent_size,)),
            jnp.zeros((self.recurrent_size,))
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: S6RecurrentState, input: S6RecurrentState
    ) -> S6RecurrentState:
        A_i, bu_i = carry
        A_j, bu_j = input
        return A_j * A_i, A_j * bu_i + bu_j


class S6(GRAS):
    """
    The S6 SSM with diagonal recurrence. We base this on the LRU, and add a trainable dt.

    You might want to use this as a building block for a more complex model.
    """

    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [S6RecurrentStateWithReset, S6RecurrentStateWithReset],
                S6RecurrentStateWithReset,
            ],
            S6RecurrentStateWithReset,
            S6RecurrentStateWithReset,
        ],
        S6RecurrentStateWithReset,
    ]
    A_log: Float[Array, "Recurrent"]
    B: nn.Linear
    C: nn.Linear
    dt: nn.Linear

    hidden_size: int  # input and output dimensions
    recurrent_size: int  # hidden state dimension

    def __init__(self, recurrent_size, hidden_size, key):
        keys = jax.random.split(key, 4)
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        unwrapped = S6Semigroup(recurrent_size)
        self.algebra = Resettable(unwrapped)
        self.scan = semigroup_scan

        self.A_log = jax.random.normal(keys[0], (self.recurrent_size,))
        self.B = nn.Linear(self.hidden_size, self.recurrent_size, key=keys[1])
        self.C = nn.Linear(self.recurrent_size, self.hidden_size, key=keys[2])
        self.dt = nn.Sequential([
            nn.Linear(self.hidden_size, self.recurrent_size, key=keys[3]),
            nn.Lambda(jax.nn.softplus)
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
