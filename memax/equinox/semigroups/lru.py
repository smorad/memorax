# https://github.com/NicolasZucchet/minimal-LRU/blob/main/lru/model.py
from beartype.typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Complex, Float, PRNGKeyArray, Scalar, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, Semigroup, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import semigroup_scan

LRURecurrentState = Tuple[Complex[Array, "Recurrent"], Complex[Array, "Recurrent"]]
LRURecurrentStateWithReset = Tuple[LRURecurrentState, StartFlag]


# Inits
@jaxtyped(typechecker=typechecker)
def glorot_init(
    key: PRNGKeyArray, shape: Tuple[int, ...], normalization: Scalar = jnp.array(1.0)
):
    return jax.random.normal(key=key, shape=shape) / normalization


class LRUSemigroup(Semigroup):
    """The Linear Recurrent Unit semigroup (recurrent update) from https://arxiv.org/abs/2303.06349."""

    recurrent_size: int

    def __init__(
        self,
        recurrent_size: int,
    ):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LRURecurrentState:
        # Represent a diagonal matrix as a vector
        return (
            jnp.ones((self.recurrent_size,), dtype=jnp.complex64),
            jnp.zeros((self.recurrent_size), dtype=jnp.complex64),
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: LRURecurrentState, input: LRURecurrentState
    ) -> LRURecurrentState:
        # Ax + Bu, but A is diagonal, and we treat it as a vector
        # So we can be more efficient by writing Ax as vec(A) * x
        A_i, bu_i = carry
        A_j, bu_j = input
        return A_j * A_i, A_j * bu_i + bu_j


class LRU(GRAS):
    """
    The Linear Recurrent Unit from https://arxiv.org/abs/2303.06349.

    You might want to use this as a building block for a more complex model.
    """

    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [LRURecurrentStateWithReset, LRURecurrentStateWithReset],
                LRURecurrentStateWithReset,
            ],
            LRURecurrentStateWithReset,
            LRURecurrentStateWithReset,
        ],
        LRURecurrentStateWithReset,
    ]
    gamma_log: Float[Array, "Recurrent"]
    B_re: Float[Array, "Recurrent Hidden"]
    B_im: Float[Array, "Recurrent Hidden"]
    C_re: Float[Array, "Hidden Recurrent"]
    C_im: Float[Array, "Hidden Recurrent"]
    D: Float[Array, "Hidden"]
    nu_log: Float[Array, "Recurrent"]
    theta_log: Float[Array, "Recurrent"]
    gamma_log: Float[Array, "Recurrent"]

    hidden_size: int  # input and output dimensions
    recurrent_size: int  # hidden state dimension
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = jnp.pi * 2

    def __init__(self, recurrent_size, hidden_size, key):
        keys = jax.random.split(key, 7)
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        unwrapped = LRUSemigroup(recurrent_size)
        self.algebra = Resettable(unwrapped)
        self.scan = semigroup_scan

        u1 = jax.random.uniform(keys[5], (self.recurrent_size,))
        u2 = jax.random.uniform(keys[6], (self.recurrent_size,))
        self.nu_log = jnp.log(
            -0.5 * jnp.log(u1 * (self.r_max**2 - self.r_min**2) + self.r_min**2)
        )
        self.theta_log = jnp.log(self.max_phase * u2)

        self.B_re = glorot_init(
            keys[0],
            (self.recurrent_size, self.hidden_size),
            normalization=jnp.sqrt(2 * self.hidden_size),
        )
        self.B_im = glorot_init(
            keys[1],
            (self.recurrent_size, self.hidden_size),
            normalization=jnp.sqrt(2 * self.hidden_size),
        )
        self.C_re = glorot_init(
            keys[2],
            (self.hidden_size, self.recurrent_size),
            normalization=jnp.sqrt(self.recurrent_size),
        )
        self.C_im = glorot_init(
            keys[3],
            (self.hidden_size, self.recurrent_size),
            normalization=jnp.sqrt(self.recurrent_size),
        )
        self.D = glorot_init(keys[4], (self.hidden_size,))

        self.gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(self.diag_lambda()) ** 2))

    @jaxtyped(typechecker=typechecker)
    def diag_lambda(self) -> Complex[Array, "Recurrent"]:
        return jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))

    @jaxtyped(typechecker=typechecker)
    def forward_map(self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None):
        emb, start = x
        B_norm = jax.lax.complex(self.B_re, self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        Bu = B_norm @ emb.astype(jnp.complex64)
        return (self.diag_lambda(), Bu), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: LRURecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        state, reset_flag = h
        emb, start = x
        lambdas, lambda_x_Bu = state
        C = jax.lax.complex(self.C_re, self.C_im)
        out = (C @ lambda_x_Bu).real + self.D * emb
        return out

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LRURecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
