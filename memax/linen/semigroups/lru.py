# https://github.com/NicolasZucchet/minimal-LRU/blob/main/lru/model.py
from functools import partial
from beartype.typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Complex, Float, PRNGKeyArray, Scalar, Shaped, jaxtyped

from memax.mtypes import Input, StartFlag
from memax.linen.groups import Semigroup, Resettable
from memax.linen.gras import GRAS
from memax.linen.scans import semigroup_scan

LRURecurrentState = Tuple[Complex[Array, "Recurrent"], Complex[Array, "Recurrent"]]
LRURecurrentStateWithReset = Tuple[LRURecurrentState, StartFlag]


# Inits
@jaxtyped(typechecker=typechecker)
def glorot_init(
    key: PRNGKeyArray, shape: Tuple[int, ...], normalization: Scalar = jnp.array(1.0)
):
    return jax.random.normal(key=key, shape=shape) / normalization


@jaxtyped(typechecker=typechecker)
def nu_init(
    key: PRNGKeyArray,
    shape: Tuple[int, ...],
    r_min: float,
    r_max: float,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


@jaxtyped(typechecker=typechecker)
def theta_init(
    key: PRNGKeyArray,
    shape: Tuple[int, ...],
    max_phase: float,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


@jaxtyped(typechecker=typechecker)
def diag_lambda(nu_log: jax.Array, theta_log: jax.Array) -> jax.Array:
    return jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))


@jaxtyped(typechecker=typechecker)
def gamma_log_init(
    key: PRNGKeyArray, nu_log: jax.Array, theta_log: jax.Array
) -> jax.Array:
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda(nu_log, theta_log)) ** 2))


class LRUSemigroup(Semigroup):
    """The Linear Recurrent Unit semigroup (recurrent update) from https://arxiv.org/abs/2303.06349."""

    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LRURecurrentState:
        # Represent a diagonal matrix as a vector
        return (
            jnp.ones((self.recurrent_size,), dtype=jnp.complex64),
            jnp.zeros((self.recurrent_size), dtype=jnp.complex64),
        )

    @nn.nowrap
    def zero_carry(self) -> LRURecurrentState:
        return (
            jnp.zeros((self.recurrent_size,), dtype=jnp.complex64),
            jnp.zeros((self.recurrent_size), dtype=jnp.complex64),
        )

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self, carry: LRURecurrentState, input: LRURecurrentState
    ) -> LRURecurrentState:
        # Ax + Bu, but A is diagonal, and we treat it as a vector
        # So we can be more efficient by writing Ax as vec(A) * x
        A_i, bu_i = carry
        A_j, bu_j = input
        return A_j * A_i, A_j * bu_i + bu_j
        # return carry * self.diag_lambda() + input


class LRU(GRAS):
    """
    The Linear Recurrent Unit from https://arxiv.org/abs/2303.06349.

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int  # output dimensions
    recurrent_size: int  # hidden state dimension
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = jnp.pi * 2

    def setup(self):
        self.B_re = self.param(
            "B_re",
            partial(glorot_init, normalization=jnp.sqrt(2 * self.hidden_size)),
            (self.recurrent_size, self.hidden_size),
        )
        self.B_im = self.param(
            "B_im",
            partial(glorot_init, normalization=jnp.sqrt(2 * self.hidden_size)),
            (self.recurrent_size, self.hidden_size),
        )
        self.C_re = self.param(
            "C_re",
            partial(glorot_init, normalization=jnp.sqrt(self.recurrent_size)),
            (self.hidden_size, self.recurrent_size),
        )
        self.C_im = self.param(
            "C_im",
            partial(glorot_init, normalization=jnp.sqrt(self.recurrent_size)),
            (self.hidden_size, self.recurrent_size),
        )
        self.D = self.param("D", glorot_init, (self.hidden_size,))
        self.nu_log = self.param(
            "nu_log",
            partial(nu_init, r_min=self.r_min, r_max=self.r_max),
            (self.recurrent_size,),
        )
        self.theta_log = self.param(
            "theta_log",
            partial(theta_init, max_phase=self.max_phase),
            (self.recurrent_size,),
        )
        self.gamma_log = self.param(
            "gamma_log",
            partial(gamma_log_init, nu_log=self.nu_log, theta_log=self.theta_log),
        )

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
        return (diag_lambda(self.nu_log, self.theta_log), Bu), start

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
        out = (jax.lax.complex(self.C_re, self.C_im) @ lambda_x_Bu).real + self.D * emb
        return out

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LRURecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> LRURecurrentStateWithReset:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(LRUSemigroup(**kwargs))

    @staticmethod
    def default_scan():
        return semigroup_scan
