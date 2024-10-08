# https://github.com/NicolasZucchet/minimal-LRU/blob/main/lru/model.py
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PRNGKeyArray, Scalar

from memorax.groups import BinaryAlgebra, Monoid, Resettable
from memorax.memoroid import Memoroid
from memorax.mtypes import Input, StartFlag
from memorax.scans import monoid_scan

LRURecurrentState = Complex[Array, "Time Recurrent"]
LRURecurrentStateWithReset = Tuple[LRURecurrentState, StartFlag]


# Inits
def glorot_init(
    key: PRNGKeyArray, shape: Tuple[int, ...], normalization: Scalar = jnp.array(1.0)
) -> Float[Array, "{*shape}"]:
    return jax.random.normal(key=key, shape=shape) / normalization


class LRUMonoid(Monoid):
    """The Linear Recurrent Unit monoid (recurrent update) from https://arxiv.org/abs/2303.06349."""

    recurrent_size: int
    nu_log: Float[Array, "Recurrent"]
    theta_log: Float[Array, "Recurrent"]

    def __init__(
        self,
        recurrent_size: int,
        r_min: float = 0.0,
        r_max: float = 1.0,
        max_phase: float = jnp.pi * 2,
        *,
        key: PRNGKeyArray
    ):
        self.recurrent_size = recurrent_size
        keys = jax.random.split(key, 3)
        u1 = jax.random.uniform(keys[0], (self.recurrent_size,))
        u2 = jax.random.uniform(keys[1], (self.recurrent_size,))
        self.nu_log = jnp.log(
            -0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2)
        )
        self.theta_log = jnp.log(max_phase * u2)

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> LRURecurrentState:
        # Represent a diagonal matrix as a vector
        return jnp.ones((1, self.recurrent_size), dtype=jnp.complex64)

    def diag_lambda(self):
        return jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))

    def __call__(
        self, carry: LRURecurrentState, input: LRURecurrentState
    ) -> LRURecurrentState:
        # Ax + Bu, but A is diagonal, and we treat it as a vector
        # So we can be more efficient by writing Ax as vec(A) * x
        return carry * self.diag_lambda() + input


class LRU(Memoroid):
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

    hidden_size: int  # input and output dimensions
    recurrent_size: int  # hidden state dimension

    def __init__(self, recurrent_size, hidden_size, key):
        keys = jax.random.split(key, 6)
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        unwrapped = LRUMonoid(recurrent_size, key=keys[0])
        self.algebra = Resettable(unwrapped)
        self.scan = monoid_scan

        self.B_re = glorot_init(
            keys[1],
            (self.recurrent_size, self.hidden_size),
            normalization=jnp.sqrt(2 * self.hidden_size),
        )
        self.B_im = glorot_init(
            keys[2],
            (self.recurrent_size, self.hidden_size),
            normalization=jnp.sqrt(2 * self.hidden_size),
        )
        self.C_re = glorot_init(
            keys[3],
            (self.hidden_size, self.recurrent_size),
            normalization=jnp.sqrt(self.recurrent_size),
        )
        self.C_im = glorot_init(
            keys[4],
            (self.hidden_size, self.recurrent_size),
            normalization=jnp.sqrt(self.recurrent_size),
        )
        self.D = glorot_init(keys[5], (self.hidden_size,))
        self.gamma_log = jnp.log(jnp.sqrt(1 - jnp.abs(unwrapped.diag_lambda()) ** 2))

    def forward_map(self, x: Input):
        emb, start = x
        B_norm = jax.lax.complex(self.B_re, self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        Bu = B_norm @ emb.astype(jnp.complex64)
        return Bu, start

    def backward_map(
        self, h: LRURecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        state, reset_flag = h
        emb, start = x
        out = (jax.lax.complex(self.C_re, self.C_im) @ state).real + self.D * emb
        return out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> LRURecurrentStateWithReset:
        return self.algebra.initialize_carry(batch_shape)
