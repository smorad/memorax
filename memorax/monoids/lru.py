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

    def __init__(self, hidden_size, recurrent_size, key):
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

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> LRURecurrentState:
        return self.algebra.initialize_carry(batch_shape)

    # def __call__(self, state, x, start):
    #     """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
    #     diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
    #     B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(jnp.exp(self.gamma_log), axis=-1)
    #     C = self.C_re + 1j * self.C_im

    #     Lambda_elements = jnp.repeat(diag_lambda[None, ...], x.shape[0], axis=0)
    #     Bu_elements = jax.vmap(lambda u: B_norm @ u)(x.astype(jnp.complex64))

    #     Lambda_elements = jnp.concatenate([
    #         jnp.ones((1, diag_lambda.shape[0])),
    #         Lambda_elements,
    #     ])

    #     Bu_elements = jnp.concatenate([
    #         state,
    #         Bu_elements,
    #     ])

    #     start = start.reshape([-1, 1])
    #     start = jnp.concatenate([jnp.zeros_like(start[:1]), start], axis=0)

    #     # Compute hidden states
    #     _, _, xs = parallel_scan(wrapped_associative_update, (start, Lambda_elements, Bu_elements))
    #     xs = xs[1:]

    #     # Use them to compute the output of the module
    #     outputs = jax.vmap(lambda x, u: (C @ x).real + self.D * u)(xs, x)

    #     return xs[None, -1], outputs


# class SequenceLayer(eqx.Module):
#     """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""

#     lru: LRU  # lru module
#     hidden_size: int  # model size
#     recurrent_size: int # hidden size
#     out1: eqx.Module  # first output linear layer
#     out2: eqx.Module  # second output linear layer
#     normalization: eqx.Module  # layer norm

#     def __init__(self, hidden_size, recurrent_size, key):
#         """Initializes the ssm, layer norm and dropout"""
#         keys = jax.random.split(key, 3)
#         self.hidden_size = hidden_size
#         self.recurrent_size = recurrent_size
#         self.lru = LRU(self.hidden_size, recurrent_size, key=keys[0])
#         self.out1 = eqx.filter_vmap(nn.Linear(self.hidden_size, self.hidden_size, key=keys[1]))
#         self.out2 = eqx.filter_vmap(nn.Linear(self.hidden_size, self.hidden_size, key=keys[2]))
#         self.normalization = eqx.filter_vmap(nn.LayerNorm(self.hidden_size))

#     def __call__(self, state, x, start):
#         skip = x
#         x = self.normalization(x)  # pre normalization
#         state, x = self.lru(state, x, start)  # call LRU
#         x = jax.nn.gelu(x)
#         o1 = self.out1(x)
#         x = o1 * jax.nn.sigmoid(self.out2(x))  # GLU
#         return state, skip + x  # skip connection


# class StackedLRU(MemoryModule):
#     """Encoder containing several SequenceLayer"""

#     layers: List[SequenceLayer]
#     encoder: eqx.Module
#     hidden_size: int
#     recurrent_size: int
#     n_layers: int
#     name: str = "StackedLRU"

#     def __init__(self, input_size, hidden_size, recurrent_size, n_layers, key):
#         keys = jax.random.split(key, 7)
#         self.hidden_size = hidden_size
#         self.recurrent_size = recurrent_size
#         self.n_layers = n_layers

#         # self.encoder = nn.Dense(self.hidden_size)
#         self.encoder = nn.Linear(input_size, self.hidden_size, key=keys[0])
#         self.layers = [
#             SequenceLayer(
#                 hidden_size=self.hidden_size,
#                 recurrent_size=self.recurrent_size,
#                 key=keys[i+1]
#             )
#             for i in range(self.n_layers)
#         ]

#     def __call__(self, x, state, start, next_done, key=None):
#         new_states = []
#         for i, layer in enumerate(self.layers):
#             new_s, x = layer(state[i], x, start)
#             new_states.append(new_s)

#         return x, new_states

#     def initial_state(self, shape=tuple()):
#         return [
#             jnp.zeros(
#                 (1, *shape, self.recurrent_size), dtype=jnp.complex64
#             ) for _ in range(self.n_layers)
#         ]
