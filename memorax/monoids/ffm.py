from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import Array, Complex, Float, Int, Real

from memorax.groups import BinaryAlgebra, Monoid, Resettable
from memorax.memoroid import Memoroid
from memorax.mtypes import Input, StartFlag
from memorax.scans import monoid_scan

FFMRecurrentState = Tuple[Complex[Array, "Time Trace Context"], Int[Array, "Time"]]
FFMRecurrentStateWithReset = Tuple[FFMRecurrentState, StartFlag]


class Gate(eqx.Module):
    linear: nn.Linear

    def __init__(self, input_size, output_size, key):
        self.linear = nn.Linear(input_size, output_size, key=key)

    def __call__(self, x):
        return jax.nn.sigmoid(self.linear(x))


class FFMMonoid(Monoid):
    """The Fast and Forgetful Memory monoid (recurrent update) from https://arxiv.org/abs/2310.04128."""

    trace_size: int
    context_size: int
    params: Tuple[Float[Array, "Trace"], Float[Array, "Context"]]

    def __init__(
        self, trace_size, context_size, deterministic_init, min_period, max_period, key
    ):
        self.trace_size = trace_size
        self.context_size = context_size
        if deterministic_init:
            a_low = 1e-6
            a_high = 0.5
            a = jnp.linspace(a_low, a_high, trace_size)
            b = 2 * jnp.pi / jnp.linspace(min_period, max_period, context_size)
            self.params = (a, b)
        else:
            raise NotImplementedError

    def log_gamma(self, t: Real[Array, "Time"]) -> Complex[Array, "Time Trace Context"]:
        a, b = self.params
        a = -jnp.abs(a).reshape((1, self.trace_size, 1))
        b = b.reshape(1, 1, self.context_size)
        ab = jax.lax.complex(a, b)
        return ab * t.reshape(t.shape[0], 1, 1)

    def gamma(self, t: Real[Array, "Time"]) -> Complex[Array, "Time Trace Context"]:
        return jnp.exp(self.log_gamma(t))

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> FFMRecurrentState:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        carry_shape = (*batch_shape, 1, self.trace_size, self.context_size)
        t_shape = (*batch_shape, 1)

        return jnp.zeros(carry_shape, dtype=jnp.complex64), jnp.zeros(
            t_shape, dtype=jnp.int32
        )

    def __call__(
        self, carry: FFMRecurrentState, input: FFMRecurrentState
    ) -> FFMRecurrentState:
        (
            state,
            i,
        ) = carry
        x, j = input
        state = state * self.gamma(j) + x
        return state, j + i


class FFM(Memoroid):
    """Fast and Forgetful Memory from https://arxiv.org/abs/2310.04128."""

    hidden_size: int
    trace_size: int
    context_size: int
    scan: Callable[
        [
            Callable[
                [FFMRecurrentStateWithReset, FFMRecurrentStateWithReset],
                FFMRecurrentStateWithReset,
            ],
            FFMRecurrentStateWithReset,
            FFMRecurrentStateWithReset,
        ],
        FFMRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    pre: nn.Linear
    gate_in: Gate
    gate_out: Gate
    mix: nn.Linear
    ln: nn.LayerNorm

    def __init__(
        self,
        hidden_size: int,
        trace_size: int,
        context_size: int,
        key: Array,
    ):
        self.hidden_size = hidden_size
        self.trace_size = trace_size
        self.context_size = context_size
        self.scan = monoid_scan

        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.pre = nn.Linear(hidden_size, trace_size, key=k1)
        self.gate_in = Gate(hidden_size, trace_size, key=k2)
        self.gate_out = Gate(hidden_size, hidden_size, key=k3)
        self.algebra = Resettable(
            FFMMonoid(trace_size, context_size, True, 1, 1000, key=k4)
        )
        self.mix = nn.Linear(2 * trace_size * context_size, hidden_size, key=k5)
        self.ln = nn.LayerNorm(hidden_size, use_weight=False, use_bias=False)

    def forward_map(self, x: Input) -> FFMRecurrentStateWithReset:
        emb, start = x
        gate_in = self.gate_in(emb)
        pre = self.pre(emb)
        gated = pre * gate_in
        scan_input = jnp.repeat(
            jnp.expand_dims(gated, 1), self.context_size, axis=1
        ).astype(jnp.complex64)
        dt = jnp.array(1)
        return (scan_input, dt), start

    def backward_map(
        self, h: FFMRecurrentStateWithReset, x: Input
    ) -> Float[Array, "{self.hidden_size}"]:
        (z, dt), reset_flag = h
        emb, start = x
        z = jnp.concatenate([jnp.real(z), jnp.imag(z)], axis=-1).reshape(-1)
        z = self.mix(z)
        gate_out = self.gate_out(emb)
        out = self.ln(z * gate_out) + emb * (1 - gate_out)
        return out

    def initialize_carry(
        self, batch_shape: Tuple[int, ...] = ()
    ) -> FFMRecurrentStateWithReset:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return self.algebra.initialize_carry(batch_shape)
