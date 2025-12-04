from beartype.typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Complex, Float, Int, PRNGKeyArray, Real, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, Semigroup, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import semigroup_scan

FFMRecurrentState = Tuple[Complex[Array, "Trace Context"], Int[Array, ""]]
FFMRecurrentStateWithReset = Tuple[FFMRecurrentState, StartFlag]


class Gate(eqx.Module):
    linear: nn.Linear

    def __init__(self, input_size, output_size, key):
        self.linear = nn.Linear(input_size, output_size, key=key)

    def __call__(self, x):
        return jax.nn.sigmoid(self.linear(x))


class FFMSemigroup(Semigroup):
    """The Fast and Forgetful Memory semigroup (recurrent update) from https://arxiv.org/abs/2310.04128."""

    trace_size: int
    context_size: int
    params: Tuple[Float[Array, "Trace"], Float[Array, "Context"]]

    def __init__(
        self,
        trace_size,
        context_size,
        deterministic_init,
        min_period=1,
        max_period=1024,
        *,
        key,
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

    @jaxtyped(typechecker=typechecker)
    def log_gamma(self, t: Real[Array, ""]) -> Complex[Array, "Trace Context"]:
        a, b = self.params
        a = -jnp.abs(a).reshape((self.trace_size, 1))
        b = b.reshape(1, self.context_size)
        ab = jax.lax.complex(a, b)
        return ab * t.reshape(1, 1)

    @jaxtyped(typechecker=typechecker)
    def gamma(self, t: Real[Array, ""]) -> Complex[Array, "Trace Context"]:
        return jnp.exp(self.log_gamma(t))

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FFMRecurrentState:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        carry_shape = (self.trace_size, self.context_size)

        return jnp.zeros(carry_shape, dtype=jnp.complex64), jnp.array(
            0, dtype=jnp.int32
        )

    @jaxtyped(typechecker=typechecker)
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


class FFM(GRAS):
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
        self.scan = semigroup_scan

        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.pre = nn.Linear(hidden_size, trace_size, key=k1)
        self.gate_in = Gate(hidden_size, trace_size, key=k2)
        self.gate_out = Gate(hidden_size, hidden_size, key=k3)
        self.algebra = Resettable(
            FFMSemigroup(trace_size, context_size, True, 1, 10_000, key=k4)
        )
        self.mix = nn.Linear(2 * trace_size * context_size, hidden_size, key=k5)
        self.ln = nn.LayerNorm(hidden_size, use_weight=False, use_bias=False)

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FFMRecurrentStateWithReset:
        emb, start = x
        gate_in = self.gate_in(emb)
        pre = self.pre(emb)
        gated = pre * gate_in
        scan_input = jnp.repeat(
            jnp.expand_dims(gated, 1), self.context_size, axis=1
        ).astype(jnp.complex64)
        dt = jnp.array(1)
        return (scan_input, dt), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: FFMRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        (z, dt), reset_flag = h
        emb, start = x
        z = jnp.concatenate([jnp.real(z), jnp.imag(z)], axis=-1).reshape(-1)
        z = self.mix(z)
        gate_out = self.gate_out(emb)
        out = self.ln(z * gate_out) + emb * (1 - gate_out)
        return out

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FFMRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
