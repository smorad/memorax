from beartype.typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, Semigroup, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import semigroup_scan

NMaxRecurrentState = Float[Array, "Hidden"]
NMaxRecurrentStateWithReset = Tuple[NMaxRecurrentState, StartFlag]


class NMaxSemigroup(Semigroup):
    recurrent_size: int
    apply_decay: bool
    decay: jax.Array

    def __init__(self, recurrent_size: int, decay=True):
        self.recurrent_size = recurrent_size
        self.apply_decay = decay
        self.decay = jnp.ones((self.recurrent_size,), dtype=jnp.float32)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> NMaxRecurrentState:
        return jnp.zeros((self.recurrent_size,))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: NMaxRecurrentState, input: NMaxRecurrentState
    ) -> NMaxRecurrentState:
        if self.apply_decay:
            carry = self.decay * carry
        return jnp.maximum(carry, input)


class NMax(GRAS):
    """A Gated Impulse Linear Recurrent layer.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [NMaxRecurrentStateWithReset, NMaxRecurrentStateWithReset],
                NMaxRecurrentStateWithReset,
            ],
            NMaxRecurrentStateWithReset,
            NMaxRecurrentStateWithReset,
        ],
        NMaxRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    g: nn.Sequential

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(NMaxSemigroup(recurrent_size))
        self.scan = semigroup_scan

        self.g = nn.Sequential(
            [
                nn.Linear(recurrent_size, recurrent_size, key=key),
                nn.Lambda(jax.nn.sigmoid),
            ]
        )

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> NMaxRecurrentStateWithReset:
        emb, start = x
        z = emb
        return z, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: NMaxRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        z = state / jnp.linalg.norm(state, ord=1)
        return z
        # g = self.g(emb)
        # return g * z + (1 - g) * emb

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> NMaxRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
