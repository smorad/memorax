from beartype.typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import filter_vmap, nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, Module, Semigroup, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import semigroup_scan

DLSERecurrentState = Float[Array, "Hidden Hidden"]
DLSERecurrentStateWithReset = Tuple[DLSERecurrentState, StartFlag]


class DLSESemigroup(Semigroup):
    recurrent_size: int
    gamma: Float[Array, "Hidden Hidden"]

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size
        self.gamma = jnp.eye(recurrent_size)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> DLSERecurrentState:
        return jnp.eye(self.recurrent_size)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: DLSERecurrentState, input: DLSERecurrentState
    ) -> DLSERecurrentState:
        # return carry @ input
        max_val = jnp.maximum(self.gamma @ carry, input)
        return max_val + jnp.log1p(jnp.exp(-jnp.abs(self.gamma @ carry - input)))


class DLSE(GRAS):
    """The Decaying LogSumExp memory model.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [DLSERecurrentStateWithReset, DLSERecurrentStateWithReset],
                DLSERecurrentStateWithReset,
            ],
            DLSERecurrentStateWithReset,
            DLSERecurrentStateWithReset,
        ],
        DLSERecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    K: nn.Linear
    Q: nn.Linear
    V: nn.Linear

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(DLSESemigroup(recurrent_size))
        self.scan = semigroup_scan

        keys = jax.random.split(key, 3)

        self.K = nn.Linear(recurrent_size, recurrent_size, use_bias=False, key=keys[0])
        self.Q = nn.Linear(recurrent_size, recurrent_size, use_bias=False, key=keys[1])
        self.V = nn.Linear(recurrent_size, recurrent_size, use_bias=False, key=keys[2])

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> DLSERecurrentStateWithReset:
        emb, start = x
        k = 1 + jax.nn.elu(self.K(emb))
        v = 1 + jax.nn.elu(self.V(emb))
        kv = jnp.outer(k, v)
        return kv, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: DLSERecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        q = self.Q(emb)
        out = q @ (state / jnp.linalg.norm(state))
        return out

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> DLSERecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
