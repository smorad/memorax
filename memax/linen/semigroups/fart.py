from beartype.typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.mtypes import Input, StartFlag
from memax.linen.groups import BinaryAlgebra, Semigroup, Resettable
from memax.linen.gras import GRAS
from memax.linen.scans import semigroup_scan

FARTRecurrentState = Tuple[Float[Array, "Key Value"], Float[Array, "Key"]]
FARTRecurrentStateWithReset = Tuple[FARTRecurrentState, StartFlag]


def phi(x):
    return 1 + jax.nn.elu(x)


class FARTSemigroup(Semigroup):
    """The Fast AutoRegressive Transformer semigroup (recurrent update) from https://arxiv.org/abs/2006.16236"""

    recurrent_size: int

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FARTRecurrentState:
        return (
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
            jnp.zeros((self.recurrent_size)),
        )

    @jaxtyped(typechecker=typechecker)
    def zero_carry(self) -> FARTRecurrentState:
        return (
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
            jnp.zeros((self.recurrent_size)),
        )

    @jaxtyped(typechecker=typechecker)
    @nn.compact
    def __call__(
        self,
        carry: FARTRecurrentState,
        input: FARTRecurrentState,
    ) -> FARTRecurrentState:
        (
            kv_sum,
            k_sum,
        ) = carry
        kv, k = input
        kv_sum = kv_sum + kv
        k_sum = k_sum + k
        return kv_sum, k


class FART(GRAS):
    """The Fast AutoRegressive Transformer from https://arxiv.org/abs/2006.16236.

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int
    recurrent_size: int
    algebra: BinaryAlgebra

    def setup(self):
        self.K = nn.Dense(self.recurrent_size)
        self.Q = nn.Dense(self.recurrent_size)
        self.V = nn.Dense(self.recurrent_size)
        self.output = nn.Dense(self.hidden_size)

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FARTRecurrentStateWithReset:
        emb, start = x
        k = phi(self.K(emb))
        v: Float[Array, "Time Feat"] = self.V(emb)
        kv = jnp.outer(k, v)
        return (kv, k), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: FARTRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        (kv_sum, k_sum), reset_flag = h
        q = phi(self.Q(emb))
        out = q @ kv_sum / (1e-6 + jnp.dot(k_sum, q))
        return self.output(out) + emb

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FARTRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)

    @nn.nowrap
    def zero_carry(self) -> FARTRecurrentState:
        return self.algebra.zero_carry()

    @staticmethod
    def default_algebra(**kwargs):
        return Resettable(FARTSemigroup(**kwargs))

    @staticmethod
    def default_scan():
        return semigroup_scan
