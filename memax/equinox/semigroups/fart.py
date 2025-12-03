from beartype.typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, Semigroup, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import semigroup_scan

FARTRecurrentState = Tuple[Float[Array, "Key Value"], Float[Array, "Key"]]
FARTRecurrentStateWithReset = Tuple[FARTRecurrentState, StartFlag]


def phi(x, key=None):
    return 1 + jax.nn.elu(x)


class FARTSemigroup(Semigroup):
    """The Fast AutoRegressive Transformer semigroup (recurrent update) from https://arxiv.org/abs/2006.16236"""

    recurrent_size: int

    def __init__(self, recurrent_size):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FARTRecurrentState:
        return (
            jnp.zeros((self.recurrent_size, self.recurrent_size)),
            jnp.zeros((self.recurrent_size)),
        )

    @jaxtyped(typechecker=typechecker)
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
    scan: Callable[
        [
            Callable[
                [FARTRecurrentStateWithReset, FARTRecurrentStateWithReset],
                FARTRecurrentStateWithReset,
            ],
            FARTRecurrentStateWithReset,
            FARTRecurrentStateWithReset,
        ],
        FARTRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    K: nn.Linear
    Q: nn.Linear
    V: nn.Linear
    output: nn.Linear

    def __init__(self, hidden_size, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.algebra = Resettable(FARTSemigroup(recurrent_size))
        self.scan = semigroup_scan

        keys = jax.random.split(key, 4)

        self.K = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[0])
        self.Q = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[1])
        self.V = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[2])
        self.output = nn.Linear(recurrent_size, hidden_size, key=keys[3])

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
        # return self.ff(out + emb)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FARTRecurrentStateWithReset:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return self.algebra.initialize_carry(key)
