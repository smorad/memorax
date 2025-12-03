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

FWPRecurrentState = Float[Array, "Key Value"]
FWPRecurrentStateWithReset = Tuple[FWPRecurrentState, StartFlag]


def phi(x, key=None):
    return 1 + jax.nn.elu(x)


class FWPSemigroup(Semigroup):
    """The Additive Fast Weight Programmer semigroup (recurrent update) 
    from https://arxiv.org/pdf/2508.08435"""

    recurrent_size: int

    def __init__(self, recurrent_size):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FWPRecurrentState:
        return jnp.zeros((self.recurrent_size, self.recurrent_size))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        carry: FWPRecurrentState,
        input: FWPRecurrentState,
    ) -> FWPRecurrentState:
        return carry + input


class FWP(GRAS):
    """The Additive Fast Weight Programmer from https://arxiv.org/pdf/2508.08435

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int
    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [FWPRecurrentStateWithReset, FWPRecurrentStateWithReset],
                FWPRecurrentStateWithReset,
            ],
            FWPRecurrentStateWithReset,
            FWPRecurrentStateWithReset,
        ],
        FWPRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    K: nn.Linear
    Q: nn.Linear
    V: nn.Linear
    output: nn.Linear

    def __init__(self, hidden_size, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.algebra = Resettable(FWPSemigroup(recurrent_size))
        self.scan = semigroup_scan

        keys = jax.random.split(key, 4)

        self.K = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[0])
        self.Q = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[1])
        self.V = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[2])
        self.output = nn.Linear(recurrent_size, hidden_size, key=keys[3])

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FWPRecurrentStateWithReset:
        emb, start = x
        k = phi(self.K(emb))
        v = self.V(emb)
        kv = jnp.outer(k, v)
        return kv, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: FWPRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        kv_sum, reset_flag = h
        q = phi(self.Q(emb))
        return self.output(kv_sum @ q)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> FWPRecurrentStateWithReset:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return self.algebra.initialize_carry(key)
