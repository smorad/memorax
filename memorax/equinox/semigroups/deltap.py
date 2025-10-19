from beartype.typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memorax.equinox.groups import BinaryAlgebra, Semigroup, Resettable
from memorax.equinox.gras import GRAS
from memorax.mtypes import Input, StartFlag
from memorax.equinox.scans import semigroup_scan

DeltaProductRecurrentState = Tuple[
    Float[Array, "Key Value"],
    Float[Array, "Key Value"],
]
DeltaProductRecurrentStateWithReset = Tuple[DeltaProductRecurrentState, StartFlag]


def phi(x, key=None):
    # https://arxiv.org/pdf/2102.11174 uses relu
    # https://arxiv.org/pdf/2406.06484 and 
    # https://arxiv.org/pdf/2502.10297 use silu
    return jax.nn.silu(x)

def psi(x, key=None):
    # https://arxiv.org/pdf/2102.11174 uses sigmoid
    # https://arxiv.org/pdf/2508.08435 suggests 2 * sigmoid
    return 2 * jax.nn.sigmoid(x)


class DeltaProductSemigroup(Semigroup):
    """The Fast Weight Programmer w/ Delta update semigroup (recurrent update) 
    from https://arxiv.org/pdf/2508.08435"""

    recurrent_size: int

    def __init__(self, recurrent_size):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> DeltaProductRecurrentState:
        return (
            jnp.eye(self.recurrent_size),
            jnp.zeros((self.recurrent_size, self.recurrent_size))
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        carry: DeltaProductRecurrentState,
        input: DeltaProductRecurrentState,
    ) -> DeltaProductRecurrentState:
        # Amazing resource: https://sustcsonglin.github.io/blog/2024/DeltaProduct-2/
        # Based on Songlin's factorization
        M_i, X_i = carry
        M_j, X_j = input
        return M_j @ M_i, M_j @ X_i + X_j


class DeltaProduct(GRAS):
    """The Additive Fast Weight Programmer w/ Delta update from https://arxiv.org/pdf/2508.08435

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int
    recurrent_size: int
    rank: int
    scan: Callable[
        [
            Callable[
                [DeltaProductRecurrentStateWithReset, DeltaProductRecurrentStateWithReset],
                DeltaProductRecurrentStateWithReset,
            ],
            DeltaProductRecurrentStateWithReset,
            DeltaProductRecurrentStateWithReset,
        ],
        DeltaProductRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    K: nn.Linear
    Q: nn.Linear
    V: nn.Linear
    w: nn.Linear
    output: nn.Linear

    def __init__(self, hidden_size: int, recurrent_size: int, rank: int, key):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.algebra = Resettable(DeltaProductSemigroup(recurrent_size))
        self.scan = semigroup_scan

        keys = jax.random.split(key, 5)

        self.K = nn.Linear(hidden_size, recurrent_size * rank, use_bias=False, key=keys[0])
        self.Q = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[1])
        self.V = nn.Linear(hidden_size, recurrent_size * rank, use_bias=False, key=keys[2])
        self.w = nn.Linear(hidden_size, self.rank, key=keys[3])
        self.output = nn.Linear(recurrent_size, hidden_size, key=keys[4])

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> DeltaProductRecurrentStateWithReset:
        emb, start = x
        k = phi(self.K(emb)).reshape(-1, self.rank)
        k = k / (1e-8 + jnp.linalg.norm(k, axis=0, keepdims=True))
        v = self.V(emb).reshape(-1, self.rank)
        beta = psi(self.w(emb)).reshape(-1, self.rank)

        beta_outer = lambda u, v, beta: beta * jnp.outer(u, v)

        # Outer returns tensor of (rank, recurrent, recurrent), sum reduces to (recurrent, recurrent)
        M = jnp.eye(self.recurrent_size) - jax.vmap(beta_outer, in_axes=-1)(k, k, beta).sum(axis=0)
        X = jax.vmap(beta_outer, in_axes=-1)(v, k, beta).sum(axis=0)
        return (M, X), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: DeltaProductRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        (M, X), reset_flag = h
        q = phi(self.Q(emb))
        return self.output(X @ q)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> DeltaProductRecurrentStateWithReset:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return self.algebra.initialize_carry(key)
