from beartype.typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, Semigroup, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import semigroup_scan

GDNRecurrentState = Tuple[
    Float[Array, "Key Value"],
    Float[Array, "Key Value"],
]
GDNRecurrentStateWithReset = Tuple[GDNRecurrentState, StartFlag]


def phi(x, key=None):
    # https://arxiv.org/pdf/2102.11174 uses relu
    # https://arxiv.org/pdf/2406.06484 uses silu
    return jax.nn.silu(x)

def psi(x, key=None):
    # https://arxiv.org/pdf/2102.11174 uses sigmoid
    # https://arxiv.org/pdf/2508.08435 suggests 2 * sigmoid
    return 2 * jax.nn.sigmoid(x)


class GDNSemigroup(Semigroup):
    """The Gated Delta Net semigroup (recurrent update) 
    from https://arxiv.org/pdf/2412.06464"""

    recurrent_size: int

    def __init__(self, recurrent_size):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GDNRecurrentState:
        return (
            jnp.eye(self.recurrent_size),
            jnp.zeros((self.recurrent_size, self.recurrent_size))
        )

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        carry: GDNRecurrentState,
        input: GDNRecurrentState,
    ) -> GDNRecurrentState:
        # Amazing resource: https://sustcsonglin.github.io/blog/2024/deltanet-2/
        # Based on Songlin's factorization
        M_i, X_i = carry
        M_j, X_j = input
        return M_j @ M_i, M_j @ X_i + X_j


class GDN(GRAS):
    """The Gated Delta Network from https://arxiv.org/pdf/2412.06464

    You might want to use this as a building block for a more complex model.
    """

    hidden_size: int
    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [GDNRecurrentStateWithReset, GDNRecurrentStateWithReset],
                GDNRecurrentStateWithReset,
            ],
            GDNRecurrentStateWithReset,
            GDNRecurrentStateWithReset,
        ],
        GDNRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    K: nn.Linear
    Q: nn.Linear
    V: nn.Linear
    w: nn.Linear
    alpha: nn.Linear
    output: nn.Linear

    def __init__(self, hidden_size, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.algebra = Resettable(GDNSemigroup(recurrent_size))
        self.scan = semigroup_scan

        keys = jax.random.split(key, 6)

        self.K = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[0])
        self.Q = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[1])
        self.V = nn.Linear(hidden_size, recurrent_size, use_bias=False, key=keys[2])
        self.w = nn.Linear(hidden_size, 1, key=keys[3])
        alpha = nn.Linear(hidden_size, 1, key=keys[4])
        # Initialize alpha bias to 4.0 so that sigmoid(alpha) is near 1.0 at init
        self.alpha = eqx.tree_at(
            lambda l: l.bias, alpha, jnp.full_like(alpha.bias, 4.0)
        )
        self.output = nn.Linear(recurrent_size, hidden_size, key=keys[5])

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GDNRecurrentStateWithReset:
        emb, start = x
        k = phi(self.K(emb))
        k = k / (jnp.linalg.norm(k) + 1e-6)  # normalize key
        v = self.V(emb)
        beta = psi(self.w(emb))
        alpha = jax.nn.sigmoid(self.alpha(emb))
        M = alpha * (jnp.eye(self.recurrent_size) - beta * jnp.outer(k, k))
        X = beta * jnp.outer(v, k)
        return (M, X), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: GDNRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        emb, start = x
        (M, X), reset_flag = h
        q = phi(self.Q(emb))
        q = q / (jnp.linalg.norm(q) + 1e-6)  # normalize query
        return self.output(X @ q)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> GDNRecurrentStateWithReset:
        # inputs should be of shape [*batch, time, feature]
        # recurrent states should be of shape [*batch, 1, feature]
        return self.algebra.initialize_carry(key)
