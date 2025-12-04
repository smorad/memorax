from beartype.typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped

from memax.equinox.groups import BinaryAlgebra, SetAction, Module, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import set_action_scan

SphericalRecurrentState = Float[Array, "Recurrent"]
SphericalRecurrentStateWithReset = Tuple[SphericalRecurrentState, StartFlag]


class SphericalMagma(SetAction):
    """
    The RotRNN (recurrent update) from https://arxiv.org/abs/2407.07239

    However, this is implemented in a less efficient manner (sequential)
    """

    recurrent_size: int
    project: nn.Linear
    initial_state: jax.Array

    def __init__(self, recurrent_size: int, sequence_length: int = 1024, *, key):
        self.recurrent_size = recurrent_size
        proj_size = int(self.recurrent_size * (self.recurrent_size - 1) / 2)
        self.project = nn.Linear(recurrent_size, proj_size, key=key)
        self.initial_state = jnp.ones((self.recurrent_size,))

    @jaxtyped(typechecker=typechecker)
    def rot(self, z: Array) -> Array:
        q = self.project(z)
        A = jnp.zeros((self.recurrent_size, self.recurrent_size))
        tri_idx = jnp.triu_indices_from(A, 1)
        A = A.at[tri_idx].set(q)
        A = A - A.T
        R = jax.scipy.linalg.expm(A)
        return R

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: SphericalRecurrentState, input: SphericalRecurrentState
    ) -> SphericalRecurrentState:
        R = self.rot(input)
        return R @ carry

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentState:
        return self.initial_state / jnp.linalg.norm(self.initial_state)


class Spherical(GRAS):
    """The Spherical RNN from https://arxiv.org/abs/2407.07239
    
    However, this is implemented in a less efficient manner (sequential)
    than the spherical semigroup.
    """

    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[
                [SphericalRecurrentStateWithReset, SphericalRecurrentStateWithReset],
                SphericalRecurrentStateWithReset,
            ],
            SphericalRecurrentStateWithReset,
            SphericalRecurrentStateWithReset,
        ],
        SphericalRecurrentStateWithReset,
    ]
    W_y: nn.Linear
    recurrent_size: int
    hidden_size: int

    def __init__(self, recurrent_size, hidden_size, key):
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        keys = jax.random.split(key)
        self.algebra = Resettable(SphericalMagma(recurrent_size, key=keys[0]))
        self.scan = set_action_scan
        self.W_y = nn.Linear(recurrent_size, hidden_size, key=keys[1])

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentStateWithReset:
        emb, start = x
        return emb, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: SphericalRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.hidden_size}"]:
        z, reset_flag = h
        emb, start = x
        return self.W_y(z)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SphericalRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
