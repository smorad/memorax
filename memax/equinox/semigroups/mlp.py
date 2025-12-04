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

MLPRecurrentState = Float[Array, "0"] # Empty array because MLP is not recurrent
MLPRecurrentStateWithReset = Tuple[MLPRecurrentState, StartFlag]


class MLPSemigroup(Semigroup):

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MLPRecurrentState:
        return jnp.zeros((0,))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: MLPRecurrentState, input: MLPRecurrentState
    ) -> MLPRecurrentState:
        return jnp.zeros((0,))


class MLP(GRAS):
    """A simple non-recurrent MLP. This is useful for debug purposes,
    or comparing a sequence model against a memory-free baseline.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [MLPRecurrentStateWithReset, MLPRecurrentStateWithReset],
                MLPRecurrentStateWithReset,
            ],
            MLPRecurrentStateWithReset,
            MLPRecurrentStateWithReset,
        ],
        MLPRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    project: nn.Linear

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(MLPSemigroup())
        self.scan = semigroup_scan

        keys = jax.random.split(key)

        self.project = nn.Sequential(
            [
                nn.Linear(recurrent_size, recurrent_size, key=keys[1]),
                nn.Lambda(jax.nn.leaky_relu),
            ]
        )

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MLPRecurrentStateWithReset:
        _, start = x
        return (jnp.zeros((0,)), start)

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: MLPRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        none, reset_carry = h
        return self.project(emb)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> MLPRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
