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

LinearRNNRecurrentState = Float[Array, "Hidden"]
LinearRNNRecurrentStateWithReset = Tuple[LinearRNNRecurrentState, StartFlag]


class LinearRNNSemigroup(Semigroup):
    """A simple (associative) linear recurrence"""
    recurrent_size: int

    def __init__(self, recurrent_size: int):
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LinearRNNRecurrentState:
        return jnp.zeros((self.recurrent_size,))

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: LinearRNNRecurrentState, input: LinearRNNRecurrentState
    ) -> LinearRNNRecurrentState:
        return carry + input


class LinearRecurrent(GRAS):
    """A simple Linear Recurrent layer.

    You might want to use this as a building block for a more complex model.
    """

    recurrent_size: int
    scan: Callable[
        [
            Callable[
                [LinearRNNRecurrentStateWithReset, LinearRNNRecurrentStateWithReset],
                LinearRNNRecurrentStateWithReset,
            ],
            LinearRNNRecurrentStateWithReset,
            LinearRNNRecurrentStateWithReset,
        ],
        LinearRNNRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    project: nn.Linear

    def __init__(self, recurrent_size, key):
        self.recurrent_size = recurrent_size
        self.algebra = Resettable(LinearRNNSemigroup(recurrent_size))
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
    ) -> LinearRNNRecurrentStateWithReset:
        emb, start = x
        z = emb
        return z, start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: LinearRNNRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        return self.project(state)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> LinearRNNRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
