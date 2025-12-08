from beartype.typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped, Bool

from memax.equinox.groups import BinaryAlgebra, Semigroup, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import semigroup_scan
from memax.utils import combine_and_right_align

StackRecurrentState = Tuple[Float[Array, "Stack Recurrent"], Bool[Array, "Stack"]]
StackRecurrentStateWithReset = Tuple[StackRecurrentState, StartFlag]


class StackSemigroup(Semigroup):
    """A sliding window semigroup example.

    This allows you to define recurrent function that, for example,
    rely on the two (or more) most recent inputs through a sliding window.

    Applications include dot-product attention, RWKV, etc.
    """
    stack_size: int
    recurrent_size: int

    def __init__(self, recurrent_size: int, stack_size: int):
        self.stack_size = stack_size
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> StackRecurrentState:
        stack = jnp.zeros((self.stack_size, self.recurrent_size))
        #return stack
        # Valid (non-pad) mask
        mask = jnp.zeros((self.stack_size,), dtype=bool)
        return (stack, mask)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: StackRecurrentState, input: StackRecurrentState
    ) -> StackRecurrentState:
        # We would like to do the below
        # But cannot due to concretization error
        # Caused by dynamic indexing (mask)
        #
        # left = cstack[cmask]
        # right = stack[mask]

        # mleft = cmask[cmask]
        # mright = mask[mask]

        # stack = jnp.concatenate([left, right])[-stack_size:]
        # mask = jnp.concatenate([mleft, mright])[-stack_size:]

        # So we use a tricky function instead
        cstack, cmask = carry
        stack, mask = input
        out_stack, out_mask = combine_and_right_align(cstack, cmask, stack, mask)
        return (out_stack, out_mask)


class Stack(GRAS):
    """A fairly straightforward "recurrent" model that merely keeps a sliding
    window of the past. You may use this model as a blueprint to implement 
    "less recurrent" models that rely on more than the current input and recurrent state.

    Applications include dot-product attention, RWKV, etc.
    """

    recurrent_size: int
    stack_size: int
    scan: Callable[
        [
            Callable[
                [StackRecurrentStateWithReset, StackRecurrentStateWithReset],
                StackRecurrentStateWithReset,
            ],
            StackRecurrentStateWithReset,
            StackRecurrentStateWithReset,
        ],
        StackRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra

    g: nn.Sequential

    def __init__(self, recurrent_size, window_size, key):
        self.recurrent_size = recurrent_size
        self.stack_size = window_size
        self.algebra = Resettable(StackSemigroup(recurrent_size, stack_size=window_size))
        self.scan = semigroup_scan
        self.g = nn.Linear(recurrent_size * window_size, recurrent_size, key=key)

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> StackRecurrentStateWithReset:
        emb, start = x
        # Add stack dim for concat
        #return emb.reshape(1, -1), start
        mask = jnp.concatenate([
            jnp.zeros((self.stack_size - 1), dtype=bool),
            jnp.ones((1,), dtype=bool)
        ])
        emb = jnp.concatenate([
            jnp.zeros((self.stack_size - 1, *emb.shape), dtype=emb.dtype),
            emb.reshape(1, -1)
        ])
        return (emb, mask), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: StackRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        z, mask = state
        # You can do something more intelligent with masking if needed
        z = self.g(z.reshape(-1))
        return z

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> StackRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
