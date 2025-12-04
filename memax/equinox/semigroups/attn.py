from beartype.typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from equinox import nn
from jaxtyping import Array, Float, PRNGKeyArray, Shaped, jaxtyped, Bool, Int

from memax.equinox.groups import BinaryAlgebra, Semigroup, Resettable
from memax.equinox.gras import GRAS
from memax.mtypes import Input, StartFlag
from memax.equinox.scans import semigroup_scan
from memax.utils import apply_rope, apply_sinusoidal_pe, combine_and_right_align

AttentionRecurrentState = Tuple[Float[Array, "Window Recurrent"], Float[Array, "Window Recurrent"], Bool[Array, "Window"], Int[Array, "Window"]]
AttentionRecurrentStateWithReset = Tuple[AttentionRecurrentState, StartFlag]


class AttentionSemigroup(Semigroup):
    """A sliding window attention semigroup example.

    See the Stack semigroup for how to implement sliding windows.
    """
    window_size: int
    recurrent_size: int

    def __init__(self, recurrent_size: int, window_size: int):
        self.window_size = window_size 
        self.recurrent_size = recurrent_size

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> AttentionRecurrentState:
        key = jnp.zeros((self.window_size, self.recurrent_size))
        value = jnp.zeros((self.window_size, self.recurrent_size))
        # Valid (non-pad) mask
        mask = jnp.zeros((self.window_size,), dtype=bool)
        ts = jnp.zeros((self.window_size), dtype=jnp.int32)
        return (key, value, mask, ts)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, carry: AttentionRecurrentState, input: AttentionRecurrentState
    ) -> AttentionRecurrentState:
        # We would like to do the below
        # But cannot due to concretization error
        # Caused by dynamic indexing (mask)
        #
        # left = cwindow[cmask]
        # right = window[mask]

        # mleft = cmask[cmask]
        # mright = mask[mask]

        # window = jnp.concatenate([left, right])[-window_size:]
        # mask = jnp.concatenate([mleft, mright])[-window_size:]

        # So we use a tricky function instead
        ckey, cvalue, cmask, cts = carry
        key, value, mask, ts = input
        out_key, out_mask = combine_and_right_align(ckey, cmask, key, mask)
        out_value, _ = combine_and_right_align(cvalue, cmask, value, mask)
        out_ts = cts + ts
        return (out_key, out_value, out_mask, out_ts)


class Attention(GRAS):
    """Standard dot-product attention with a sliding window. This utilizes
    the Stack semigroup for maintaining a recurrent sliding window.
    """

    K: nn.Linear
    Q: nn.Linear
    V: nn.Linear
    recurrent_size: int
    window_size: int
    positional_embedding: Optional[str]
    scan: Callable[
        [
            Callable[
                [AttentionRecurrentStateWithReset, AttentionRecurrentStateWithReset],
                AttentionRecurrentStateWithReset,
            ],
            AttentionRecurrentStateWithReset,
            AttentionRecurrentStateWithReset,
        ],
        AttentionRecurrentStateWithReset,
    ]
    algebra: BinaryAlgebra


    def __init__(self, recurrent_size: int, window_size: int, positional_embedding: Optional[str], key):
        """Standard dot-product attention with a sliding window. 
        Arguments:
            recurrent_size: The size of the attention embeddings.
            window_size: The size of the attention window (context length).
            rope: Whether to use RoPE embeddings (False means no embeddings).
        """
        assert positional_embedding in [None, "rope", "alibi"], "positional_embedding must be one of None, 'rope', or 'alibi'"
        self.recurrent_size = recurrent_size
        self.window_size = window_size
        self.positional_embedding = positional_embedding
        self.algebra = Resettable(AttentionSemigroup(recurrent_size, window_size=window_size))
        self.scan = semigroup_scan
        keys = jax.random.split(key, 5)
        self.K = nn.Linear(recurrent_size, recurrent_size, use_bias=False, key=keys[0])
        self.Q = nn.Linear(recurrent_size, recurrent_size, use_bias=False, key=keys[1])
        self.V = nn.Linear(recurrent_size, recurrent_size, key=keys[2])

    @jaxtyped(typechecker=typechecker)
    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> AttentionRecurrentStateWithReset:
        emb, start = x
        # Add Attention dim for concat
        mask = jnp.concatenate([
            jnp.zeros((self.window_size - 1), dtype=bool),
            jnp.ones((1,), dtype=bool)
        ])
        k = self.K(emb)
        v = self.V(emb)
        key = jnp.concatenate([
            jnp.zeros((self.window_size - 1, *emb.shape), dtype=emb.dtype),
            k.reshape(1, -1)
        ])
        value = jnp.concatenate([
            jnp.zeros((self.window_size - 1, *emb.shape), dtype=emb.dtype),
            v.reshape(1, -1)
        ])
        ts = jnp.ones((self.window_size), dtype=jnp.int32)
        return (key, value, mask, ts), start

    @jaxtyped(typechecker=typechecker)
    def backward_map(
        self,
        h: AttentionRecurrentStateWithReset,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Float[Array, "{self.recurrent_size}"]:
        emb, start = x
        state, reset_carry = h
        K, V, mask, ts = state
        q = self.Q(emb)

        # B = batch size
        # S = length of the key/value (source)
        # T = length of the query (target)
        # N = number of attention heads
        # H = dimensions of each attention head
        # K = number of key/value heads
        # G = number of groups, which equals to N // K
        n, k, t, s, h = 1, 1, 1, self.window_size, self.recurrent_size
        bias = None
        if self.positional_embedding == "alibi":
            m = 2 ** -8
            # T-1 to 0
            bias = m * (ts[0] + jnp.arange(-s + 1, 1))
        elif self.positional_embedding == "rope":
            K, q = apply_rope(K, q)

        mask = mask.reshape(n, t, s)
        bias = bias if bias is None else bias.reshape(n, t, s)
        K = K.reshape(s, k, h)
        q = q.reshape(t, n, h) # Only for current timestep
        V = V.reshape(s, k, h)
        z = jax.nn.dot_product_attention(q, K, V, mask=mask, bias=bias)
        return z.reshape(-1)

    @jaxtyped(typechecker=typechecker)
    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> AttentionRecurrentStateWithReset:
        return self.algebra.initialize_carry(key)
