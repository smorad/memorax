"""
This module contains framework-agnostic utility functions used throughout the Memorax library.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Shaped


def debug_shape(x: jax.Array) -> str:
    """Pretty-prints the structure of an arbitrary pytree and the shape
    and dtype of its array nodes."""
    import equinox as eqx

    return eqx.tree_pprint(jax.tree.map(lambda x: {x.shape: x.dtype}, x))

class PositionalEncoding:
    """A simple sinusoidal positional encoding module."""

    def __init__(self, d_model: int):
        self.d_model = d_model

    def __call__(self, x: Shaped[Array, "Time Feat"], key: jax.random.PRNGKey) -> Shaped[Array, "Time Feat"]:
        time_indices = jnp.arange(x.shape[0])
        pos_encodings = jax.vmap(
            lambda t: transformer_positional_encoding(self.d_model, t)
        )(time_indices)
        return x + pos_encodings

def transformer_positional_encoding(
    d_model: int, time_index: Int[Array, ""]
) -> jnp.ndarray:
    """
    Generate a positional encoding vector for a given time index.

    Args:
        time_index (int): The time step index to encode.
        d_model (int): The dimensionality of the encoding vector.

    Returns:
        jnp.ndarray: A positional encoding vector of shape (d_model,).
    """
    position = time_index
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
    pos_encoding = jnp.zeros(d_model)
    pos_encoding = pos_encoding.at[0::2].set(jnp.sin(position * div_term))
    pos_encoding = pos_encoding.at[1::2].set(jnp.cos(position * div_term))
    return pos_encoding


def combine_and_right_align(
    left_array: Shaped[Array, "Time Feat"],
    left_mask: Shaped[Array, "Time"],
    right_array: Shaped[Array, "Time Feat"],
    right_mask: Shaped[Array, "Time"],
) -> Tuple[Shaped[Array, "Time Feat"], Shaped[Array, "Time"]]:
    """
    JIT-compatible function to combine two masked arrays and right-align the result.

    This is the JAX-native equivalent of:

    left = left_array[left_mask]
    right = right_array[right_mask]
    combined = jnp.concatenate([left, right])
    # Manual right-alignment:
    result = jnp.zeros_like(left_array)
    result = result.at[-len(combined):].set(combined)
    """
    # Ensure inputs are JAX arrays and masks are boolean
    left_mask = left_mask.astype(bool)
    right_mask = right_mask.astype(bool)

    # N is the static size of the output array.
    N = left_array.shape[0]

    # 1. Count the number of valid items in each input.
    num_valid_left = jnp.sum(left_mask)
    num_valid_right = jnp.sum(right_mask)

    # 2. Allocate slots based on priority.
    #    The right array gets priority, but can't take more than N slots.
    num_to_keep_right = jnp.minimum(num_valid_right, N)
    #    The left array gets the remaining space.
    space_for_left = N - num_to_keep_right
    num_to_keep_left = jnp.minimum(num_valid_left, space_for_left)

    # 3. Create new, "truncated" masks that select the rightmost valid items.
    #    We do this by ranking the valid items from the right using a flipped cumsum.
    right_ranks_left = jnp.flip(jnp.cumsum(jnp.flip(left_mask)))
    new_left_mask = (right_ranks_left <= num_to_keep_left) & left_mask

    right_ranks_right = jnp.flip(jnp.cumsum(jnp.flip(right_mask)))
    new_right_mask = (right_ranks_right <= num_to_keep_right) & right_mask

    # 4. Combine the arrays and their NEW truncated masks.
    combined_stack = jnp.concatenate([left_array, right_array])
    combined_mask = jnp.concatenate([new_left_mask, new_right_mask])

    # 5. Calculate destination indices for right-alignment.
    total_valid = num_to_keep_left + num_to_keep_right
    start_index = N - total_valid
    ranks = jnp.cumsum(combined_mask) - 1
    dest_indices = start_index + ranks

    # 6. Perform the standard scatter operation.
    new_stack = jnp.zeros_like(left_array)
    updates = jnp.where(combined_mask.reshape(-1, 1), combined_stack, 0)
    new_stack = new_stack.at[dest_indices].add(updates)

    # 7. Generate the final right-aligned mask.
    new_mask = jnp.arange(N) >= start_index

    return new_stack, new_mask
