"""
This module contains framework-agnostic utility functions used throughout the memax library.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Shaped, Float


def debug_shape(x: jax.Array) -> str:
    """Pretty-prints the structure of an arbitrary pytree and the shape
    and dtype of its array nodes."""
    import equinox as eqx

    return eqx.tree_pprint(jax.tree.map(lambda x: {x.shape: x.dtype}, x))

def apply_rope(keys: Float[Array, "Time Feat"], query: Float[Array, "Feat"]) -> Tuple[Float[Array, "Time Feat"], Float[Array, "Feat"]]:
    """
    Applies RoPE assuming contiguous time indices.
    
    Constraints:
    - Keys correspond to time steps [1, 2, ..., T]
    - Query corresponds to time step T
    
    Args:
        keys: Array of shape (T, F)
        query: Array of shape (F)
        
    Returns:
        keys_rope: Embedded keys (T, F)
        query_rope: Embedded query (F)
    """
    T, F = keys.shape
    assert F % 2 == 0, "Feature dimension must be even"
    
    # 1. Generate the Position Indices based on shape T
    # Keys are positions 1 to T
    key_indices = jnp.arange(1, T + 1, dtype=jnp.float32) 
    # Query is position T
    query_index = jnp.array(T, dtype=jnp.float32)

    # 2. Calculate RoPE Frequencies (Theta)
    # Standard formula: theta_i = 10000^(-2i/d)
    theta_indices = jnp.arange(0, F, 2)
    theta = 1.0 / (10000.0 ** (theta_indices / F)) # Shape: (F/2,)

    # 3. Create Complex Rotation Angles
    # Keys: (T, F/2) -> broadcast positions against frequencies
    key_angles = jnp.outer(key_indices, theta)
    # Query: (F/2,) -> scalar T against frequencies
    query_angle = query_index * theta

    # Calculate rotation vectors: e^(i * angle)
    key_rotators = jnp.exp(1j * key_angles)
    query_rotator = jnp.exp(1j * query_angle)

    # 4. Apply Rotation using Complex Numbers
    # Reshape (T, F) -> (T, F/2, 2) and convert to complex
    keys_complex = keys.reshape(T, -1, 2)
    keys_complex = keys_complex[..., 0] + 1j * keys_complex[..., 1]
    
    # Reshape (F) -> (F/2, 2) and convert to complex
    query_complex = query.reshape(-1, 2)
    query_complex = query_complex[..., 0] + 1j * query_complex[..., 1]

    # Multiply (rotate)
    keys_out_complex = keys_complex * key_rotators
    query_out_complex = query_complex * query_rotator

    # 5. Convert back to Real
    keys_rope = jnp.stack([keys_out_complex.real, keys_out_complex.imag], axis=-1).reshape(T, F)
    query_rope = jnp.stack([query_out_complex.real, query_out_complex.imag], axis=-1).reshape(F)

    return keys_rope, query_rope

def apply_sinusoidal_pe(keys: Float[Array, "Time Feat"], query: Float[Array, "Feat"], offset: Int[Array, ""] = jnp.array(0)):
    """
    Applies Standard Sinusoidal Positional Encoding with a temporal offset.
    
    Args:
        keys: Array of shape (T, F).
        query: Array of shape (F).
        offset: (int or scalar) The starting time offset. 
                If offset=10, keys map to positions 11...10+T.
        
    Returns:
        keys_pe: keys + PE(pos)
        query_pe: query + PE(pos)
    """
    T, F = keys.shape
    # Don't allow python ints which force recompile
    assert isinstance(offset, jax.Array), "Offset must be a JAX array scalar."
    assert F % 2 == 0, "Feature dimension F must be even."

    # 1. Define Positions with Offset
    # Cast to float32 immediately for instruction efficiency in sin/cos later
    offset_arr = jnp.array(offset, dtype=jnp.float32)
    
    # Keys: [1+offset, 2+offset, ..., T+offset]
    key_positions = jnp.arange(1, T + 1, dtype=jnp.float32) + offset_arr
    key_positions = key_positions[:, None] # Shape (T, 1) for broadcasting
    
    # Query: T + offset
    # corresponds to the last time step in this batch
    query_position = (jnp.array(T, dtype=jnp.float32) + offset_arr)

    # 2. Calculate Frequency Divisor
    # Note: Standard simplified implementation is just F. 
    # The exact Vaswani paper uses exp(arange(0, d, 2) * -(log(10000.0) / d))
    
    dim_indices = jnp.arange(0, F, 2, dtype=jnp.float32)
    div_term = jnp.exp(dim_indices * -(jnp.log(10000.0) / F)) # Shape (F/2,)
    
    # 3. Generate Embeddings for Keys
    # Broadcast (T, 1) * (F/2,) -> (T, F/2)
    k_args = key_positions * div_term
    
    # Interleave Sin/Cos for keys
    # Shape: (T, F/2, 2) -> Flatten to (T, F)
    pe_keys = jnp.stack([jnp.sin(k_args), jnp.cos(k_args)], axis=-1).reshape(T, F)
    
    # 4. Generate Embeddings for Query
    # Broadcast Scalar * (F/2,) -> (F/2,)
    q_args = query_position * div_term
    
    # Interleave Sin/Cos for query
    pe_query = jnp.stack([jnp.sin(q_args), jnp.cos(q_args)], axis=-1).reshape(F)

    # 5. Add to Inputs
    return keys + pe_keys, query + pe_query

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
