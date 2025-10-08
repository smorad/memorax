import jax
import jax.numpy as jnp
from jaxtyping import Array, Int


def debug_shape(x):
    """Pretty-prints the structure of an arbitrary pytree and the shape
    and dtype of its array nodes. """
    import equinox as eqx
    return eqx.tree_pprint(jax.tree.map(lambda x: {x.shape: x.dtype}, x))

def leaky_hard_sigmoid(x, alpha=0.01):
    return jnp.maximum(jnp.minimum(1.0 + alpha * x, (x + 1) / 2), alpha * x)

def leaky_hard_tanh(x, alpha=0.01):
    return jnp.maximum(jnp.minimum(1.0 + alpha * x, x), alpha * x)

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

def combine_and_right_align(left_array, left_mask, right_array, right_mask):
    """
    JIT-compatible function to combine two masked arrays and right-align the result.

    This is the JAX-native equivalent of:

    # left = left_array[left_mask]
    # right = right_array[right_mask]
    # combined = jnp.concatenate([left, right])
    # # Manual right-alignment:
    # result = jnp.zeros_like(left_array)
    # result = result.at[-len(combined):].set(combined)
    """
    # Ensure inputs are JAX arrays and masks are boolean
    left_mask = left_mask.astype(bool)
    right_mask = right_mask.astype(bool)

    # Get the static size N from the input array.
    N = left_array.shape[0]

    # 1. Combine the inputs into single, larger arrays.
    #    This is a safe operation as shapes are static.
    combined_stack = jnp.concatenate([left_array, right_array], axis=0)
    combined_mask = jnp.concatenate([left_mask, right_mask], axis=0)

    # 2. Count the total number of valid elements.
    total_valid = jnp.sum(combined_mask)

    # 3. Calculate destination indices for right-alignment.
    #    The 'rank' is the 0-indexed position of each valid item (0th, 1st, ...).
    ranks = jnp.cumsum(combined_mask) - 1
    #    The destination block starts at index N - total_valid.
    start_index = N - total_valid
    dest_indices = start_index + ranks

    # 4. Perform the scatter operation.
    #    Start with a zeroed output array.
    new_stack = jnp.zeros_like(left_array)
    #    Prepare updates: valid items from combined_stack, zeros elsewhere.
    updates = jnp.where(combined_mask.reshape(-1, 1), combined_stack, 0)
    #    Scatter-add the updates to their right-aligned positions.
    new_stack = new_stack.at[dest_indices].add(updates)

    # 5. Generate the final right-aligned mask.
    #    The mask should be True for all indices >= start_index.
    new_mask = jnp.arange(N) >= start_index
    
    return new_stack, new_mask