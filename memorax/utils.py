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