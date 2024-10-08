import jax


def debug_shape(x):
    return jax.tree.map(lambda x: {x.shape: x.dtype}, x)
