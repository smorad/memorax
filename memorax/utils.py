import jax


def debug_shape(x):
    return jax.tree.map(lambda x: {x.shape: x.dtype}, x)


def relu(x, key):
    return jax.nn.relu(x)
