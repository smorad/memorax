import jax


def debug_shape(x):
    return jax.tree.map(lambda x: {x.shape: x.dtype}, x)


def relu(x, key=None):
    return jax.nn.relu(x)


def leaky_relu(x, key=None):
    return jax.nn.leaky_relu(x)
