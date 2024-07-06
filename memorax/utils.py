import jax


def debug_shape(x):
    print(jax.tree.map(lambda x: x.shape, x))


def relu(x, key):
    return jax.nn.relu(x)
