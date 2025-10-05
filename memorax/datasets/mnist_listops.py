import statistics

import jax
import jax.numpy as jnp
from datasets import load_dataset  # huggingface datasets

from memorax.train_utils import get_residual_memory_models


NUM_EPOCHS = 100
BATCH_SIZE = 32
SEQ_LEN = 784
NUM_LABELS = 10
SEED = 0


def draw_x():
    image = jnp.zeros((28, 28))
    coords = jnp.arange(28)
    x_image = image.at[coords, coords].set(192).at[coords, 27 - coords].set(192)
    x_image += jnp.roll(x_image, 1, axis=0)
    x_image += jnp.roll(x_image, -1, axis=0)
    x_image += jnp.roll(x_image, 1, axis=1)
    x_image += jnp.roll(x_image, -1, axis=1)
    x_image = (x_image > 0) * 192

    return x_image.astype(jnp.uint8)


def draw_plus():
    image = jnp.zeros((28, 28))
    center = (12, 13, 14, 15, 16)
    plus_image = image.at[center, :].set(192).at[:, center].set(192)
    return plus_image.astype(jnp.uint8)


def draw_minus():
    image = jnp.zeros((28, 28))
    center = (12, 13, 14, 15, 16)
    minus_image = image.at[center, :].set(192)
    return minus_image.astype(jnp.uint8)


def draw_lparen():
    image = jnp.zeros((28, 28))
    for y in range(4, 24):
        x = int(6 - 3 * jnp.cos((y - 4) * jnp.pi / 20))
        image = image.at[y, x].set(1)
    return image


def draw_rparen():
    image = jnp.zeros((28, 28))
    for y in range(4, 24):
        x = int(21 + 3 * jnp.cos((y - 4) * jnp.pi / 20))
        image = image.at[y, x].set(1)
    return image


operator_fns = {
    0: max,
    1: min,
    2: statistics.mean,
    3: statistics.median,
    4: statistics.mode,
}
operator_pixels = {
    0: draw_plus(),
    1: draw_minus(),
    2: draw_x(),
    3: draw_plus() + draw_x(),
    4: draw_minus() + draw_x(),
}


def make_equation_simple(x, y, op_idx, key):
    # x, y should be pre permuted
    pixel_ops = [operator_pixels[idx] for idx in op_idx.tolist()]
    fn_ops = [operator_fns[idx] for idx in op_idx.tolist()]

    result = y[0]
    results = [result]
    pixels = [x[0]]
    for xi, yi, pop, fop in zip(x[1:], y[1:], pixel_ops[1:], fn_ops[1:]):
        key, key1 = jax.random.split(key)
        result = fop(result, yi)
        results += [result]
        noisy_op = pop * jax.random.uniform(
            key, pop.shape, minval=0.8, maxval=1.2
        ) + 20 * jax.random.normal(key1, pop.shape)
        pixels += [noisy_op, xi]

    # Order should be number -> term
    # this way the model must read one term at a time instead of figuring
    # out the equation from the top k lines
    # [28, k * 28]
    pixels = jnp.concatenate(pixels, axis=0)
    return pixels, jnp.stack(results), result


def normalize_and_flatten(x):
    # batch, time, feature
    x = x.reshape(x.shape[0], -1, 1) / 255.0
    return x


def make_dataset(dataset_size=3, num_terms=5, key=jax.random.key(0), batch_size=32):
    dataset = load_dataset("mnist")
    x, y = jnp.array(dataset["train"]["image"]), jnp.array(dataset["train"]["label"])

    keys = jax.random.split(key, 3)
    data_idx = jax.random.randint(
        keys[0],
        (
            dataset_size,
            num_terms,
        ),
        0,
        x.shape[0],
    )
    ops_idx = jax.random.randint(
        keys[1],
        (
            dataset_size,
            num_terms,
        ),
        0,
        len(operator_fns),
    )
    datas = x[data_idx]
    labels = y[data_idx]
    xs = []
    ys = []
    y_inters = []
    for eq_idx in range(dataset_size):
        key, subkey = jax.random.split(key)
        x, y_inter, y = make_equation_simple(
            datas[eq_idx], labels[eq_idx], ops_idx[eq_idx], subkey
        )
        xs.append(x)
        ys.append(y)
        y_inters.append(y_inter)

    # x, y = jax.vmap(make_equation, in_axes=(None, None, 0, 0, None, None, 0))(x, y, data_idx, ops_idx, 5, operators, keys)
    x = normalize_and_flatten(x)
    y = jax.nn.one_hot(y, NUM_LABELS)
    batch_index = jnp.arange(x.shape[0], step=batch_size)
    return {
        "x": jnp.stack(x, axis=0),
        "y_intermediate": jnp.stack(y_inters, axis=0),
        "y": jnp.stack(y, axis=0),
        "batch_index": batch_index,
        "num_labels": NUM_LABELS,
    }


results = make_dataset()


key = jax.random.key(SEED)
models = get_residual_memory_models(input=1, hidden=256, output=NUM_LABELS, key=key)
