"""
This file contains the code for creating the mnist calculator dataset,
as well as the code that preprocesses the dataset before training
"""

import operator

import jax
import jax.numpy as jnp
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value, load_dataset

NUM_LABELS = 10
SEQ_LENS = [100, 1_000, 10_000, 100_000, 1_000_000]

FEATURES = Features(
    {
        "image": Image(),
        "image_flat": Sequence(Value("uint8")),
        "operand": Sequence(Value("int8")),
        "cumulative_result": Sequence(Value("int32")),
        "result": Value("int32"),
        "result_percentile": Value("uint8"),
    }
)


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


operator_fns = {
    0: operator.add,
    1: operator.sub,
    # 2: operator.mul,
}
operator_pixels = {
    0: draw_plus(),
    1: draw_minus(),
    # 2: draw_x(),
}

operators = [
    (draw_plus(), jnp.add),
    (draw_minus(), jnp.subtract),
    (draw_x(), jnp.multiply),
]  # x, y


def reduce_eq(carry, input):
    op_idx, digit = input
    out = operators[op_idx][1](carry, digit)
    return out, None


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
    pixels = jnp.concatenate(pixels, axis=0).astype(jnp.uint8)
    return pixels, jnp.stack(results), result


def normalize_and_flatten(x):
    # batch, time, feature
    x = x.reshape(x.shape[0], -1, 1) / 255.0
    return x


def generate_dataset(key, mnist, dataset_size=10, num_terms=5):
    x, y = jnp.array(mnist["image"]), jnp.array(mnist["label"])

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
    for eq_idx in tqdm.tqdm(range(dataset_size)):
        key, subkey = jax.random.split(key)
        x, y_inter, y = make_equation_simple(
            datas[eq_idx], labels[eq_idx], ops_idx[eq_idx], subkey
        )
        xs.append(x)
        ys.append(y)
        y_inters.append(y_inter)

    xs = jnp.stack(xs, axis=0)
    ys = jnp.stack(ys, axis=0)
    y_inters = jnp.stack(y_inters, axis=0)

    # xs = normalize_and_flatten(xs)

    # Make this a classification task
    percentiles = jnp.round(
        jnp.array([jnp.percentile(ys, pct) for pct in range(0, 100, 10)])
    )
    percentile_indices = jnp.searchsorted(percentiles, ys, side="right") - 1
    # ys_percentile = jax.nn.one_hot(percentile_indices, NUM_LABELS)
    # batch_index = jnp.arange(xs.shape[0], step=batch_size)
    return {
        "image": xs,
        "image_flat": xs.reshape(xs.shape[0], -1),
        "operand": labels,
        "cumulative_result": y_inters,
        "result": ys,
        "result_percentile": percentile_indices,
    }


def make_hf_datasets():
    mnist = load_dataset("mnist")
    train_dset = generate_dataset(jax.random.key(0), mnist["train"], 60_000)
    test_dset = generate_dataset(jax.random.key(1), mnist["test"], 10_000)
    train_dset = Dataset.from_dict(train_dset).cast(FEATURES)
    test_dset = Dataset.from_dict(test_dset).cast(FEATURES)
    return train_dset, test_dset


def upload_hf_datasets(length):
    train, test = make_hf_datasets()
    train.push_to_hub(f"smorad/mnist-math-{length}", split="train")
    test.push_to_hub(f"smorad/mnist-math-{length}", split="test")


def get_dataset(seq_len=5):
    assert seq_len in SEQ_LENS, f"Invalid length, must be in {SEQ_LENS}"
    dataset = load_dataset("smorad/mnist-math").with_format("np")
    num_labels = 10

    x = dataset["train"]["image_flat"]
    y = dataset["train"]["result_percentile"]
    x = normalize_and_flatten(x)
    y = jax.nn.one_hot(y, num_labels)

    test_x = dataset["test"]["image_flat"]
    test_y = dataset["test"]["result_percentile"]
    test_x = normalize_and_flatten(test_x)
    test_y = jax.nn.one_hot(test_y, num_labels)

    return {
        "x_train": x,
        "y_train": y,
        "x_test": test_x,
        "y_test": test_y,
        "num_labels": num_labels,
        "size": x.shape[0],
    }


if __name__ == "__main__":
    for seq_len in SEQ_LENS:
        upload_hf_datasets(seq_len)
