import jax
import jax.numpy as jnp
from datasets import load_dataset


def normalize_and_flatten(x):
    # batch, time, feature
    x = x.reshape(x.shape[0], -1, 1) / 255.0
    return x


def get_dataset():
    dataset = load_dataset("mnist")
    num_labels = 10

    x = jnp.array(dataset["train"]["image"])
    y = jnp.array(dataset["train"]["label"])
    x = normalize_and_flatten(x)
    y = jax.nn.one_hot(y, num_labels)

    test_x = jnp.array(dataset["test"]["image"])
    test_y = jnp.array(dataset["test"]["label"])
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