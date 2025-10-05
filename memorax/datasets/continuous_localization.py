import jax.numpy as jnp
import jax
from jax.scipy.spatial.transform import Rotation
from datasets import Dataset, Features, Array2D, load_dataset

SEQ_LENS = [100, 1_000, 10_000, 100_000, 1_000_000]

def step(carry, inputs):
    (x, rot) = carry
    (dx, drot) = inputs
    transform_x = jax.vmap(lambda rot_t, x_t, dx: rot_t.apply(dx) + x_t)
    transform_r = jax.vmap(lambda rot_t, drot: rot_t * drot)
    x = transform_x(rot, x, dx)
    rot = transform_r(rot, drot)
    return ((x, rot), (x, rot))


def generate_dataset(
        key,
        batch_size,
        num_steps
):
    spatial_dim = 3
    keys = jax.random.split(key, 4)
    x = jnp.zeros((batch_size, spatial_dim))
    rot = jax.vmap(Rotation.identity, axis_size=batch_size)()
    dx_mag = jax.random.exponential(keys[0], (num_steps, batch_size, 1)) * 2
    dx_dir = jax.random.uniform(keys[1], (num_steps, batch_size, 3), minval=-1., maxval=1.)
    dx = dx_dir / jnp.linalg.norm(dx_dir, axis=-1, keepdims=True) * dx_mag
    drot_dir = jax.random.uniform(keys[2], (num_steps, batch_size, spatial_dim), minval=-1, maxval=1)
    drot_mag = jax.random.uniform(keys[3], (num_steps, batch_size, 1), maxval=jnp.pi)
    drot_vec = drot_dir / jnp.linalg.norm(drot_dir, axis=-1, keepdims=True) * drot_mag
    drot = Rotation.from_rotvec(drot_vec)

    _, (x_abs, rot_abs) = jax.lax.scan(step, (x, rot), (dx, drot))

    drot_vec = drot.as_rotvec()
    rot_abs_vec = rot_abs.as_rotvec()

    inputs = jnp.concatenate([drot_vec, dx], axis=-1)
    outputs = jnp.concatenate([rot_abs_vec, x_abs], axis=-1)

    return {
        "inputs": jnp.permute_dims(inputs, (1,0,2)),
        "outputs": jnp.permute_dims(outputs, (1,0,2)),
        "delta_rotation": jnp.permute_dims(drot_vec, (1,0,2)),
        "delta_position": jnp.permute_dims(dx, (1,0,2)),
        "absolute_rotation": jnp.permute_dims(rot_abs_vec, (1,0,2)),
        "absolute_position": jnp.permute_dims(x_abs, (1,0,2)),
    }


def upload_hf_datasets(
    batch_size_train = 1e6,
    batch_size_test = 2e5,
    sequence_length = 20,
):
    name = f"bolt-lab/continuous-localization-{sequence_length}",
    batch_size_train = int(batch_size_train)
    batch_size_test = int(batch_size_test)
    FEATURES = Features({
        "inputs": Array2D(dtype='float32', shape=(sequence_length, 6)),
        "outputs": Array2D(dtype='float32', shape=(sequence_length, 6)),
        "delta_rotation": Array2D(dtype='float32', shape=(sequence_length, 3)),
        "delta_position": Array2D(dtype='float32', shape=(sequence_length, 3)),
        "absolute_rotation": Array2D(dtype='float32', shape=(sequence_length, 3)),
        "absolute_position": Array2D(dtype='float32', shape=(sequence_length, 3)),
    })
    key = jax.random.key(0)
    keys = jax.random.split(key, 2)
    train_dict = generate_dataset(keys[0], batch_size=batch_size_train, num_steps=sequence_length)
    test_dict = generate_dataset(keys[1], batch_size=batch_size_test, num_steps=sequence_length)
    train_dataset = Dataset.from_dict(train_dict).cast(features=FEATURES, batch_size=batch_size_train)
    test_dataset = Dataset.from_dict(test_dict).cast(features=FEATURES, batch_size=batch_size_test)
    train_dataset.push_to_hub(name, split="train")
    test_dataset.push_to_hub(name, split="test")

def get_rot_dataset(sequence_length=20):
    seq_lens = [20]
    assert sequence_length in seq_lens, f"Invalid sequenec length, must be one of {seq_lens}"
    num_labels = 10
    dataset = load_dataset(f"bolt-lab/continuous-localization-{sequence_length}")

    x = jnp.array(dataset["train"]["delta_rotation"])
    y = jnp.array(dataset["train"]["absolute_rotation"][-1])
    #y = jnp.array(dataset["train"]["absolute_rotation_percentile"])
    # TODO: One-hot based on abs of rotation angle?
    #y = jax.nn.one_hot(y, num_labels)

    test_x = jnp.array(dataset["test"]["delta_rotation"])
    #test_y = jnp.array(dataset["test"]["absolute_rotation_percentile"])
    test_y = jnp.array(dataset["test"]["absolute_rotation"][-1])
    #test_y = jax.nn.one_hot(test_y, num_labels)

    return {
        "x_train": x,
        "y_train": y,
        "x_test": test_x,
        "y_test": test_y,
        "size": x.shape[0],
        "num_labels": 10,
    }

def get_trans_dataset(sequence_length=20):
    seq_lens = [20, 1024]
    assert sequence_length in seq_lens, f"Invalid sequenec length, must be one of {seq_lens}"
    dataset = load_dataset(f"bolt-lab/continuous-localization-{sequence_length}")

    x = jnp.concatenate([
        jnp.array(dataset["train"]["delta_rotation"]),
        jnp.array(dataset["train"]["delta_translation"])
    ], axis=-1)
    y = jnp.array(dataset["train"]["absolute_translation_percentile"])

    test_x = jnp.concatenate([
        jnp.array(dataset["test"]["delta_rotation"]),
        jnp.array(dataset["test"]["delta_translation"])
    ], axis=-1)
    test_y = jnp.array(dataset["test"]["absolute_translation_percentile"])

    return {
        "x_train": x,
        "y_train": y,
        "x_test": test_x,
        "y_test": test_y,
        "size": x.shape[0],
        "num_labels": 10,
    }


if __name__ == "__main__":
    for seq_len in SEQ_LENS:
        upload_hf_datasets(batch_size_train=60_000, batch_size_test=10_000, sequence_length=seq_len)