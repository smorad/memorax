import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm
from datasets import load_dataset  # huggingface datasets

from memorax.train_utils import (
    get_residual_memory_models,
    loss_classify_terminal_output,
    scan_one_epoch,
)

NUM_EPOCHS = 100
BATCH_SIZE = 32
SEQ_LEN = 784
NUM_LABELS = 10
SEED = 0


def normalize_and_flatten(x):
    # batch, time, feature
    x = x.reshape(x.shape[0], -1, 1) / 255.0
    return x


dataset = load_dataset("mnist")
x = jnp.array(dataset["train"]["image"])
y = jnp.array(dataset["train"]["label"])

x = normalize_and_flatten(x)
y = jax.nn.one_hot(y, NUM_LABELS)
# batch_index = jnp.arange(jnp.ceil(x.shape[0] / BATCH_SIZE).astype(int))
batch_index = jnp.arange(x.shape[0], step=BATCH_SIZE)

key = jax.random.key(SEED)
models = get_residual_memory_models(input=1, hidden=256, output=NUM_LABELS, key=key)

for name, model in models.items():
    lr_schedule = optax.constant_schedule(0.001)
    opt = optax.chain(
        optax.zero_nans(),
        optax.adamw(lr_schedule),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    key = jax.random.key(SEED)

    pbar = tqdm.tqdm(range(NUM_EPOCHS))
    for epoch in pbar:
        key, shuffle_key = jax.random.split(key)
        shuffle_idx = jax.random.permutation(shuffle_key, x.shape[0])
        x = x[shuffle_idx]
        y = y[shuffle_idx]
        # Create batches
        # x = batch(x, BATCH_SIZE)
        # y = batch(y, BATCH_SIZE)
        model, opt_state, loss_info = scan_one_epoch(
            model=model,
            opt=opt,
            opt_state=opt_state,
            loss_fn=loss_classify_terminal_output,
            xs=x,
            ys=y,
            batch_size=BATCH_SIZE,
            batch_index=batch_index,
            key=key,
        )
        pbar.set_description(
            f"{name} epoch: {epoch}, "
            + ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_info.items())
        )
