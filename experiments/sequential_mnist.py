import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
import optax
import tqdm
from datasets import load_dataset  # huggingface datasets

from memorax.train_utils import (
    accuracy,
    cross_entropy_loss,
    get_residual_memory_models,
    model_update,
)

NUM_EPOCHS = 100
BATCH_SIZE = 32
SEQ_LEN = 784
NUM_LABELS = 10


def normalize_and_flatten(x):
    # batch, time, feature
    x = x.reshape(x.shape[0], -1, 1) / 255.0
    return x


dataset = load_dataset("mnist")
dataloader = jdl.DataLoader(
    dataset["train"], backend="jax", batch_size=32, shuffle=True, drop_last=False
)

models = get_residual_memory_models(input=1, hidden=256, output=NUM_LABELS)

for name, model in models.items():
    lr_schedule = optax.constant_schedule(0.001)
    opt = optax.chain(
        optax.zero_nans(),
        optax.adamw(lr_schedule),
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))


def loss_fn(model, x, y):
    starts = jnp.zeros((BATCH_SIZE, SEQ_LEN), dtype=bool)
    h0 = model.initialize_carry((BATCH_SIZE,))

    _, y_preds = eqx.filter_vmap(model)(h0, (x, starts))
    # batch, time, feature
    y_pred = y_preds[:, -1]

    loss = cross_entropy_loss(y_pred, y)
    acc = accuracy(y_pred, y)
    return loss, {"loss": loss, "accuracy": acc}


def update(model, opt_state, batch):
    x = normalize_and_flatten(batch["image"])
    y = jax.nn.one_hot(batch["label"], NUM_LABELS)
    grads, loss_info = eqx.filter_grad(loss_fn, has_aux=True)(model, x, y)
    updates, opt_state = opt.update(
        grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_info


for epoch in range(NUM_EPOCHS):
    pbar = tqdm.tqdm(dataloader)
    for batch in pbar:
        model, opt_state, loss_info = eqx.filter_jit(model_update)(
            model=model,
            opt=opt,
            opt_state=opt_state,
            loss_fn=loss_fn,
            x=batch["image"],
            y=batch["label"],
            x_transform=normalize_and_flatten,
            y_transform=lambda y: jax.nn.one_hot(y, NUM_LABELS),
        )
        pbar.set_description(
            f"{name} epoch: {epoch}, "
            + ", ".join(f"{k}: {v.item():.4f}" for k, v in loss_info.items())
        )
