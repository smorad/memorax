# THIS FILE IS SUPER HARDCODED AND JUST TEMPORARY
from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
import tqdm
from jaxtyping import Array, Shaped

import wandb
from memorax.datasets.sequential_mnist import get_dataset

# Conditional imports based on the framework
framework = "equinox"  # 'equinox' or 'flax'

if framework == "equinox":
    import equinox as eqx

    from memorax.magmas.gru import GRU
    from memorax.models.residual import ResidualModel

    # Import specific models as needed
    from memorax.semigroups.lru import LRU
elif framework == "flax":
    import flax.linen as nn
    from flax.core import FrozenDict
    from memorax_flax.magmas.gru import GRU
    from memorax_flax.models.residual import ResidualModel
    from memorax_flax.monoids.gilr import GILR
    from memorax_flax.monoids.lru import LRU
else:
    raise ValueError(f"Unknown framework: {framework}")

config = {
    "seed": 0,
    "num_epochs": 5,
    "batch_size": 16,
    "recurrent_size": 256,
    "num_layers": 2,
    "lr": 0.001,
}

# Dataset loading
dataset = get_dataset(batch_size=config["batch_size"])
key = jax.random.PRNGKey(config["seed"])

# Model initialization
if framework == "equinox":

    def get_residual_memory_models(
        input_size: int,
        hidden: int,
        output: int,
        num_layers: int = 2,
        *,
        key: jax.random.PRNGKey,
    ) -> Dict[str, eqx.Module]:
        layers = {
            # "lru": lambda recurrent_size, key: LRU(
            #     hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
            # ),
            "gru": lambda recurrent_size, key: GRU(
                recurrent_size=recurrent_size, key=key
            ),
        }
        return {
            name: ResidualModel(
                make_layer_fn=fn,
                input_size=input_size,
                recurrent_size=hidden,
                output_size=output,
                num_layers=num_layers,
                key=key,
            )
            for name, fn in layers.items()
        }

    models = get_residual_memory_models(
        input_size=1,
        hidden=config["recurrent_size"],
        output=dataset["num_labels"],
        key=key,
    )
elif framework == "flax":

    def get_residual_memory_models(
        hidden: int,
        output: int,
        num_layers: int = 2,
    ) -> Dict[str, nn.Module]:
        layers = {
            #     "lru": lambda recurrent_size: LRU(
            #         algebra=LRU.default_algebra(recurrent_size=recurrent_size),
            #         scan=LRU.default_scan(),
            #         hidden_size=recurrent_size,
            #         recurrent_size=recurrent_size
            #     ),
            #     "gilr": lambda recurrent_size: GILR(
            #     algebra=GILR.default_algebra(recurrent_size=recurrent_size), scan=GILR.default_scan(), recurrent_size=recurrent_size,
            # ),
            "gru": lambda recurrent_size: GRU(
                algebra=GRU.default_algebra(recurrent_size=recurrent_size),
                scan=GRU.default_scan(),
                recurrent_size=recurrent_size,
            ),
        }
        return {
            name: ResidualModel(
                make_layer_fn=fn,
                recurrent_size=hidden,
                output_size=output,
                num_layers=num_layers,
            )
            for name, fn in layers.items()
        }

    models = get_residual_memory_models(
        hidden=config["recurrent_size"], output=dataset["num_labels"]
    )
else:
    raise ValueError(f"Unknown framework: {framework}")


# Define shared utility functions
def cross_entropy(
    y_hat: Shaped[Array, "Batch ... Classes"], y: Shaped[Array, "Batch ... Classes"]
) -> Shaped[Array, "1"]:
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1))


def accuracy(
    y_hat: Shaped[Array, "Batch ... Classes"], y: Shaped[Array, "Batch ... Classes"]
) -> Shaped[Array, "1"]:
    return jnp.mean(jnp.argmax(y_hat, axis=-1) == jnp.argmax(y, axis=-1))


def add_batch_dim(x: Array, batch_size: int) -> Shaped[Array, "Batch ..."]:
    """Given an input `x`, add a new batch dimension of size `batch_size`."""
    return jnp.repeat(jnp.expand_dims(x, 0), batch_size, axis=0)


# Define framework-specific functions
if framework == "equinox":

    def loss_classify_terminal_output(
        model: eqx.Module,
        x: Shaped[Array, "Batch Time Feature"],
        y: Shaped[Array, "Batch Classes"],
    ) -> Tuple[Shaped[Array, "1"], Dict[str, Array]]:
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        starts = jnp.zeros((batch_size, seq_len), dtype=bool)
        h0 = jax.tree_map(
            partial(add_batch_dim, batch_size=batch_size), model.initialize_carry()
        )

        _, y_preds = eqx.filter_vmap(model)(h0, (x, starts))
        y_pred = y_preds[:, -1]

        loss = cross_entropy(y_pred, y)
        acc = accuracy(y_pred, y)
        return loss, {"loss": loss, "accuracy": acc}

    def update_model(
        model: eqx.Module,
        loss_fn: Callable,
        opt: optax.GradientTransformation,
        opt_state: optax.OptState,
        x: Shaped[Array, "Batch ..."],
        y: Shaped[Array, "Batch ..."],
        key=None,
    ) -> Tuple[eqx.Module, optax.OptState, Dict[str, Array]]:
        grads, loss_info = eqx.filter_grad(loss_fn, has_aux=True)(model, x, y)
        updates, opt_state = opt.update(
            grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_info

elif framework == "flax":

    def loss_classify_terminal_output(
        params: FrozenDict,
        x: Shaped[Array, "Batch Time Feature"],
        y: Shaped[Array, "Batch Classes"],
        init_carry_fn,
        model_apply_fn,
    ) -> Tuple[Shaped[Array, "1"], Dict[str, Array]]:
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        starts = jnp.zeros((batch_size, seq_len), dtype=bool)
        h0 = init_carry_fn(params)
        h0 = jax.tree_map(partial(add_batch_dim, batch_size=batch_size), h0)

        _, y_preds = jax.vmap(model_apply_fn, in_axes=[None, 0, 0])(
            params, h0, (x, starts)
        )
        y_pred = y_preds[:, -1]

        loss = cross_entropy(y_pred, y)
        acc = accuracy(y_pred, y)
        return loss, {"loss": loss, "accuracy": acc}

    def update_model(
        params: FrozenDict,
        loss_fn: Callable,
        opt: optax.GradientTransformation,
        opt_state: optax.OptState,
        x: Shaped[Array, "Batch ..."],
        y: Shaped[Array, "Batch ..."],
        key=None,
    ) -> Tuple[FrozenDict, optax.OptState, Dict[str, Array]]:
        grads, loss_info = jax.grad(loss_fn, has_aux=True)(params, x, y)
        updates, opt_state = opt.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_info

else:
    raise ValueError(f"Unknown framework: {framework}")

# Training loop
for name, model in models.items():
    print("Evaluating Model:", name)
    config["model"] = name
    wandb.init(project="memorax-debug", name=name)
    lr_schedule = optax.constant_schedule(config["lr"])
    opt = optax.chain(
        optax.zero_nans(),
        optax.adamw(lr_schedule),
    )

    if framework == "equinox":
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        loss_fn = loss_classify_terminal_output
    elif framework == "flax":
        dummy_x = dataset["x_train"][0]
        dummy_starts = jnp.zeros(dummy_x.shape[0], dtype=bool)
        dummy_h = model.zero_carry()
        params = model.init(key, dummy_h, (dummy_x, dummy_starts))
        opt_state = opt.init(params)
        initialise_carry_fn = partial(model.apply, method="initialize_carry")
        model_apply_fn = model.apply
        loss_fn = partial(
            loss_classify_terminal_output,
            init_carry_fn=initialise_carry_fn,
            model_apply_fn=model_apply_fn,
        )
    else:
        raise ValueError(f"Unknown framework: {framework}")

    key = jax.random.PRNGKey(config["seed"])

    for epoch in range(config["num_epochs"]):
        key, shuffle_key = jax.random.split(key)
        shuffle_idx = jax.random.permutation(shuffle_key, dataset["size"])
        x = dataset["x_train"][shuffle_idx]
        y = dataset["y_train"][shuffle_idx]
        pbar = tqdm.tqdm(range(x.shape[0] // config["batch_size"]))
        for update in pbar:
            key, subkey = jax.random.split(key)
            x_batch = x[
                update * config["batch_size"] : (update + 1) * config["batch_size"]
            ]
            y_batch = y[
                update * config["batch_size"] : (update + 1) * config["batch_size"]
            ]

            if framework == "equinox":
                model, opt_state, metrics = eqx.filter_jit(update_model)(
                    model, loss_fn, opt, opt_state, x_batch, y_batch, key=subkey
                )
            elif framework == "flax":
                params, opt_state, metrics = jax.jit(
                    update_model, static_argnames=("loss_fn", "opt")
                )(params, loss_fn, opt, opt_state, x_batch, y_batch, key=subkey)
            else:
                raise ValueError(f"Unknown framework: {framework}")

            mean_metrics = {k: jnp.mean(v).item() for k, v in metrics.items()}
            pbar.set_description(
                f"{name} epoch: {epoch}, "
                + ", ".join(f"{k}: {v:.4f}" for k, v in mean_metrics.items())
            )
            wandb.log({**mean_metrics, "epoch": epoch})
    wandb.finish()
