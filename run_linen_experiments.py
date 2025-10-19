"""This script runs experiments training various recurrent memory models
on different datasets using Flax Linen. It serves as a reference implementation
for training and evaluating memorax modules."""
import argparse
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm

import wandb
from memorax.datasets.mnist_math import get_dataset as get_mnist_math
from memorax.datasets.sequential_mnist import get_dataset as get_sequential_mnist
from memorax.linen.train_utils import (
    get_residual_memory_models,
    loss_classify_terminal_output,
    update_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train recurrent memory models.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument(
        "--recurrent-size", type=int, help="Recurrent size of the model"
    )
    parser.add_argument("--num-layers", type=int, help="Number of layers in the model")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="memorax-debug",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sequential_mnist",
        help="Dataset name (e.g., mnist_math, other_dataset)",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default="loss_classify_terminal_output",
        help="Loss function to use (e.g., loss_classify_terminal_output, other_loss_fn)",
    )
    parser.add_argument("--models", type=str, nargs="+")
    return parser.parse_args()


def get_default_hyperparameters(dataset_name):
    defaults = {
        "mnist_math_5": {
            "num_epochs": 5,
            "batch_size": 16,
            "recurrent_size": 256,
            "num_layers": 2,
            "lr": 0.0001,
        },
        "sequential_mnist": {
            "num_epochs": 5,
            "batch_size": 16,
            "recurrent_size": 256,
            "num_layers": 2,
            "lr": 0.0001,
        },
        # Add more datasets and their default hyperparameters here
    }
    if dataset_name in defaults:
        return defaults[dataset_name]
    else:
        raise ValueError(
            f"No default hyperparameters defined for dataset: {dataset_name}"
        )


def update_config_with_defaults(args):
    defaults = get_default_hyperparameters(args.dataset_name)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def run_test(config, name, model, dataset, loss_fn):
    if config.use_wandb:
        wandb.init(project=config.project_name, name=name)

    lr_schedule = optax.constant_schedule(config.lr)
    opt = optax.chain(
        optax.zero_nans(),
        optax.adamw(lr_schedule),
    )
    key = jax.random.PRNGKey(config.seed)

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

    for epoch in range(config.num_epochs):
        key, shuffle_key = jax.random.split(key)
        shuffle_idx = jax.random.permutation(shuffle_key, dataset["size"])
        x = dataset["x_train"][shuffle_idx]
        y = dataset["y_train"][shuffle_idx]
        pbar = tqdm.tqdm(range(x.shape[0] // config.batch_size))

        for update in pbar:
            key, subkey = jax.random.split(key)
            x_batch = x[update * config.batch_size : (update + 1) * config.batch_size]
            y_batch = y[update * config.batch_size : (update + 1) * config.batch_size]

            params, opt_state, metrics = jax.jit(
                update_model, static_argnames=("loss_fn", "opt")
            )(params, loss_fn, opt, opt_state, x_batch, y_batch, key=subkey)

            mean_metrics = {k: jnp.mean(v).item() for k, v in metrics.items()}
            pbar.set_description(
                f"{name} epoch: {epoch}, "
                + ", ".join(f"{k}: {v:.4f}" for k, v in mean_metrics.items())
            )
            if config.use_wandb:
                wandb.log({**mean_metrics, "epoch": epoch})

    if config.use_wandb:
        wandb.finish()


def main():
    args = parse_args()
    update_config_with_defaults(args)

    # Dynamically load dataset
    if args.dataset_name == "mnist_math_5":
        dataset = get_mnist_math(5)
    elif args.dataset_name == "sequential_mnist":
        dataset = get_sequential_mnist()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    # Dynamically select loss function
    if args.loss_function == "loss_classify_terminal_output":
        loss_fn = loss_classify_terminal_output
    else:
        raise ValueError(f"Unknown loss function: {args.loss_function}")

    models = get_residual_memory_models(
        hidden=args.recurrent_size,
        output=dataset["num_labels"],
        num_layers=args.num_layers,
    )

    for name, model in models.items():
        run_test(args, name, model, dataset, loss_fn)


if __name__ == "__main__":
    main()
