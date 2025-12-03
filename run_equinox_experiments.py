"""This script runs experiments training various recurrent memory models
on different datasets using Equinox. It serves as a reference implementation
for training and evaluating memax modules."""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm

import wandb
from memax.datasets.mnist_math import get_dataset as get_mnist_math
from memax.datasets.sequential_mnist import get_dataset as get_sequential_mnist
from memax.datasets.continuous_localization import get_rot_dataset, get_trans_dataset
from memax.equinox.train_utils import (
    get_residual_memory_models,
    loss_classify_terminal_output,
    loss_regress_terminal_output,
    update_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train recurrent memory models.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument(
        "--recurrent_size", type=int, help="Recurrent size of the model"
    )
    parser.add_argument("--num_layers", type=int, help="Number of layers in the model")
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
        default="memax-debug",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sequential_mnist",
        help="Dataset name (e.g., mnist_math, other_dataset)",
    )
    parser.add_argument(
        "--loss-function",
        type=str,
        default=None,
        help="Loss function to use (e.g., loss_classify_terminal_output, other_loss_fn)",
    )
    parser.add_argument("--models", type=str, nargs="+", default="all")
    return parser.parse_args()


def get_default_loss(dataset_name):
    defaults = {
        "sequential_mnist": "loss_classify_terminal_output",
        "mnist_math_5": "loss_classify_terminal_output",
        "sequential_rotation": "loss_regress_terminal_output"
    }
    if dataset_name in defaults:
        return defaults[dataset_name]
    else:
        raise ValueError(
            f"No default loss function defined for dataset: {dataset_name}"
        )

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
        "sequential_rotation": {
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
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    key = jax.random.PRNGKey(config.seed)

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

            model, opt_state, metrics = eqx.filter_jit(update_model)(
                model=model,
                loss_fn=loss_fn,
                opt=opt,
                opt_state=opt_state,
                x=x_batch,
                y=y_batch,
                key=subkey,
            )

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
    elif args.dataset_name == "sequential_rotation":
        dataset = get_rot_dataset()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    feature_in = dataset["x_test"].shape[-1]
    feature_out = dataset["y_test"].shape[-1] 

    # Select loss function
    if args.loss_function is None:
        loss_fn_name = get_default_loss(args.dataset_name)
    else:
        loss_fn_name = args.loss_fn

    if loss_fn_name =="loss_classify_terminal_output":
        loss_fn = loss_classify_terminal_output
    elif loss_fn_name == "loss_regress_terminal_output":
        loss_fn = loss_regress_terminal_output
    else:
        raise ValueError(f"Unknown loss function: {args.loss_function}")

    # Create model
    key = jax.random.PRNGKey(args.seed)
    models = get_residual_memory_models(
        input=feature_in,
        hidden=args.recurrent_size,
        output=feature_out,
        num_layers=args.num_layers,
        models=args.models,
        key=key,
    )

    # Run experiments
    for name, model in models.items():
        run_test(args, name, model, dataset, loss_fn)


if __name__ == "__main__":
    main()
