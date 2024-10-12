from typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Shaped

import memorax
from memorax.magmas.elman import Elman
from memorax.magmas.gru import GRU
from memorax.magmas.mgu import MGU
from memorax.magmas.spherical import Spherical
from memorax.models.residual import ResidualModel
from memorax.monoids.bayes import LogBayes
from memorax.monoids.fart import FART
from memorax.monoids.ffm import FFM
from memorax.monoids.gilr import GILR
from memorax.monoids.lrnn import LinearRecurrent
from memorax.monoids.lru import LRU
from memorax.monoids.mlstm import MLSTM


def cross_entropy(
    y_hat: Shaped[Array, "Batch ..."], y: Shaped[Array, "Batch ..."]
) -> Shaped[Array, "1"]:
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1))


def accuracy(y_hat, y):
    return jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))


def loss_classify_terminal_output(
    model: memorax.groups.Module,
    x: Shaped[Array, "Batch Time Feature"],
    y: Shaped[Array, "Batch Feature"],
) -> Tuple[Shaped[Array, "1"], Dict[str, Array]]:
    """Given a sequence of inputs x1, ..., xn and predicted outputs y1p, ..., y1n,
    return the cross entropy loss between the true yn and predicted y1n.

    Args:
        model: memorax.groups.Module
        x: (batch, time, in_feature)
        y: (batch, num_classes)

    Returns:
        loss: scalar
        info: dict
    """
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    assert (
        x.shape[0] == y.shape[0]
    ), f"batch size mismatch: {x.shape[0]} != {y.shape[0]}"
    assert x.ndim == 3, f"expected 3d input, got {x.ndim}d"
    assert y.ndim == 2, f"expected 2d input, got {y.ndim}d"

    starts = jnp.zeros((batch_size, seq_len), dtype=bool)
    h0 = model.initialize_carry((batch_size,))

    _, y_preds = eqx.filter_vmap(model)(h0, (x, starts))
    # batch, time, feature
    y_pred = y_preds[:, -1]

    loss = cross_entropy(y_pred, y)
    acc = accuracy(y_pred, y)
    return loss, {"loss": loss, "accuracy": acc}


def update_model(
    model: memorax.groups.Module,
    loss_fn: Callable,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    x: Shaped[Array, "Batch ..."],
    y: Shaped[Array, "Batch ..."],
    key=None,
) -> Tuple[memorax.groups.Module, optax.OptState, Dict[str, Array]]:
    """Update the model using the given loss function and optimizer."""
    grads, loss_info = eqx.filter_grad(loss_fn, has_aux=True)(model, x, y)
    updates, opt_state = opt.update(
        grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_info


@eqx.filter_jit
def scan_one_epoch(
    model: memorax.groups.Module,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: Callable,
    xs: Shaped[Array, "Datapoint ..."],
    ys: Shaped[Array, "Datapoint ..."],
    batch_size: int,
    batch_index: Shaped[Array, "Batch ..."],
    *,
    key: jax.random.PRNGKey,
) -> Tuple[memorax.groups.Module, optax.OptState, Dict[str, Array]]:
    """Train a single epoch using the scan operator. Functions as a dataloader and train loop."""
    assert (
        xs.shape[0] == ys.shape[0]
    ), f"batch size mismatch: {xs.shape[0]} != {ys.shape[0]}"
    params, static = eqx.partition(model, eqx.is_array)

    def get_batch(x, y, step):
        """Returns a specific batch of size `batch_size` from `x` and `y`."""
        start = step * batch_size
        x_batch = jax.lax.dynamic_slice_in_dim(x, start, batch_size, 0)
        y_batch = jax.lax.dynamic_slice_in_dim(y, start, batch_size, 0)
        return x_batch, y_batch

    def inner(carry, index):
        params, opt_state, key = carry
        x, y = get_batch(xs, ys, index)
        key = jax.random.split(key)[0]
        model = eqx.combine(params, static)
        params, opt_state, metrics = update_model(
            model, loss_fn, opt, opt_state, x, y, key=key
        )
        params, _ = eqx.partition(params, eqx.is_array)
        return (params, opt_state, key), metrics

    (params, opt_state, key), epoch_metrics = jax.lax.scan(
        inner,
        (params, opt_state, key),
        batch_index,
    )
    model = eqx.combine(params, static)
    return model, opt_state, epoch_metrics


def get_residual_memory_models(
    input: int,
    hidden: int,
    output: int,
    num_layers: int = 2,
    *,
    key: jax.random.PRNGKey,
) -> Dict[str, memorax.groups.Module]:
    layers = {
        # monoids
        "ffm": lambda recurrent_size, key: FFM(
            hidden_size=recurrent_size,
            trace_size=recurrent_size,
            context_size=recurrent_size,
            key=key,
        ),
        "fart": lambda recurrent_size, key: FART(
            hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "lru": lambda recurrent_size, key: LRU(
            hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "mlstm": lambda recurrent_size, key: MLSTM(
            recurrent_size=recurrent_size, key=key
        ),
        "linear_rnn": lambda recurrent_size, key: LinearRecurrent(
            recurrent_size=recurrent_size, key=key
        ),
        "gilr": lambda recurrent_size, key: GILR(
            recurrent_size=recurrent_size, key=key
        ),
        "log_bayes": lambda recurrent_size, key: LogBayes(
            recurrent_size=recurrent_size, key=key
        ),
        # magmas
        "gru": lambda recurrent_size, key: GRU(recurrent_size=recurrent_size, key=key),
        "elman": lambda recurrent_size, key: Elman(
            hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "ln_elman": lambda recurrent_size, key: Elman(
            hidden_size=recurrent_size,
            recurrent_size=recurrent_size,
            ln_variant=True,
            key=key,
        ),
        "spherical": lambda recurrent_size, key: Spherical(
            hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "mgu": lambda recurrent_size, key: MGU(recurrent_size=recurrent_size, key=key),
    }
    return {
        name: ResidualModel(
            make_layer_fn=fn,
            input_size=input,
            recurrent_size=hidden,
            output_size=output,
            num_layers=num_layers,
            key=key,
        )
        for name, fn in layers.items()
    }
