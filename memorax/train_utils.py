from beartype.typing import Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Shaped

import memorax
import memorax.groups
from memorax.set_actions.elman import Elman
from memorax.set_actions.gru import GRU
from memorax.set_actions.lstm import LSTM
from memorax.set_actions.mgu import MGU
from memorax.set_actions.spherical import Spherical
from memorax.models.residual import ResidualModel
from memorax.semigroups.fart import FART, FARTSemigroup
from memorax.semigroups.ffm import FFM, FFMSemigroup
from memorax.semigroups.lrnn import LinearRecurrent, LinearRNNSemigroup
from memorax.semigroups.lru import LRU, LRUSemigroup
from memorax.semigroups.nmax import NMax, NMaxSemigroup
from memorax.semigroups.spherical import PSpherical, PSphericalSemigroup
from memorax.semigroups.s6 import S6, S6Semigroup
from memorax.semigroups.s6_diag import S6Diagonal
from memorax.semigroups.mlp import MLP


def add_batch_dim(h, batch_size: int, axis: int = 0) -> Shaped[Array, "Batch ..."]:
    """Given an recurrent state (pytree) `h`, add a new batch dimension of size `batch_size`.

    E.g., add_batch_dim(h, 32) will return a new state with shape (32, *h.shape). The state will
    be repeated along the new batch dimension.
    """
    expand = lambda x: jnp.repeat(jnp.expand_dims(x, axis), batch_size, axis=axis)
    h = jax.tree.map(expand, h)
    return h

def cross_entropy(
    y_hat: Shaped[Array, "Batch ... Classes"], y: Shaped[Array, "Batch ... Classes"]
) -> Shaped[Array, "1"]:
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1))

def mse(
    y_hat: Shaped[Array, "Batch ... Feature"], y: Shaped[Array, "Batch ... Feature"]
) -> Shaped[Array, "1"]:
    return jnp.mean(jnp.linalg.norm(y - y_hat, axis=-1, ord=2))

def l1_error(
    y_hat: Shaped[Array, "Batch ... Feature"], y: Shaped[Array, "Batch ... Feature"]
) -> Shaped[Array, "1"]:
    return jnp.mean(jnp.linalg.norm(y - y_hat, axis=-1, ord=1))

def accuracy(
    y_hat: Shaped[Array, "Batch ... Classes"], y: Shaped[Array, "Batch ... Classes"]
) -> Shaped[Array, "1"]:
    return jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))

def loss_regress_terminal_output(
    model: memorax.groups.Module,
    x: Shaped[Array, "Batch Time Feature"],
    y: Shaped[Array, "Batch Classes"],
    key = None
) -> Tuple[Shaped[Array, "1"], Dict[str, Array]]:
    """Given a sequence of inputs x1, ..., xn and predicted outputs y1p, ..., y1n,
    return the mean square error loss between the true yn and predicted y1n.

    Args:
        model: memorax.groups.Module
        x: (batch, time, in_feature)
        y: (batch, out_feature)

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
    key, init_key, model_key = jax.random.split(key, 3)
    init_key = jax.random.split(init_key, batch_size)
    # TODO: These all initialize in the same state, probably do not want this
    h0 = eqx.filter_vmap(model.initialize_carry)(init_key)

    model_key = jax.random.split(model_key, batch_size)

    _, y_preds = eqx.filter_vmap(model)(h0, (x, starts), model_key)
    # batch, time, feature
    y_pred = y_preds[:, -1]

    loss = mse(y_pred, y)
    l1 = l1_error(y_pred, y)
    return loss, {"loss": loss, "l1_error": l1}

def loss_classify_terminal_output(
    model: memorax.groups.Module,
    x: Shaped[Array, "Batch Time Feature"],
    y: Shaped[Array, "Batch Classes"],
    key = None,
    decay = 0.01,
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
    key, init_key, model_key = jax.random.split(key, 3)
    init_key = jax.random.split(init_key, batch_size)
    # TODO: These all initialize in the same state, probably do not want this
    h0 = eqx.filter_vmap(model.initialize_carry)(init_key)

    model_key = jax.random.split(model_key, batch_size)

    h, y_preds = eqx.filter_vmap(model)(h0, (x, starts), model_key)
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
    grads, loss_info = eqx.filter_grad(loss_fn, has_aux=True)(model, x, y, key)
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
        # JIT this otherwise it takes ages to compile the epoch
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

def get_semigroups(
    recurrent_size: int,
    key: jax.random.PRNGKey,
) -> Dict[str, memorax.groups.Module]:
    """Returns a dictionary containing all implemented semigroups.
    
    This returns the operator used in the scan, not the full recurrent cell. 
    """
    return {
        "pspherical": PSphericalSemigroup(recurrent_size),
        "ffm": FFMSemigroup(recurrent_size, recurrent_size, recurrent_size, key=key),
        "fart": FARTSemigroup(recurrent_size),
        "linear_rnn": LinearRNNSemigroup(recurrent_size),
        "lru": LRUSemigroup(recurrent_size),
        "s6": S6Semigroup(recurrent_size),
        "nmax": NMaxSemigroup(recurrent_size),
    }

def get_residual_memory_models(
    input: int,
    hidden: int,
    output: int,
    num_layers: int = 2,
    models: str = "all",
    *,
    key: jax.random.PRNGKey,
) -> Dict[str, memorax.groups.Module]:
    """Returns a dictionary of models, correponding to all semigroups and set actions.
    
    This returns a dictionary of models, each consisting of multiple recurrent cells 
    with residual and DenseNet connections between them.
    """
    layers = {
        # for debug
        "MLP": lambda recurrent_size, key: MLP(
            recurrent_size=recurrent_size, key=key
        ),
        # semigroups
        "NMax": lambda recurrent_size, key: NMax(
            recurrent_size=recurrent_size, key=key
        ),
        "FART": lambda recurrent_size, key: FART(
           hidden_size=recurrent_size, recurrent_size=round(recurrent_size ** 0.5), key=key
        ),
        "FFM": lambda recurrent_size, key: FFM(
           hidden_size=recurrent_size, context_size=recurrent_size//4, trace_size=4, key=key
        ),
        "S6Diagonal": lambda recurrent_size, key: S6Diagonal(
            hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "S6": lambda recurrent_size, key: S6(
            hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "PSpherical": lambda recurrent_size, key: PSpherical(
            recurrent_size=round(recurrent_size ** 0.5),
            hidden_size=recurrent_size,
            key=key
        ),
        "LRU": lambda recurrent_size, key: LRU(
            hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "LinearRNN": lambda recurrent_size, key: LinearRecurrent(
            recurrent_size=recurrent_size, key=key
        ),
        # set actions
        "GRU": lambda recurrent_size, key: GRU(recurrent_size=recurrent_size, key=key),
        "Elman": lambda recurrent_size, key: Elman(
           hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "ElmanReLU": lambda recurrent_size, key: Elman(
           hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key, activation=jax.nn.relu,
        ),
        "Spherical": lambda recurrent_size, key: Spherical(
            hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
        ),
        "MGU": lambda recurrent_size, key: MGU(recurrent_size=recurrent_size, key=key),
        "LSTM": lambda recurrent_size, key: LSTM(recurrent_size=recurrent_size, key=key),
    }
    if models == "all":
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
    else:
        return {
            name: ResidualModel(
                make_layer_fn=layers[name],
                input_size=input,
                recurrent_size=hidden,
                output_size=output,
                num_layers=num_layers,
                key=key,
            )
            for name in models
        }