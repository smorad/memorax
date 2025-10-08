from functools import partial
from beartype.typing import Callable, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from jaxtyping import Array, Shaped

from memorax.linen.set_actions.gru import GRU
from memorax.linen.models.residual import ResidualModel
from memorax.linen.semigroups.fart import FARTSemigroup, FART
from memorax.linen.semigroups.lru import LRUSemigroup, LRU
from memorax.linen.semigroups.s6d import S6DSemigroup, S6D


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


def accuracy(
    y_hat: Shaped[Array, "Batch ... Classes"], y: Shaped[Array, "Batch ... Classes"]
) -> Shaped[Array, "1"]:
    return jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))

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
    # h0 = jax.tree_map(partial(add_batch_dim, batch_size=batch_size), h0)
    h0 = add_batch_dim(h0, batch_size)

    _, y_preds = jax.vmap(model_apply_fn, in_axes=[None, 0, 0])(params, h0, (x, starts))
    y_pred = y_preds[:, -1]

    loss = cross_entropy(y_pred, y)
    acc = accuracy(y_pred, y)
    return loss, {"loss": loss, "accuracy": acc}

def get_semigroups(
    recurrent_size: int,
) -> Dict[str, FrozenDict]:
    """Returns a dictionary containing all implemented semigroups.
    
    This returns the operator used in the scan, not the full recurrent cell. 
    """
    return {
        "FART": FARTSemigroup(recurrent_size),
        "LRU": LRUSemigroup(recurrent_size),
        "S6D": S6DSemigroup(recurrent_size),
    }

def get_residual_memory_models(
    hidden: int,
    output: int,
    num_layers: int = 2,
) -> Dict:
    layers = {
        "FART": lambda recurrent_size: FART(
            algebra=FART.default_algebra(recurrent_size=round(recurrent_size**0.5)),
            scan=FART.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=round(recurrent_size**0.5),
        ),
        "LRU": lambda recurrent_size: LRU(
            algebra=LRU.default_algebra(recurrent_size=recurrent_size),
            scan=LRU.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=recurrent_size,
        ),
        "S6D": lambda recurrent_size: S6D(
            algebra=S6D.default_algebra(recurrent_size=recurrent_size),
            scan=S6D.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=recurrent_size,
        ),
        "GRU": lambda recurrent_size: GRU(
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
