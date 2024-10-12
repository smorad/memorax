import equinox as eqx
import jax
import jax.numpy as jnp
import optax

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


def cross_entropy_loss(y_hat, y):
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1))


def accuracy(y_hat, y):
    return jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))


def model_update(
    model,
    opt,
    opt_state,
    loss_fn,
    x,
    y,
    x_transform=lambda x: x,
    y_transform=lambda y: y,
):
    x = x_transform(x)
    y = y_transform(y)
    grads, loss_info = eqx.filter_grad(loss_fn, has_aux=True)(model, x, y)
    updates, opt_state = opt.update(
        grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_info


def get_residual_memory_models(
    input: int, hidden: int, output: int, num_layers: int = 2
):
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
            key=jax.random.PRNGKey(0),
        )
        for name, fn in layers.items()
    }
