import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from equinox import nn

from memorax.magmas.elman import Elman
from memorax.magmas.gru import GRU
from memorax.magmas.mgu import MGU
from memorax.magmas.spherical import Spherical
from memorax.memoroid import Memoroid
from memorax.monoids.bayes import LogBayes
from memorax.monoids.dlse import DLSE
from memorax.monoids.fart import FART
from memorax.monoids.ffm import FFM
from memorax.monoids.lrnn import LinearRNN
from memorax.monoids.lru import LRU
from memorax.utils import debug_shape, relu


def ce_loss(y_hat, y):
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1))


def test_forward(model, num_seqs=5, seq_len=20, input_dims=4):
    timesteps = num_seqs * seq_len
    seq_idx = jnp.array([seq_len * i for i in range(num_seqs)])
    start = jnp.zeros((timesteps,), dtype=bool).at[seq_idx].set(True)
    h = model.initialize_carry()
    x = jax.random.randint(jax.random.PRNGKey(0), (timesteps,), 0, input_dims - 1)
    x = jax.nn.one_hot(x, input_dims - 1)
    x = jnp.concatenate([x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
    y = jnp.repeat(x[seq_idx, :-1], seq_len, axis=0)

    _, y_hat = model(h, (x, start))
    loss = ce_loss(y_hat, y)
    accuracy = jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1)
    assert y_hat.shape == y.shape


def train_initial_input(
    model, epochs=4000, num_seqs=5, seq_len=20, input_dims=4, eval_model=True
):
    timesteps = num_seqs * seq_len
    seq_idx = jnp.array([seq_len * i for i in range(num_seqs)])
    start = jnp.zeros((timesteps,), dtype=bool).at[seq_idx].set(True)

    opt = optax.adam(learning_rate=3e-3)
    state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def error(model, key):
        h = model.initialize_carry()
        x = jax.random.randint(key, (timesteps,), 0, input_dims - 1)
        x = jax.nn.one_hot(x, input_dims - 1)
        x = jnp.concatenate([x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
        y = jnp.repeat(x[seq_idx, :-1], seq_len, axis=0)

        _, y_hat = model(h, (x, start))
        loss = ce_loss(y_hat, y)
        accuracy = jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))
        return loss, {"loss": loss, "accuracy": accuracy}

    loss_fn = eqx.filter_jit(eqx.filter_grad(error, has_aux=True))
    key = jax.random.PRNGKey(0)
    losses = []
    accuracies = []
    for epoch in range(epochs):
        key, _ = jax.random.split(key)
        grads, loss_info = loss_fn(model, key)
        updates, state = jax.jit(opt.update)(grads, state)
        model = eqx.apply_updates(model, updates)
        # print(
        #    f"Step {epoch+1}, Loss: {loss_info['loss']:0.2f}, Accuracy: {loss_info['accuracy']:0.2f}"
        # )
        losses.append(loss_info["loss"])
        accuracies.append(loss_info["accuracy"])

    if not eval_model:
        return jnp.stack(losses), jnp.stack(accuracies)

    key, _ = jax.random.split(key)
    h = model.initialize_carry()
    x = jax.random.randint(key, (timesteps,), 0, input_dims - 1)
    x = jax.nn.one_hot(x, input_dims - 1)
    x = jnp.concatenate([x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
    y = jnp.repeat(x[seq_idx, :-1], seq_len, axis=0)

    state, y_hat = model(h, (x, start))
    y_hat = jnp.squeeze(y_hat)
    y = jnp.squeeze(y)
    loss = ce_loss(y_hat, y)
    accuracy = jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))

    return jnp.stack(losses), jnp.stack(accuracies)


def get_memory_models(hidden: int, input: int, output: int):
    return {
        # "dlse": DLSE(
        #     input_size=input,
        #     hidden_size=hidden,
        #     output_size=output,
        #     num_layers=2,
        #     key=jax.random.PRNGKey(0),
        # ),
        "ffm": FFM(
            input_size=input,
            hidden_size=hidden,
            output_size=output,
            num_layers=2,
            key=jax.random.PRNGKey(0),
        ),
        "fart": FART(
            input_size=input,
            hidden_size=hidden,
            output_size=output,
            num_layers=2,
            key=jax.random.PRNGKey(0),
        ),
        "spherical": Spherical(
            input_size=input,
            hidden_size=hidden,
            output_size=output,
            num_layers=2,
            key=jax.random.PRNGKey(0),
        ),
        "lru": LRU(
            input_size=input,
            hidden_size=hidden,
            output_size=output,
            num_layers=2,
            key=jax.random.PRNGKey(0),
        ),
        "elman": Elman(
            input_size=input,
            hidden_size=hidden,
            output_size=output,
            num_layers=1,
            key=jax.random.PRNGKey(0),
        ),
        "ln_elman": Elman(
            input_size=input,
            hidden_size=hidden,
            output_size=output,
            num_layers=2,
            ln_variant=True,
            key=jax.random.PRNGKey(0),
        ),
        "mgu": MGU(
            input_size=input,
            hidden_size=hidden,
            output_size=output,
            num_layers=1,
            key=jax.random.PRNGKey(0),
        ),
        "gru": GRU(
            input_size=input,
            hidden_size=hidden,
            output_size=output,
            num_layers=1,
            key=jax.random.PRNGKey(0),
        ),
        "linear_rnn": LinearRNN(
            input_size=input,
            hidden_size=hidden,
            output_size=output,
            num_layers=2,
            key=jax.random.PRNGKey(0),
        ),
        "log_bayes": LogBayes(
            input_size=input,
            hidden_size=hidden,
            output_size=output,
            num_layers=2,
            key=jax.random.PRNGKey(0),
        ),
    }


def get_desired_accuracies():
    return {
        # "dlse": 0.999,
        "ffm": 0.999,
        "fart": 0.999,
        "spherical": 0.996,
        "lru": 0.999,
        "mgu": 0.999,
        "gru": 0.999,
        "linear_rnn": 0.999,
        "elman": 0.60,
        "ln_elman": 0.60,
        "log_bayes": 0.999,
    }


def test_forwards():
    test_size = 4
    hidden = 4
    for model_name, model in get_memory_models(
        hidden, test_size, test_size - 1
    ).items():
        test_forward(model)


def test_classify():
    test_size = 4
    hidden = 4
    for model_name, model in get_memory_models(
        hidden, test_size, test_size - 1
    ).items():
        losses, accuracies = train_initial_input(model)
        losses = losses[-100:].mean()
        accuracies = accuracies[-100:].mean()
        print(f"{model_name} mean accuracy: {accuracies:0.3f}")
        assert (
            accuracies > get_desired_accuracies()[model_name]
        ), f"Failed {model_name}, expected {get_desired_accuracies()[model_name]}, got {accuracies}"


if __name__ == "__main__":
    test_forwards()
    test_classify()
