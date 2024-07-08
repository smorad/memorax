import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from equinox import nn

from memorax.magmas.elman import Elman
from memorax.memoroid import Memoroid
from memorax.monoids.fart import FART
from memorax.monoids.ffm import FFM
from memorax.monoids.lru import LRU
from memorax.utils import relu


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
    assert y_hat.shape == y.shape


def train_initial_input(
    model, epochs=1000, num_seqs=1, seq_len=10, input_dims=4, eval_model=True
):
    timesteps = num_seqs * seq_len
    seq_idx = jnp.array([seq_len * i for i in range(num_seqs)])
    start = jnp.zeros((timesteps,), dtype=bool).at[seq_idx].set(True)

    opt = optax.adam(learning_rate=0.0002)
    state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def error(model, key):
        h = model.initialize_carry()
        x = jax.random.randint(key, (timesteps,), 0, input_dims - 1)
        x = jax.nn.one_hot(x, input_dims - 1)
        x = jnp.concatenate([x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
        y = jnp.repeat(x[seq_idx, :-1], seq_len, axis=0)

        _, y_hat = model(h, (x, start))
        y_hat = jnp.squeeze(y_hat)
        y = jnp.squeeze(y)
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

    _, y_hat = model(h, (x, start))
    y_hat = jnp.squeeze(y_hat)
    y = jnp.squeeze(y)
    loss = ce_loss(y_hat, y)
    accuracy = jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))

    return jnp.stack(losses), jnp.stack(accuracies)


class Model(eqx.Module):
    ff_in: nn.Sequential
    ff_out: nn.Sequential
    model: Memoroid

    def __init__(
        self, model, in_size, model_in_size, hidden_size, model_out_size, out_size, key
    ):
        key = jax.random.split(key, 5)
        self.model = model
        self.ff_in = nn.Sequential(
            [
                nn.Linear(in_size, hidden_size, key=key[0]),
                nn.LayerNorm((hidden_size,), use_weight=False, use_bias=False),
                relu,
                nn.Linear(hidden_size, hidden_size, key=key[1]),
                nn.LayerNorm((hidden_size,), use_weight=False, use_bias=False),
                relu,
                nn.Linear(hidden_size, model_in_size, key=key[2]),
            ]
        )
        self.ff_out = nn.Sequential(
            [
                nn.Linear(model_out_size, hidden_size, key=key[3]),
                nn.LayerNorm((hidden_size,), use_weight=False, use_bias=False),
                relu,
                nn.Linear(hidden_size, hidden_size, key=key[4]),
                nn.LayerNorm((hidden_size,), use_weight=False, use_bias=False),
                relu,
                nn.Linear(hidden_size, out_size, key=key[5]),
            ]
        )

    def __call__(self, h, x):
        z, *other = x
        z = eqx.filter_vmap(self.ff_in)(z)
        h, x = self.model(h, (z, *other))
        final_recurrent_state = jax.tree.map(lambda h: h[-1:], h)
        z = eqx.filter_vmap(self.ff_out)(z)
        return final_recurrent_state, z

    def initialize_carry(self):
        return self.model.initialize_carry()


def get_memory_models(hidden: int):
    return {
        "ffm": FFM(
            hidden_size=hidden,
            trace_size=hidden,
            context_size=hidden,
            key=jax.random.PRNGKey(0),
        ),
        "fart": FART(hidden_size=hidden, key_size=hidden, key=jax.random.PRNGKey(0)),
        "lru": LRU(
            recurrent_size=hidden, hidden_size=hidden, key=jax.random.PRNGKey(0)
        ),
        "elman": Elman(
            recurrent_size=hidden, hidden_size=hidden, key=jax.random.PRNGKey(0)
        ),
    }


def test_forwards():
    test_size = 4
    hidden = 8
    for model_name, model in get_memory_models(hidden).items():
        model = Model(
            model,
            in_size=test_size,
            model_in_size=hidden,
            hidden_size=hidden,
            model_out_size=hidden,
            out_size=test_size - 1,
            key=jax.random.PRNGKey(1),
        )
        test_forward(model)


def test_classify():
    test_size = 4
    hidden = 16
    for model_name, model in get_memory_models(hidden).items():
        model = Model(
            model,
            in_size=test_size,
            model_in_size=hidden,
            hidden_size=hidden,
            model_out_size=hidden,
            out_size=test_size - 1,
            key=jax.random.PRNGKey(1),
        )
        train_initial_input(model)
        losses, accuracies = train_initial_input(model)
        losses = losses[-100:].mean()
        accuracies = accuracies[-100:].mean()
        print(f"{model_name} mean loss: {losses:0.3f}")
        print(f"{model_name} mean accuracy: {accuracies:0.3f}")
        # assert losses < 0.05


if __name__ == "__main__":
    # test_ffm()
    # test_fart()
    # test_forwards()
    test_classify()
