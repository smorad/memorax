from memorax.memoroids.fart import FART
from memorax.memoroids.ffm import FFM
import pytest
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from equinox import nn

from memorax.memoroids.memoroid import Memoroid
from memorax.utils import relu


def test_forward(model, num_seqs=5, seq_len=20, input_dims=2):
    timesteps = num_seqs * seq_len
    seq_idx = jnp.array([seq_len * i for i in range(num_seqs)])
    start = jnp.zeros((timesteps,), dtype=bool).at[seq_idx].set(True)
    h = model.initialize_carry()
    x = jnp.zeros((timesteps, input_dims))
    _, y_hat = model(h, (x, start))


def train_initial_input(model, epochs=1000, num_seqs=5, seq_len=20, input_dims=2):
    timesteps = num_seqs * seq_len
    seq_idx = jnp.array([seq_len * i for i in range(num_seqs)])
    start = jnp.zeros((timesteps,), dtype=bool).at[seq_idx].set(True)

    opt = optax.adam(learning_rate=0.001)
    state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    def error(model, key):
        h = model.initialize_carry()
        x = jax.random.uniform(key, (timesteps, input_dims - 1))
        x = jnp.concatenate([x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
        y = jnp.repeat(x[seq_idx, : input_dims + 1], seq_len, axis=0)
        # y = y.reshape(timesteps, )

        _, y_hat = model(h, (x, start))
        y_hat = jnp.squeeze(y_hat)
        y = jnp.squeeze(y)
        loss = jnp.mean(jnp.abs(y - y_hat) ** 2)
        return loss, {"loss": loss}

    loss_fn = eqx.filter_jit(eqx.filter_grad(error, has_aux=True))
    key = jax.random.PRNGKey(0)
    losses = []
    for epoch in range(epochs):
        key, _ = jax.random.split(key)
        grads, loss_info = loss_fn(model, key)
        updates, state = jax.jit(opt.update)(grads, state)
        model = eqx.apply_updates(model, updates)
        # print(f"Step {epoch+1}, Loss: {loss_info['loss']}")
        losses.append(loss_info["loss"])
    return jnp.stack(losses)


class Model(eqx.Module):
    ff_in: nn.Linear 
    ff_out: nn.Sequential
    model: Memoroid

    def __init__(self, model, in_size, model_in_size, hidden_size, model_out_size, out_size, key):
        key = jax.random.split(key, 4)
        self.model = model
        self.ff_in = nn.Linear(in_size, model_in_size, key=key[0])
        self.ff_out = nn.Sequential(
            [
                nn.Linear(model_out_size, hidden_size, key=key[1]),
                relu,
                nn.Linear(hidden_size, hidden_size, key=key[2]),
                relu,
                nn.Linear(hidden_size, out_size, key=key[3]),
            ]
        )

    def __call__(self, h, x):
        z, *other = x
        z = eqx.filter_vmap(self.ff_in)(z)
        h, x = self.model(h, (z, *other))
        final_recurrent_state = jax.tree.map(lambda h: h[-1:], h)
        return final_recurrent_state, eqx.filter_vmap(self.ff_out)(z)

    def initialize_carry(self):
        return self.model.initialize_carry()


def test_ffm():
    input_size = 2
    output_size = 2

    ffm = FFM(
        input_size=16, trace_size=16, context_size=16, output_size=16, key=jax.random.PRNGKey(0)
    )
    model = Model(
        ffm,
        in_size=input_size,
        model_in_size=16,
        hidden_size=16,
        model_out_size=16,
        out_size=output_size,
        key=jax.random.PRNGKey(1),
    )
    test_forward(model)
    losses = train_initial_input(model)[-100:].mean()
    print(f"FFM mean loss: {losses:0.3f}")
    assert losses < 0.05


def test_fart():
    input_size = 2
    output_size = 2
    model = FART(hidden_size=16, key_size=16, key=jax.random.PRNGKey(0))
    model = Model(
        model,
        in_size=input_size,
        model_in_size=16,
        hidden_size=16,
        model_out_size=16,
        out_size=output_size,
        key=jax.random.PRNGKey(1),
    )

    test_forward(model)
    losses = train_initial_input(model)[-100:].mean()
    print(f"FART mean loss: {losses:0.3f}")
    assert losses < 0.05


if __name__ == "__main__":
    test_ffm()
    test_fart()
