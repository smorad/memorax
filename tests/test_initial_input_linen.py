"""Test all models on a simple 'remember the first input in the sequence' task"""
import pytest
import jax
import jax.numpy as jnp
import optax
from functools import partial

from memax.linen.train_utils import get_residual_memory_models


def get_desired_accuracies():
    return {
        "LRU": 0.999,
        "S6": 0.999,
        "FART": 0.999,
        "GRU": 0.999,
    }

def ce_loss(y_hat, y):
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1))

@pytest.mark.parametrize("model_name, model", get_residual_memory_models(
        8, 4 - 1, 
    ).items())
def test_initial_input(
    model_name, model, epochs=4000, num_seqs=5, seq_len=20, input_dims=4
):
    timesteps = num_seqs * seq_len
    seq_idx = jnp.array([seq_len * i for i in range(num_seqs)])
    start = jnp.zeros((timesteps,), dtype=bool).at[seq_idx].set(True)
    opt = optax.adam(learning_rate=3e-3)

    # init model
    key = jax.random.PRNGKey(0)
    dummy_x = jax.random.randint(key, (timesteps,), 0, input_dims - 1)
    dummy_x = jax.nn.one_hot(dummy_x, input_dims - 1)
    dummy_x = jnp.concatenate([dummy_x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
    dummy_h = model.zero_carry()
    dummy_starts = jnp.zeros(dummy_x.shape[0], dtype=bool)
    params = model.init(key, dummy_h, (dummy_x, dummy_starts))
    init_carry_fn = partial(model.apply, method="initialize_carry")
    apply_fn = model.apply
    state = opt.init(params)

    def error(params, key):
        h = init_carry_fn(params) 
        x = jax.random.randint(key, (timesteps,), 0, input_dims - 1)
        x = jax.nn.one_hot(x, input_dims - 1)
        x = jnp.concatenate([x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
        y = jnp.repeat(x[seq_idx, :-1], seq_len, axis=0)

        _, y_hat = apply_fn(params, h, (x, start))
        loss = ce_loss(y_hat, y)
        accuracy = jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))
        return loss, {"loss": loss, "accuracy": accuracy}

    loss_fn = jax.jit(jax.grad(error, has_aux=True))
    losses = []
    accuracies = []
    for epoch in range(epochs):
        key, _ = jax.random.split(key)
        grads, loss_info = loss_fn(params, key)
        updates, state = jax.jit(opt.update)(grads, state)
        params = optax.apply_updates(params, updates)
        losses.append(loss_info["loss"])
        accuracies.append(loss_info["accuracy"])

    losses, accuracies = jnp.stack(losses), jnp.stack(accuracies)
    losses = losses[-100:].mean()
    accuracies = accuracies[-100:].mean()
    print(f"{model_name} mean accuracy: {accuracies:0.3f}")
    assert (
        accuracies >= get_desired_accuracies()[model_name]
    ), f"Failed {model_name}, expected {get_desired_accuracies()[model_name]}, got {accuracies}"

    # Verify recurrent mode works well too
    def rerror(params, key):
        h = init_carry_fn(params) 
        x = jax.random.randint(key, (timesteps,), 0, input_dims - 1)
        x = jax.nn.one_hot(x, input_dims - 1)
        x = jnp.concatenate([x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
        y = jnp.repeat(x[seq_idx, :-1], seq_len, axis=0)
        y_hats = []

        for t in range(timesteps):
            h, y_hat = apply_fn(params, h, (x[t : t + 1], start[t : t + 1]))
            h = model.latest_recurrent_state(h)
            y_hats.append(y_hat)

        y_hat = jnp.concatenate(y_hats, axis=0)
        loss = ce_loss(y_hat, y)
        accuracy = jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))
        return loss, {"loss": loss, "accuracy": accuracy}

    _, r_metrics = rerror(params, key)
    assert (
        r_metrics['accuracy']>= get_desired_accuracies()[model_name]
    ), f"Failed {model_name} (recurrent mode), expected {get_desired_accuracies()[model_name]}, got {r_metrics['accuracy']}"


if __name__ == "__main__":
    test_initial_input()
