"""Test all models on a simple 'remember the first input in the sequence' task"""
import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from memorax.train_utils import get_residual_memory_models


def ce_loss(y_hat, y):
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1))

def train_initial_input(
    model, epochs=4000, num_seqs=5, seq_len=20, input_dims=4
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
        losses.append(loss_info["loss"])
        accuracies.append(loss_info["accuracy"])

    return jnp.stack(losses), jnp.stack(accuracies)


def get_desired_accuracies():
    return {
        "MLP": 0,
        "DLSE": 1.0,
        "FFM": 1.0,
        "FART": 1.0,
        "LRU": 1.0,
        "S6": 1.0,
        "LinearRNN": 1.0,
        "PSpherical": 1.0,
        "GRU": 1.0,
        "Elman": 0.69,
        "ElmanReLU": 0.69,
        "Spherical": 1.0,
        "NMax": 1.0,
        "MGU": 1.0,
        "LSTM": 1.0,
        "S6Diagonal": 1.0,
    }


def test_classify():
    test_size = 4
    hidden = 8
    models = get_residual_memory_models(
        test_size, hidden, test_size - 1, key=jax.random.key(0), 
    )
    for model_name, model in models.items():
        losses, accuracies = train_initial_input(model)
        losses = losses[-100:].mean()
        accuracies = accuracies[-100:].mean()
        print(f"{model_name} mean accuracy: {accuracies:0.3f}")
        assert (
            accuracies >= get_desired_accuracies()[model_name]
        ), f"Failed {model_name}, expected {get_desired_accuracies()[model_name]}, got {accuracies}"


if __name__ == "__main__":
    test_classify()
