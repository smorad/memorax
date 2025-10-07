from functools import partial

import jax
import jax.numpy as jnp
import optax
from memorax_flax.groups import Resettable
from memorax_flax.models.residual import ResidualModel
from memorax_flax.monoids.lru import LRU, LRUMonoid
from memorax_flax.scans import monoid_scan


def ce_loss(y_hat, y):
    return -jnp.mean(jnp.sum(y * jax.nn.log_softmax(y_hat, axis=-1), axis=-1))


def test_forward(model, num_seqs=5, seq_len=20, input_dims=4):
    timesteps = num_seqs * seq_len
    seq_idx = jnp.array([seq_len * i for i in range(num_seqs)])
    start = jnp.zeros((timesteps,), dtype=bool).at[seq_idx].set(True)
    x = jax.random.randint(jax.random.PRNGKey(0), (timesteps,), 0, input_dims - 1)
    x = jax.nn.one_hot(x, input_dims - 1)
    x = jnp.concatenate([x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
    y = jnp.repeat(x[seq_idx, :-1], seq_len, axis=0)

    initialise_carry_fn = partial(model.apply, method="initialize_carry")
    initialise_carry_fn = jax.jit(initialise_carry_fn)
    # Get constructor carry
    h = model.zero_carry()
    # Get params
    params = model.init(jax.random.PRNGKey(0), h, (x, start))
    # Get real init carry
    h = initialise_carry_fn(params)
    # Apply the model
    _, y_hat = model.apply(params, h, (x, start))
    loss = ce_loss(y_hat, y)
    accuracy = jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1)
    assert y_hat.shape == y.shape


def train_initial_input(
    model, epochs=4000, num_seqs=5, seq_len=20, input_dims=4, eval_model=False
):
    timesteps = num_seqs * seq_len
    seq_idx = jnp.array([seq_len * i for i in range(num_seqs)])
    start = jnp.zeros((timesteps,), dtype=bool).at[seq_idx].set(True)
    dummy_h = model.zero_carry()
    dummy_x = jax.random.randint(jax.random.PRNGKey(0), (timesteps,), 0, input_dims - 1)
    dummy_x = jax.nn.one_hot(dummy_x, input_dims - 1)
    dummy_x = jnp.concatenate(
        [dummy_x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1
    )
    params = model.init(jax.random.PRNGKey(0), dummy_h, (dummy_x, start))
    opt = optax.adam(learning_rate=3e-3)
    state = opt.init(params)
    initialise_carry_fn = partial(model.apply, method="initialize_carry")
    initialise_carry_fn = jax.jit(initialise_carry_fn)

    def error(params, key):
        h = initialise_carry_fn(params)
        x = jax.random.randint(key, (timesteps,), 0, input_dims - 1)
        x = jax.nn.one_hot(x, input_dims - 1)
        x = jnp.concatenate([x, start.astype(jnp.float32).reshape(-1, 1)], axis=-1)
        y = jnp.repeat(x[seq_idx, :-1], seq_len, axis=0)

        _, y_hat = model.apply(params, h, (x, start))
        loss = ce_loss(y_hat, y)
        accuracy = jnp.mean(jnp.argmax(y, axis=-1) == jnp.argmax(y_hat, axis=-1))
        return loss, {"loss": loss, "accuracy": accuracy}

    loss_fn = jax.jit(jax.grad(error, has_aux=True))
    key = jax.random.PRNGKey(0)
    losses = []
    accuracies = []
    for epoch in range(epochs):
        key, _ = jax.random.split(key)
        grads, loss_info = loss_fn(params, key)
        updates, state = jax.jit(opt.update)(grads, state)
        params = optax.apply_updates(params, updates)
        print(
            f"Step {epoch+1}, Loss: {loss_info['loss']:0.2f}, Accuracy: {loss_info['accuracy']:0.2f}"
        )
        losses.append(loss_info["loss"])
        accuracies.append(loss_info["accuracy"])

    if not eval_model:
        return jnp.stack(losses), jnp.stack(accuracies)

    # For debug purposes only
    key, _ = jax.random.split(key)
    h = initialise_carry_fn(params)
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


def get_memory_models(hidden: int, output: int):
    layers = {
        # monoids
        "lru": lambda recurrent_size: LRU(
            algebra=LRU.default_algebra(
                recurrent_size=recurrent_size,
            ),
            scan=LRU.default_scan(),
            hidden_size=recurrent_size,
            recurrent_size=recurrent_size,
        ),
    }
    return {
        name: ResidualModel(
            make_layer_fn=fn,
            recurrent_size=hidden,
            output_size=output,
            num_layers=2,
        )
        for name, fn in layers.items()
    }


def get_desired_accuracies():
    return {
        "ffm": 1.0,
        "fart": 1.0,
        "lru": 1.0,
        "mlstm": 1.0,
        "linear_rnn": 1.0,
        "gilr": 1.0,
        "log_bayes": 1.0,
        "gru": 1.0,
        "elman": 1.0,
        "ln_elman": 1.0,
        "spherical": 1.0,
        "p_spherical": 1.0,
        "nmax": 1.0,
        "hardtanh": 1.0,
        "hard_gru": 1.0,
        "mgu": 1.0,
    }


def test_forwards():
    test_size = 4
    hidden = 4
    for model_name, model in get_memory_models(hidden, test_size - 1).items():
        try:
            test_forward(model)
        except Exception:
            print("Model crashed", model_name)
            raise


def test_classify():
    test_size = 4
    hidden = 8
    for model_name, model in get_memory_models(hidden, test_size - 1).items():
        losses, accuracies = train_initial_input(model)
        losses = losses[-100:].mean()
        accuracies = accuracies[-100:].mean()
        print(f"{model_name} mean accuracy: {accuracies:0.3f}")
        assert (
            accuracies >= get_desired_accuracies()[model_name]
        ), f"Failed {model_name}, expected {get_desired_accuracies()[model_name]}, got {accuracies}"


if __name__ == "__main__":
    test_forwards()
    test_classify()
