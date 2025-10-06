import equinox as eqx
import jax
import jax.numpy as jnp

from memorax.set_actions.gru import GRU
from memorax.models.residual import ResidualModel
from memorax.semigroups.lru import LRU, LRUSemigroup
from memorax.utils import debug_shape


def test_readme():
    # You can pack multiple subsequences into a single sequence using the start flag
    sequence_starts = jnp.array([True, False, False, True, False])
    x = jnp.zeros((5, 3))
    inputs = (x, sequence_starts)

    # Initialize a multi-layer recurrent model
    key = jax.random.key(0)
    make_layer_fn = lambda recurrent_size, key: LRU(
        hidden_size=recurrent_size, recurrent_size=recurrent_size, key=key
    )
    model = ResidualModel(
        make_layer_fn=make_layer_fn,
        input_size=3,
        recurrent_size=16,
        output_size=4,
        num_layers=2,
        key=key,
    )

    # Note: We also have layers if you want to build your own model
    layer = LRU(hidden_size=16, recurrent_size=16, key=key)
    # Or semigroups/set actions (scanned functions) if you want to build your own layer
    sg = LRUSemigroup(recurrent_size=16)

    # Run the model! All models are jit-capable, using equinox.filter_jit
    h = eqx.filter_jit(model.initialize_carry)()
    # Unlike most other libraries, we output ALL recurrent states h, not just the most recent
    h, y = eqx.filter_jit(model)(h, inputs)
    # Since we have two layers, we have a recurrent state of shape
    print(debug_shape(h))
    #     ((5, 16), # Recurrent states of first layer
    #     (5,) # Start carries for first layer
    #     (5, 16) # Recurrent states of second layer
    #     (5,)) # Start carries for second layer
    # 
    # Do your prediction
    prediction = jax.nn.softmax(y)

    # If you want to continue rolling out the RNN from h[-1]
    # you should use the following helper function to extract
    # h[-1] from the nested recurrent state
    latest_h = eqx.filter_jit(model.latest_recurrent_state)(h)
    # Continue rolling out as you please! You can use a single timestep
    # or another sequence.
    last_h, last_y = eqx.filter_jit(model)(latest_h, inputs)

    # We can use a similar approach with RNNs
    make_layer_fn = lambda recurrent_size, key: GRU(
        recurrent_size=recurrent_size, key=key
    )
    model = ResidualModel(
        make_layer_fn=make_layer_fn,
        input_size=3,
        recurrent_size=16,
        output_size=4,
        num_layers=2,
        key=jax.random.key(0),
    )
    h = eqx.filter_jit(model.initialize_carry)()
    h, y = eqx.filter_jit(model)(h, inputs)
    prediction = jax.nn.softmax(y)
    latest_h = eqx.filter_jit(model.latest_recurrent_state)(h)
    h, y = eqx.filter_jit(model)(latest_h, inputs)

def test_readme_quickstart():
    from memorax.train_utils import get_residual_memory_models
    import jax
    import jax.numpy as jnp
    from equinox import filter_jit

    model = get_residual_memory_models(input=3, hidden=8, output=1, num_layers=2, models=["LRU"], key=jax.random.key(0))["LRU"]

    starts = jnp.array([True, False, False, True, False])
    xs = jnp.zeros((5, 3)) # T x F
    hs, ys = filter_jit(model)(model.initialize_carry(), (xs, starts))
    last_h = filter_jit(model.latest_recurrent_state)(hs)

if __name__ == '__main__':
    test_readme()
    test_readme_quickstart()