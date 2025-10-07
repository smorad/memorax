from functools import partial

import jax
import jax.numpy as jnp
from memorax_flax.groups import Resettable
from memorax_flax.models.residual import ResidualModel
from memorax_flax.monoids.lru import LRU, LRUMonoid
from memorax_flax.scans import monoid_scan

from memorax.utils import debug_shape

# You can pack multiple subsequences into a single sequence using the start flag
sequence_starts = jnp.array([True, False, False, True, False])
x = jnp.zeros((5, 3))
inputs = (x, sequence_starts)

# Initialize a multi-layer recurrent model
key = jax.random.key(0)
recurrent_size = 16

make_layer_fn = lambda recurrent_size: LRU(
    algebra=LRU.default_algebra(
        recurrent_size=recurrent_size,
    ),
    scan=LRU.default_scan(),
    hidden_size=recurrent_size,
    recurrent_size=recurrent_size,
)
model = ResidualModel(
    make_layer_fn=make_layer_fn,
    recurrent_size=recurrent_size,
    output_size=4,
    num_layers=2,
)

# Note: We also have layers if you want to build your own model
layer = LRU(
    algebra=LRU.default_algebra(
        recurrent_size=recurrent_size,
    ),
    scan=LRU.default_scan(),
    hidden_size=recurrent_size,
    recurrent_size=recurrent_size,
)
# Or monoids/magmas (scanned functions) if you want to build your own layer
monoid = LRUMonoid(recurrent_size=recurrent_size, max_phase=2 * jnp.pi)

# Extract the relevant functions
model_init = model.init
model_init = jax.jit(
    model_init
)  # Get a jittable function which initializes your model.

model_apply_fn = model.apply
model_apply_fn = jax.jit(
    model_apply_fn
)  # Get a jittable function which applies your model.

initialise_carry_fn = partial(model.apply, method="initialize_carry")
initialise_carry_fn = jax.jit(
    initialise_carry_fn
)  # Get a jittable function which initializes the carry.

# Initialise a dummy carry so we can instantiate the models parameters.
# The zero_carry method is a class based method that does not use parameters and is called from the object itself.
# It returns a recurrent state populated with zeros. Its possible this is the same as the actual initial carry however
# it does not use parameters or keys and is mainly a utility method to easily instantiate the parameters.
dummy_h = model.zero_carry()
dummy_inputs = inputs
params = model_init(
    key, dummy_h, dummy_inputs
)  # Initialize a model with random weights

# This is how we would in practice intialise the carry. The reason for this is so that we can have learnt initialisations.
h = initialise_carry_fn(params)

# Run the model!
# Unlike most other libraries, we output ALL recurrent states h, not just the most recent
h, y = model_apply_fn(params, h, inputs)
# Do your prediction
prediction = jax.nn.softmax(y)
# Since we have two layers, we have a recurrent state of shape
print(debug_shape(h))
# print(h)
#     ((5, 16), # Recurrent states of first layer
#     (5,) # Done/carry flags for first layer
#     (5, 16) # Recurrent states of second layer
#     (5,)) # Done/carry flags for second layer

# Continue rolling out from the most recent recurrent state
latest_h = jax.jit(model.latest_recurrent_state)(h)
# TODO (edan): This doesn't work but neither does it work in the main version
# h, y = model_apply_fn(params, latest_h, inputs)
# print(debug_shape(h))
