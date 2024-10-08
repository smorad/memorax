# Memorax - Sequence and memory modeling in JAX

Currently under construction, there might be bugs. Run `test/test_initial_input` to ensure the models work!

Memorax is a library for efficient recurrent models. Using category theory, we utilize a [simple interface](memorax/groups.py) that should work for nearly all recurrent models.

## Currently Available Models
### [Memoroids](https://openreview.net/forum?id=nA4Q983a1v), with $O(\log{n})$ parallel-time complexity
- [Linear Recurrent Unit](https://arxiv.org/abs/2303.06349) (State Space Model) [[Code]](memorax/monoids/lru.py)
- [Linear Recurrent Neural Network](https://arxiv.org/abs/1709.04057) [[Code]](memorax/monoids/lrnn.py)
- [Gated Impulse Linear Recurrence](https://arxiv.org/abs/1709.04057) [[Code]](memorax/monoids/gilr.py)
- [Fast Autoregressive Transformer](https://arxiv.org/abs/2006.16236) [[Code]](memorax/monoids/fart.py)
- [Fast and Forgetful Memory](https://arxiv.org/abs/2310.04128) [[Code]](memorax/monoids/ffm.py)
- [mLSTM/xLSTM](https://arxiv.org/abs/2405.04517) [[Code]](memorax/monoids/mlstm.py)

### RNNs, with $O(n)$ parallel-time complexity
- [Elman Network](https://www.sciencedirect.com/science/article/pii/036402139090002E) [[Code]](memorax/magmas/elman.py)
- [Gated Recurrent Unit](https://arxiv.org/abs/1412.3555) [[Code]](memorax/magmas/gru.py)
- [Minimal Gated Unit](https://arxiv.org/abs/1603.09420) [[Code]](memorax/magmas/mgru.py)

# Getting Started

Install `memorax` using pip and git
```bash
git clone https://github.com/smorad/memorax
cd memorax
pip install -e .
```

Memorax uses the `equinox` neural network library. See [the monoids directory](https://github.com/smorad/memorax/tree/main/memorax/monoids) for fast recurrent models that utilize an associative scan.

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from memorax.magmas.gru import GRU
from memorax.models.residual import ResidualModel
from memorax.monoids.lru import LRU, LRUMonoid
from memorax.utils import debug_shape

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
# Or monoids/magmas (scanned functions) if you want to build your own layer
monoid = LRUMonoid(recurrent_size=16, max_phase=2 * jnp.pi, key=key)

# Run the model! All models are jit-capable, using equinox.filter_jit
h = eqx.filter_jit(model.initialize_carry)()
# Unlike most other libraries, we output ALL recurrent states h, not just the most recent
h, y = eqx.filter_jit(model)(h, inputs)
# Do your prediction
prediction = jax.nn.softmax(y)
# Since we have two layers, we have a recurrent state of shape
print(debug_shape(h))
#     ((5, 16), # Recurrent states of first layer
#     (5,) # Done/carry flags for first layer
#     (5, 16) # Recurrent states of second layer
#     (5,)) # Done/carry flags for second layer

# Continue rolling out from the most recent recurrent state
latest_h = model.latest_recurrent_state(h)
h, y = eqx.filter_jit(model)(latest_h, inputs)

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
latest_h = model.latest_recurrent_state(h)
h, y = eqx.filter_jit(model)(latest_h, inputs)
```

## Further Examples
See the `tests` directory
