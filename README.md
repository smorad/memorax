# Memorax - Sequence and memory modeling in JAX

Currently under construction, there might be bugs. Run `test/test_initial_input` to ensure the models work!

Memorax is a library for efficient recurrent models. Using category theory, we utilize a [simple interface](memorax/groups.py) that should work for nearly all recurrent models.

## Currently Available Models
### Memoroids, with $ O(\log n) $ parallel-time complexity
- [Linear Recurrent Unit](https://arxiv.org/abs/2303.06349) (State Space Model) [[Code]](memorax/monoids/lru.py)
- [Fast Autoregressive Transformer](https://arxiv.org/abs/2006.16236) [[Code]](memorax/monoids/fart.py)
- [Fast and Forgetful Memory](https://arxiv.org/abs/2310.04128) [[Code]](memorax/monoids/ffm.py)
- [mLSTM/xLSTM](https://arxiv.org/abs/2405.04517) [[Code]](memorax/monoids/mlstm.py)

### RNNs, with $ O(n) $ parallel-time complexity
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
import jax
import jax.numpy as jnp
from memorax.monoids.lru import LRU
import equinox as eqx

# You can pack multiple subsequences into a single sequence using the start flag
sequence_starts = jnp.array([True, False, False, True, False])
x = jnp.zeros((5, 16))
inputs = (x, sequence_starts)

# Initialize the model
model = LRU(input_size=16, hidden_size=16, output_size=16, num_layers=2, key=jax.random.key(0))

# Run the model! All models are jit-capable, using equinox.filter_jit
h = eqx.filter_jit(model.initialize_carry)()
h, y = eqx.filter_jit(model)(h, inputs)
# Unlike most other libraries, we output ALL recurrent states h, not just the most recent
jax.tree.map(lambda x: print(x.shape), h)
# Do your prediction
prediction = jax.nn.softmax(y)
```

## Further Examples
See the `tests` directory
