# Memorax - Sequence and memory modeling in JAX

Currently under construction, there might be bugs.

Memorax is a library for efficient recurrent models. Using category theory, we utilize a [simple interface](memorax/groups.py) that should work for nearly all recurrent models.

## Currently Available Models
- Linear Recurrent unit
- Fast AutoRegressive Transformer
- Fast and Forgetful Memory

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

# You can pack multiple subsequences into a single sequence using the start flag
sequence_starts = jnp.array([True, False, False, True, False])
x = jnp.zeros((5, 16))
inputs = (x, sequence_starts)

# Initialize the model
model = LRU(input_size=16, hidden_size=16, output_size=16)

# Run the model!
h = model.intialize_carry()
h, y = model(h, inputs)
# Unlike most other libraries, we output ALL recurrent states h, not just the most recent
print(h.shape) # (5, 16, 16)
# Do your prediction
prediction = jax.nn.softmax(y)
```

## Further Examples
See the `tests` directory
