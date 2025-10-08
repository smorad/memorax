# Memorax - Sequence and Memory Modeling in JAX

[![Tests](https://github.com/smorad/memorax/actions/workflows/python_app.yaml/badge.svg)](https://github.com/smorad/memorax/actions/workflows/python_app.yaml)

Memorax is a library for efficient recurrent models. Using category theory, we utilize a [simple interface](memorax/equinox/groups.py) that should work for nearly all recurrent models. Unlike most other recurrent modeling libraries, we provide a unified interface for fast recurrent state resets across the sequence, allowing you to avoid truncating BPTT.

## Available Models
### [Memoroids](https://openreview.net/forum?id=nA4Q983a1v), with $O(\log{n})$ parallel-time complexity
- [Linear Recurrent Unit](https://arxiv.org/abs/2303.06349) (State Space Model) [[Code]](memorax/equinox/semigroups/lru.py)
- [Selective State Space Model (S6)](https://arxiv.org/abs/2312.00752) [[Code]](memorax/equinox/semigroups/s6.py)
- [Diagonal Selective State Space Model (S6D)](https://arxiv.org/abs/2312.00752) [[Code]](memorax/equinox/semigroups/s6d.py)
- [Linear Recurrent Neural Network](https://arxiv.org/abs/1709.04057) [[Code]](memorax/equinox/semigroups/lrnn.py)
- [Fast Autoregressive Transformer](https://arxiv.org/abs/2006.16236) [[Code]](memorax/equinox/semigroups/fart.py)
- [Fast and Forgetful Memory](https://arxiv.org/abs/2310.04128) [[Code]](memorax/equinox/semigroups/ffm.py)
- [Rotational RNN (RotRNN)](https://arxiv.org/abs/2407.07239) [[Code]](memorax/equinox/semigroups/spherical.py)

### RNNs, with $O(n)$ parallel-time complexity
- [Elman Network](https://www.sciencedirect.com/science/article/pii/036402139090002E) [[Code]](memorax/equinox/set_actions/elman.py)
- [Gated Recurrent Unit](https://arxiv.org/abs/1412.3555) [[Code]](memorax/equinox/set_actions/gru.py)
- [Minimal Gated Unit](https://arxiv.org/abs/1603.09420) [[Code]](memorax/equinox/set_actions/mgu.py)
- [Long Short-Term Memory Unit ](https://ieeexplore.ieee.org/abstract/document/6795963) [[Code]](memorax/equinox/set_actions/lstm.py)

## Datasets
We provide datasets to test our recurrent models. 

### Sequential MNIST [[HuggingFace]](https://huggingface.co/datasets/ylecun/mnist) [[Code]](memorax/datasets/sequential_mnist.py)
> The recurrent model receives an MNIST image pixel by pixel, and must predict the digit class.
>
> **Sequence Lengths:** `[784]`

### MNIST Math [[HuggingFace]](https://huggingface.co/datasets?sort=trending&search=bolt-lab%2Fmnist-math) [[Code]](memorax/datasets/sequential_mnist.py)
> The recurrent model receives a sequence of MNIST images and operators, pixel by pixel, and must predict the percentile of the operators applied to the MNIST image classes.
>
> **Sequence Lengths:** `[784 * 5, 784 * 100, 784 * 1_000, 784 * 10_000, 784 * 1_000_000]`

### Continuous Localization [[HuggingFace]](https://huggingface.co/datasets?sort=trending&search=bolt-lab%2Fcontinuous-localization) [[Code]](memorax/datasets/sequential_mnist.py)
> The recurrent model receives a sequence of translation and rotation vectors **in the local coordinate frame**, and must predict the corresponding position and orientation **in the global coordinate frame**.
>
> **Sequence Lengths:** `[20, 100, 1_000]`

# Getting Started
Install `memorax` using pip and git for your specific framework
```bash
pip install "memorax[equinox]@git+https://github.com/smorad/memorax"
pip install "memorax[flax]@git+https://github.com/smorad/memorax"
```

## Equinox Quickstart
```python
from memorax.equinox.train_utils import get_residual_memory_models
import jax
import jax.numpy as jnp
from equinox import filter_jit, filter_vmap
from memorax.equinox.train_utils import add_batch_dim

model = get_residual_memory_models(
    input=3, hidden=8, output=1, num_layers=2, 
    models=["LRU"], key=jax.random.key(0)
)["LRU"]

starts = jnp.array([True, False, False, True, False])
xs = jnp.zeros((5, 3)) # T x F
hs, ys = filter_jit(model)(model.initialize_carry(), (xs, starts))
last_h = filter_jit(model.latest_recurrent_state)(hs)

# With batch dim
B, T, F = 5, 4, 3
starts = jnp.zeros((B, T), dtype=bool)
xs = jnp.zeros((B, T, F))
hs_0 = add_batch_dim(model.initialize_carry(), B)
hs, ys = filter_jit(filter_vmap(model))(hs_0, (xs, starts))
```

## Running Baselines
You can compare various recurrent models on our datasets with a single command
```bash
python run_equinox_experiments.py # equinox framework
python run_linen_experiments.py # flax linen framework
```


## Custom Architectures 
Memorax uses the [`equinox`](https://github.com/patrick-kidger/equinox) neural network library. See [the semigroups directory](memorax/equinox/semigroups) for fast recurrent models that utilize an associative scan. We also provide a beta [`flax.linen`](https://flax-linen.readthedocs.io/en/latest/) API. In this example, we focus on `equinox`.

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from memorax.equinox.set_actions.gru import GRU
from memorax.equinox.models.residual import ResidualModel
from memorax.equinox.semigroups.lru import LRU, LRUSemigroup
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
```

## Creating Custom Recurrent Models
All recurrent cells should follow the [`GRAS`](memorax/equinox/gras.py) interface. A recurrent cell consists of an `Algebra`. You can roughly think of the `Algebra` as the function that updates the recurrent state, and the `GRAS` as the `Algebra` and all the associated MLPs/gates. You may reuse our `Algebra`s in your custom `GRAS`, or even write your custom `Algebra`.

To implement your own `Algebra` and `GRAS`, we suggest copying one from our existing code, such as the [LRNN](memorax/equinox/semigroups/lrnn.py) for a `Semigroup` or the [Elman Network](memorax/equinox/set_actions/elman.py) for a `SetAction`.

# Citing our Work
Please cite the library as
```
@misc{morad_memorax_2025,
	title = {Memorax},
	url = {https://github.com/smorad/memorax},
	author = {Morad, Steven and Toledo, Edan and Kortvelesy, Ryan and He, Zhe},
	month = jun,
	year = {2025},
}
```
If you use the recurrent state resets (`sequence_starts`) with the log complexity memory models, please cite
```
@article{morad2024recurrent,
  title={Recurrent reinforcement learning with memoroids},
  author={Morad, Steven and Lu, Chris and Kortvelesy, Ryan and Liwicki, Stephan and Foerster, Jakob and Prorok, Amanda},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={14386--14416},
  year={2024}
}
```