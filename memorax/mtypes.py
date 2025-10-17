"""
Some definitions used in our models
Recurrent layers should take (Input, RecurrentState) 
and return (RecurrentState, OutputEmbedding)
"""
from beartype.typing import Tuple

from jaxtyping import Array, Bool, PyTree, Shaped

# Inputs
StartFlag = Bool[Array, ""]
"""Start flags denote the beginning of a new sequence. They should be
1 where a new sequence starts, and 0 elsewhere."""
InputEmbedding = Shaped[Array, "Feature"]
"""An input embedding are the per-time-step input features to the recurrent model."""
InputEmbeddings = Shaped[Array, "Time Feature"]
"""InputEmbeddings are a temporal sequence of InputEmbedding."""
Input = Tuple[InputEmbedding, StartFlag]
"""All layers take both an input embedding and a start flag as input."""

# Recurrence
RecurrentState = PyTree[Array, "..."]
"""A RecurrentState is the hidden state of a recurrent layer for a single timestep."""
RecurrentStates = PyTree[Array, "Time ..."]
"""RecurrentStates are the hidden states of a recurrent layer for all timesteps."""
SingleRecurrentState = PyTree[Array, "One ..."]
"""A single recurrent state with a leading singleton time dimension."""
ResetRecurrentState = Tuple[RecurrentState, StartFlag]
"""A single recurrent state and reset carry with a leading singleton time dimension."""

# Outputs
OutputEmbedding = Shaped[Array, "..."]
"""The output (not recurrent state) of a recurrent layer for a single timestep."""
OutputEmbeddings = Shaped[Array, "Time ..."]
"""The output (not recurrent state) of a recurrent layer for all timesteps."""
