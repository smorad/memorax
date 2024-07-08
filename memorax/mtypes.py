from typing import Tuple
from jaxtyping import Array, PyTree, Bool, Shaped

# Inputs
StartFlag = Bool[Array, "Time"]
InputEmbedding = Shaped[Array, "Time ..."]
Input = Tuple[InputEmbedding, StartFlag]

# Recurrence
RecurrentState = PyTree[Array, "Time ..."]
SingleRecurrentState = PyTree[Array, "One ..."]
ResetRecurrentState = Tuple[RecurrentState, StartFlag]

# Outputs
OutputEmbedding = Shaped[Array, "Time ..."]
