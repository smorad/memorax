from beartype.typing import Tuple

from jaxtyping import Array, Bool, PyTree, Shaped

# Inputs
StartFlag = Bool[Array, ""]
InputEmbedding = Shaped[Array, "Feature"]
InputEmbeddings = Shaped[Array, "Time Feature"]
Input = Tuple[InputEmbedding, StartFlag]

# Recurrence
RecurrentState = PyTree[Array, "..."]
RecurrentStates = PyTree[Array, "Time ..."]
SingleRecurrentState = PyTree[Array, "One ..."]
ResetRecurrentState = Tuple[RecurrentState, StartFlag]

# Outputs
OutputEmbedding = Shaped[Array, "..."]
OutputEmbeddings = Shaped[Array, "Time ..."]
