from beartype.typing import Callable, Optional, Tuple

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, Shaped

from memax.equinox.groups import BinaryAlgebra, Module
from memax.mtypes import Input, OutputEmbedding, RecurrentState, SingleRecurrentState


class GRAS(Module):
    r"""A Generalized Recurrent Algebraic Structure (GRAS). Given a recurrent state and inputs, returns the corresponding recurrent states and outputs

    # Generalized Recurrent Algebraic Structure (GRAS)

    A GRAS contains a **set action** $(H, Z, \bullet)$, an initial state $h_0 \in H$, and two maps/functions 
    
    $f$ maps input features and a boolean start flag to the action space $Z$. 

    $f: X^n \times \\{0, 1\\}^n \mapsto Z^n$

    $\bullet$ applies an action $z \in Z$ to the recurrent state $h \in H$.

    $\bullet: H \times Z \mapsto H$

    $g$ maps the recurrent states, inputs, and start flags to outputs.

    $g: H^n \times X^n \times \\{0, 1\\}^n \mapsto Y^n$

    We utilize `vmap` and apply our GRAS to some input following the below pseudocode:
    ```
    z = vmap(f)(x, start)
    h = scan(., h0, z)
    y = vmap(g)(h, x, start)
    ```

    Note that we can represent any recurrent function in this form.

    # Efficient GRAS via Semigroups

    A semigroup is a special case of a GRAS where the $\bullet$ is associative

    $ a \bullet (b \bullet c) = (a \bullet b) \bullet c $.

    This enables us to execute $\bullet$ via a parallel scan, which is much more efficient than a sequential scan. 
    The semigroup GRAS therefore contains the same maps $f$ and $g$ as above, but $\bullet$ is now a semigroup operation.
    Furthermore, in a semigroup, the action and recurrent state spaces are identical, i.e., $Z = H$

    $\bullet: H \times H \mapsto H$. We execute semigroup-based GRAS just as above, but using a parallel scan for efficiency
    ```
    z = vmap(f)(x, start)
    h = associative_scan(., h0, z)
    y = vmap(g)(h, x, start)
    ```
    """

    algebra: BinaryAlgebra
    scan: Callable[
        [
            Callable[[RecurrentState, RecurrentState], RecurrentState],
            RecurrentState,
            RecurrentState,
        ],
        RecurrentState,
    ]

    def forward_map(
        self, x: Input, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> RecurrentState:
        """Maps inputs to the recurrent space.
        
        `(feature, start) -> H`
        """
        raise NotImplementedError

    def backward_map(
        self,
        h: RecurrentState,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> OutputEmbedding:
        """Maps the recurrent space to the output space.

        `(h, (feature, start)) -> Y`
        """
        raise NotImplementedError

    def __call__(
        self,
        h: SingleRecurrentState,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Tuple[RecurrentState, OutputEmbedding]:
        """Calls the mapping and scan functions.

        ```
        z = vmap(f)(feature, start)
        h = scan(., h0, z)
        y = vmap(g)(h, feature, start)
        ```

        You probably do not need to override this."""
        emb, start = x
        T = start.shape[0]
        if key is None:
            in_key, scan_key, out_key = (None, None, None)
        else:
            in_key, scan_key, out_key = jax.random.split(key, 3)
            in_key = jax.random.split(in_key, T)
            out_key = jax.random.split(out_key, T)
        scan_input = eqx.filter_vmap(self.forward_map)(x, in_key)
        next_h = self.scan(self.algebra, h, scan_input)
        y = eqx.filter_vmap(self.backward_map)(next_h, x, out_key)
        return next_h, y

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SingleRecurrentState:
        """Initialize the recurrent state for a new sequence."""
        return self.algebra.initialize_carry(key=key)

    def latest_recurrent_state(self, hs: RecurrentState) -> RecurrentState:
        """Get the latest state from a sequence of recurrent states."""
        return jax.tree.map(lambda x: x[-1], hs)
