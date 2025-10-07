from beartype.typing import Callable, Optional, Tuple

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, Shaped

from memorax.equinox.groups import BinaryAlgebra, Module
from memorax.mtypes import Input, OutputEmbedding, RecurrentState, SingleRecurrentState


class GRAS(Module):
    r"""A Generalized Recurrent Algebraic System (GRAS) 

    Given a recurrent state and inputs, returns the corresponding recurrent states and outputs

    A GRAS contains a set action or semigroup :math:`(H, \bullet)` and two maps/functions :math:`f,g`

    For a semigroup, we express a GRAS via

    .. math::

        f: X^n \times \{0, 1\}^n \mapsto H^n

        \bullet: H \times H \mapsto H

        g: H^n \times X^n \{0, \1}^n \mapsto Y^n

    where :math:`\bullet` may be an associative/parallel scan or sequential scan.

    For a set action, the GRAS recurrent update is slightly altered

    .. math::
        f: X^n \times \{0, 1\}^n \mapsto Z^n

        \bullet: H \times Z \mapsto H

        g: H^n \times X^n \{0, \1}^n \mapsto Y^n

    where :math:`\bullet` must be a sequential scan.

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
        
        (x, start) -> H
        """
        raise NotImplementedError

    def backward_map(
        self,
        h: RecurrentState,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> OutputEmbedding:
        """Maps the recurrent space to the output space.
        
        (H, (x, start)) -> Y
        """
        raise NotImplementedError

    def __call__(
        self,
        h: SingleRecurrentState,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Tuple[RecurrentState, OutputEmbedding]:
        """Calls the mapping and scan functions.

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

    def latest_recurrent_state(self, h: RecurrentState) -> RecurrentState:
        """Get the latest state from a sequence of recurrent states."""
        return jax.tree.map(lambda x: x[-1], h)
