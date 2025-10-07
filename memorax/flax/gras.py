from beartype.typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
from jaxtyping import PRNGKeyArray, Shaped

from memorax.mtypes import Input, OutputEmbedding, RecurrentState, SingleRecurrentState
from memorax.flax.groups import BinaryAlgebra, Module


class GRAS(Module):
    r"""A Generalized Recurrent Algebraic Structure (GRAS) 

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
        """Maps inputs to the recurrent space"""
        raise NotImplementedError

    def backward_map(
        self,
        h: RecurrentState,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> OutputEmbedding:
        """Maps the recurrent space to output space"""
        raise NotImplementedError

    @nn.compact
    def __call__(
        self,
        h: SingleRecurrentState,
        x: Input,
        key: Optional[Shaped[PRNGKeyArray, ""]] = None,
    ) -> Tuple[RecurrentState, OutputEmbedding]:
        """Calls the mapping and scan functions.

        You probably do not need to override this."""
        if key is None:
            in_key, out_key = (None, None)
        else:
            in_key, out_key = jax.random.split(key)
        scan_input = jax.vmap(self.forward_map)(x, in_key)
        next_h = self.scan(self.algebra, h, scan_input)
        y = jax.vmap(self.backward_map)(next_h, x, out_key)
        return next_h, y

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> SingleRecurrentState:
        """Initialize the recurrent state for a new sequence."""
        return self.algebra.initialize_carry(key=key)

    @nn.nowrap
    def zero_carry(self) -> RecurrentState:
        return self.algebra.zero_carry()

    @nn.nowrap
    def latest_recurrent_state(self, h: RecurrentState) -> RecurrentState:
        """Get the latest state from a sequence of recurrent states."""
        return jax.tree.map(lambda x: x[-1], h)

    @staticmethod
    def default_algebra(**kwargs):
        raise NotImplementedError

    @staticmethod
    def default_scan():
        raise NotImplementedError
