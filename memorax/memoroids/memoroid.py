from memorax.groups import Module, Monoid
from memorax.mtypes import OutputEmbedding, RecurrentState, SingleRecurrentState, Input
from typing import Tuple
from memorax.scans import monoid_scan
import equinox as eqx


class Memoroid(Module):
    r"""A memoroid from https://arxiv.org/abs/2402.09900

    Given a recurrent state and inputs, returns the corresponding recurrent states and outputs

    A memoroid contains a monoid :math:`(H, \bullet, e_I)` and two maps/functions :math:`f,g`

    We use f, g to map to and from the monoid space

    .. math::

        f: X^n \times \{0, 1\}^n \mapsto H^n

        \bullet: H \times H \mapsto H

        g: H^n \times X^n \{0, \1}^n \mapsto Y^n
    """

    monoid: Monoid

    def forward_map(self, x: Input) -> RecurrentState:
        raise NotImplementedError

    def backward_map(self, h: RecurrentState, x: Input) -> OutputEmbedding:
        raise NotImplementedError

    def __call__(self, h: SingleRecurrentState, x: Input) -> Tuple[RecurrentState, OutputEmbedding]:
        scan_input = eqx.filter_vmap(self.forward_map)(x)
        next_h = monoid_scan(self.monoid, h, scan_input)
        y = eqx.filter_vmap(self.backward_map)(next_h, x)
        return next_h, y

    def initialize_carry(self, batch_shape: Tuple[int, ...] = ()) -> SingleRecurrentState:
        return self.monoid.initialize_carry(batch_shape)
