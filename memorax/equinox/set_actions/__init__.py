
"""This module contains set-action-based (classical) recurrent layers.
Each RNN type gets its own file.
+ `memorax.equinox.set_actions.elman` provides a basic Elman RNN layer.
+ `memorax.equinox.set_actions.lstm` provides the Long Short-Term Memory layer.
+ `memorax.equinox.set_actions.gru` provides the Gated Recurrent Unit layer.
+ `memorax.equinox.set_actions.mru` provides the Minimal Gated Unit layer
+ `memorax.equinox.set_actions.spherical` provides a recurrent formulation of the Rotational RNN. See `memorax.equinox.semigroups.spherical` for the semigroup version.
"""