
"""This module contains set-action-based (classical) recurrent layers.
Each RNN type gets its own file.
+ `memax.equinox.set_actions.elman` provides a basic Elman RNN layer.
+ `memax.equinox.set_actions.lstm` provides the Long Short-Term Memory layer.
+ `memax.equinox.set_actions.gru` provides the Gated Recurrent Unit layer.
+ `memax.equinox.set_actions.mru` provides the Minimal Gated Unit layer
+ `memax.equinox.set_actions.spherical` provides a recurrent formulation of the Rotational RNN. See `memax.equinox.semigroups.spherical` for the semigroup version.
"""