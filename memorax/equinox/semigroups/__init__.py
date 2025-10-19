"""This module contains semigroup-based recurrent layers.
They are generally much faster than standard RNNs.

Each RNN type gets its own file.
+ `memorax.equinox.semigroups.attn` provides dot-product attention layer. 
+ `memorax.equinox.semigroups.delta` provides the DeltaNet layer.
+ `memorax.equinox.semigroups.stack` provides framestacking (sliding-window) as an RNN.
+ `memorax.equinox.semigroups.fart` provides the Fast AutoRegressive Transformer layer.
+ `memorax.equinox.semigroups.ffm` provides the Fast and Forgetful Memory layer.
+ `memorax.equinox.semigroups.fwp` provides the Fast Weight Programmer layer.
+ `memorax.equinox.semigroups.lru` provides the Linear Recurrent Unit layer.
+ `memorax.equinox.semigroups.gdn` provides the Gated DeltaNet layer.
+ `memorax.equinox.semigroups.lrnn` provides a basic linear recurrence. 
+ `memorax.equinox.semigroups.mlp` provides an MLP (no memory) for completeness.
+ `memorax.equinox.semigroups.s6` provides the Selective State Space Model (Mamba) layer.
+ `memorax.equinox.semigroups.s6d` provides a diagonal variant of the Selective State Space Model (Mamba) layer.
+ `memorax.equinox.semigroups.spherical` provides Rotational RNN layer (spherical projection).
"""