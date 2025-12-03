"""This module contains semigroup-based recurrent layers.
They are generally much faster than standard RNNs.

Each RNN type gets its own file.
+ `memax.equinox.semigroups.attn` provides dot-product attention layer. 
+ `memax.equinox.semigroups.delta` provides the DeltaNet layer.
+ `memax.equinox.semigroups.deltap` provides the DeltaProduct layer.
+ `memax.equinox.semigroups.stack` provides framestacking (sliding-window) as an RNN.
+ `memax.equinox.semigroups.fart` provides the Fast AutoRegressive Transformer layer.
+ `memax.equinox.semigroups.ffm` provides the Fast and Forgetful Memory layer.
+ `memax.equinox.semigroups.fwp` provides the Fast Weight Programmer layer.
+ `memax.equinox.semigroups.lru` provides the Linear Recurrent Unit layer.
+ `memax.equinox.semigroups.gdn` provides the Gated DeltaNet layer.
+ `memax.equinox.semigroups.lrnn` provides a basic linear recurrence. 
+ `memax.equinox.semigroups.mlp` provides an MLP (no memory) for completeness.
+ `memax.equinox.semigroups.s6` provides the Selective State Space Model (Mamba) layer.
+ `memax.equinox.semigroups.spherical` provides Rotational RNN layer (spherical projection).
"""