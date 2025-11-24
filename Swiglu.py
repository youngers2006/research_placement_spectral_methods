import jax
import jax.numpy as jnp
import flax.nnx as nnx

class Swiglu(nnx.Module):
    def __init__(self, input_dim, output_dim, rngs: nnx.Rngs):
        self.gate = nnx.Linear(
            in_features=input_dim, 
            out_features=output_dim,
            rngs=rngs
        )
        self.value_net = nnx.Linear(
            in_features=input_dim,
            out_features=output_dim,
            rngs=rngs
        )

    def __call__(self, x):
        hidden = nnx.swish(self.gate(x))
        value = self.value_net(x)
        return hidden * value
    