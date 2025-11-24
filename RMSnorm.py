import jax
import jax.numpy as jnp
import flax.nnx as nnx

class RMSnorm(nnx.Module):
    def __init__(self, dim, eps = 1e-8):
        self.scale = nnx.Param(jnp.ones(dim,))
        self.eps = eps

    def RMS(self, x: jax.Array):
        mean_square = jnp.mean(jnp.square(x))
        return jnp.sqrt(mean_square + self.eps)

    def __call__(self, x):
        X = (x / self.RMS(x)) 
        return X * self.scale