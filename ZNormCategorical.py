import jax
import jax.numpy as jnp
import flax.nnx as nnx

class ZNormCategorical(nnx.Module):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return (x - self.mu) / self.sigma