import jax.numpy as jnp
import flax.nnx as nnx

class MahalanobisFilter(nnx.Module):
    def __init__(self, learn_rate, threshold, input_size):
        self.alpha = learn_rate
        self.threshold = threshold
        self.mu = jnp.zeros(input_size, dtype=jnp.float32)
        self.P = jnp.eye(input_size, dtype=jnp.float32)

    def update_distribution(self, x):
        d = x - self.mu
        self.P = (
            (self.P / (1 - self.alpha)) - (self.alpha / (1 - self.alpha)) 
            * (((self.P @ self.d) * (jnp.transpose(d) @ self.P)) 
            / ((1 - self.alpha) + self.alpha * (jnp.transpose(d) @ self.P @ d)))
        ) # sherman morris update
        self.mu = (1 - self.alpha) * self.mu + self.alpha * x

    def get_M_distance(self, x):
        d = x - self.mu
        return jnp.sqrt(jnp.transpose(d) * self.P * d)
    
    def filter(self, x):
        D_m = self.get_M_distance(x)
        self.update_distribution(x)
        is_valid = jnp.where(D_m < self.threshold, 1, 0)
        return is_valid
