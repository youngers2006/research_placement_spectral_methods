import jax
import jax.numpy as jnp
import flax.nnx as nnx
from Swiglu import Swiglu
from RMSnorm import RMSnorm
from ZNormCategorical import ZNormCategorical

class GatedResidualBlock(nnx.Module):
    def __init__(self, dim, rngs):
        self.swiglu_layer = Swiglu(
            input_dim=dim,
            output_dim=dim,
            rngs=rngs
        )
        self.rms_norm_layer = RMSnorm(
            dim=dim,
            eps=1e-8
        )

    def __call__(self, x):
        x_norm = self.rms_norm_layer(x)
        fx = self.swiglu_layer(x_norm)
        return x + fx

class GatedResnet(nnx.Module):
    def __init__(self, input_dim, encoded_dim, output_dim, N, mu, sigma, rngs):
        self.z_scaler = ZNormCategorical(
            mu,
            sigma
        )
        self.encoder = nnx.Linear(
            input_dim,
            encoded_dim,
            rngs=rngs
        )
        layers = []
        for _ in range(N):
            layers.append(
                GatedResidualBlock(
                    encoded_dim,
                    rngs=rngs
                )
            )
        self.network = nnx.Sequential(*layers)
        self.prediction_head = nnx.Linear(
            encoded_dim,
            output_dim,
            rngs=rngs
        )

    def __call__(self, x):
        x = self.z_scaler(x)
        x = self.encoder(x)
        x = self.network(x)
        return self.prediction_head(x)