import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.nnx as nnx

class Encoder(nnx.Module):
    """
    Takes (batch size, num points, features) and operates on the last axis to form (batch size, num points, encoded dim)
    """
    def __init__(
            self, 
            input_dim, 
            enc_h1_dim, 
            enc_h2_dim, 
            latent_dim, 
            rngs: nnx.Rngs
        ):
        self.encoder_network = nnx.Sequential(
            nnx.Linear(input_dim, enc_h1_dim, rngs=rngs), 
            nnx.silu(),
            nnx.Linear(enc_h1_dim, enc_h2_dim, rngs=rngs),
            nnx.silu(),
            nnx.Linear(enc_h2_dim, latent_dim, rngs=rngs)
        )
    def __call__(self, x):
        return self.encoder_network(x)

class Decoder(nnx.Module):
    def __init__(
            self, 
            latent_dim, 
            dec_h1_dim, 
            dec_h2_dim, 
            original_dim, 
            rngs: nnx.Rngs
        ):
        self.decoder_network = nnx.Sequential(
            nnx.Linear(latent_dim, dec_h1_dim, rngs=rngs), 
            nnx.silu(),
            nnx.Linear(dec_h1_dim, dec_h2_dim, rngs=rngs),
            nnx.silu(),
            nnx.Linear(dec_h2_dim, original_dim * 2, rngs=rngs)
        )
    def __call__(self, x):
        raw_output = self.decoder_network(x)
        mean, log_var = jnp.split(raw_output, 2, axis=-1)
        return mean, log_var

class FourierNeuralLayer(nnx.Module):
    def __init__(
            self, 
            X_dim, 
            Y_dim,
            Z_dim,
            modes_x, # number of x modes to keep
            modes_y, # number of y modes to keep
            modes_z, # number of z modes to keep
            channels, # (latent dim)
            rngs: nnx.Rngs
        ):
        self.X_dim = X_dim
        self.Y_dim = Y_dim
        self.Z_dim = Z_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.modes_z_compressed = (modes_z // 2) + 1
        initialiser = nnx.initializers.lecun_normal()
        self.W = nnx.Linear(channels, channels, use_bias=False, rngs=rngs)
        real_key, imag_key = jax.random.split(rngs.params(), 2)
        r_shape = (modes_x, modes_y, self.modes_z_compressed, channels)
        scale = 1.0 / (channels * channels)
        R_real = initialiser(real_key, r_shape) * scale 
        R_imaginary = initialiser(imag_key, r_shape) * scale
        self.R = nnx.Param(R_real + 1j * R_imaginary)
    
    def __call__(self, vt: jax.Array) -> jax.Array:
        skip_connect = self.W(vt)
        vt_freq = jnp.fft.rfftn(vt, axes=(1,2,3))
        B, X_freq, Y_freq, Z_freq, C = vt_freq.shape
        convoluted_nodes = jnp.zeros(
            (B, X_freq, Y_freq, Z_freq, C), 
            dtype=vt_freq.dtype
        )
        multiplied_nodes = self.R * vt_freq[
            :,:self.modes_x,:self.modes_y,:self.modes_z_compressed,:
        ]
        convoluted_nodes = convoluted_nodes.at[
            :,:self.modes_x,:self.modes_y,:self.modes_z_compressed,:
        ].set(multiplied_nodes)
        convoluted_kernel = jnp.fft.irfftn(
            convoluted_nodes, 
            s=(self.X_dim, self.Y_dim, self.Z_dim),
            axes=(1,2,3)
        )
        return nnx.silu(convoluted_kernel + skip_connect)

class FourierNeuralOperator(nnx.Module):
    def __init__( # a: (B, X, Y, Z, C) , v: (B, X, Y, Z, L)
            self,
            X_dim,
            Y_dim,
            Z_dim,
            physical_channels,
            latent_channels,
            encoder_h1, 
            encoder_h2,
            num_fno_layers,
            modes_x,
            modes_y,
            modes_z,
            decoder_h1,
            decoder_h2,
            rngs
        ):
        self.encoder = Encoder(
            physical_channels,
            encoder_h1,
            encoder_h2,
            latent_channels,
            rngs=rngs
            )
        fno_layers = []
        for _ in range(num_fno_layers):
            fno_layers.append(
                FourierNeuralLayer(
                    X_dim,
                    Y_dim,
                    Z_dim,
                    modes_x,
                    modes_y,
                    modes_z,
                    latent_channels, 
                    rngs=rngs
                )
            )
        self.fourier_layers = nnx.Sequential(*fno_layers)
        self.decoder = Decoder(
            latent_channels,
            decoder_h1,
            decoder_h2,
            physical_channels,
            rngs=rngs
        )

    def __call__(self, a):
        v0 = self.encoder(a)
        vn = self.fourier_layers(v0)
        mean_vn, log_variance = self.decoder(vn)
        variance = jnp.exp(log_variance)
        return mean_vn, variance