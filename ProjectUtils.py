import os
import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax.nnx as nnx
from flax import struct
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
from typing import Any
import jraph
import optax
from dataclasses import dataclass
from functools import partial

@jax.jit
def restitch(idx_1, idx_2, array1: jax.Array, array2: jax.Array) -> jax.Array:
    "Takes 2 arrays that have been separated from eachother and recombines them"
    length = idx_1.shape[0] + idx_2.shape[0]
    output_shape = (length,) + array1.shape[1:]
    stitched_array = jnp.zeros(shape=output_shape, dtype=jnp.float32)
    stitched_array = stitched_array.at[idx_1].set(array1).at[idx_2].set(array2)
    return stitched_array