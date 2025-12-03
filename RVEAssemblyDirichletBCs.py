from dataclasses import dataclass
from typing import Callable, Optional
from typing import Optional, Dict, Tuple, Iterator
from HyperElasticClass2 import HyperElasticRVE  # your frozen class using jax.numpy
import jax
import jax.numpy as np

from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as np

@dataclass(frozen=True, slots=True)
class DirichletBC:
    axis: str   # 'x' | 'y' | 'z'
    side: str   # 'lo' | 'hi'
    u_fun: Callable[[np.ndarray], np.ndarray]        # global pts -> target u, (N^2,3)
    N: Optional[int] = None                          # samples per face tile (defaults to rve.nq+1)
    solid_only: bool = True
    thresh: Optional[float] = None                   # default = 0.5*(E_in+E_out) if None
    w_fun: Optional[Callable[[np.ndarray], np.ndarray]] = None  # optional extra weights (N^2,) or (N^2,1)

    def __post_init__(self):
        if self.axis not in ("x", "y", "z"):
            raise ValueError("axis must be 'x','y','z'")
        if self.side not in ("lo", "hi"):
            raise ValueError("side must be 'lo' or 'hi'")

def compress_top(pts: np.ndarray) -> np.ndarray:
    # Global compression of -0.2 in z on the z=hi plane
    N = pts.shape[0]
    return np.stack([np.zeros((N,)), np.zeros((N,)), -1.0*np.ones((N,))], axis=1)

def mid_strip_weight(pts: np.ndarray, width=0.3) -> np.ndarray:
    # Keep a central band on the face (global coords in [0,1])
    s = pts[:,0]  # use x as 's' parameter for example
    t = pts[:,1]  # and y as 't'
    a = (1.0 - width)/2.0
    b = 1.0 - a
    mask = np.logical_and((s >= a) & (s <= b), (t >= a) & (t <= b))
    return mask.astype(np.float64)  # (N^2,)