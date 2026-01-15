from dataclasses import dataclass
from typing import Callable, Optional, Dict, Tuple, Iterator
from HyperElasticClass2 import HyperElasticRVE  # your frozen class using jax.numpy
import jax
import jax.numpy as np
from RVEAssemblyDirichletBCs import *
#from RVEAssembly_Test import *
import os
import jax
import jax.numpy as np
import numpy as onp          # only for PyVista I/O conversion
import pyvista as pv
            
class MultiRVEComponent:
    """
    Pure-Python container for an M x N x O grid of HyperElasticRVE instances.
    - No NumPy object arrays; just nested lists, so we stay JAX-only for numerics.
    - All tensors live inside each RVE as jax.numpy arrays.
    """
    def __init__(self, dims: Tuple[int, int, int], rve_kwargs: Optional[Dict] = None):
        """
        dims: (M, N, O)
        rve_kwargs: kwargs passed to each HyperElasticRVE() constructor
        """
        self.M, self.N, self.O = dims
        self.rve_kwargs = rve_kwargs or {}
        self.face_cache = {}      # (i,j,k,axis,side) -> dict(pts, B, mask, N)
        self._slices = {}         # (i,j,k) -> slice into the flat coeff vector
        self.dirichlet_bcs = []   # list of DirichletBC at the assembly level

        # Nested Python lists: rves[i][j][k] -> HyperElasticRVE
        self.rves = [
            [
                [HyperElasticRVE(**self.rve_kwargs) for _ in range(self.O)]
                for _ in range(self.N)
            ]
            for _ in range(self.M)
        ]

    def get(self, i: int, j: int, k: int) -> HyperElasticRVE:
        return self.rves[i][j][k]

    def __iter__(self) -> Iterator[HyperElasticRVE]:
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    yield self.rves[i][j][k]

    @classmethod
    def from_index_map(cls, index_map, catalog):
        """
        index_map: (M,N,O) integer array; each entry picks a catalog key
        catalog: dict[int, dict | callable] where dict -> HyperElasticRVE(**dict),
                 or callable -> returns a HyperElasticRVE()
        NOTE: all RVEs must share the same order (same NB) to use batched constraints.
        """
        index_map = np.asarray(index_map)
        M, N, O = map(int, index_map.shape)

        self = cls.__new__(cls)
        self.M, self.N, self.O = M, N, O
        self.rve_kwargs = None
        self.face_cache = {}
        self._slices = {}
        self.dirichlet_bcs = []
        self.rves = []

        for i in range(M):
            row = []
            for j in range(N):
                col = []
                for k in range(O):
                    key = int(index_map[i, j, k].item())
                    spec = catalog[key]
                    if callable(spec):
                        rve = spec()
                    else:
                        rve = HyperElasticRVE(**spec)
                    col.append(rve)
                row.append(col)
            self.rves.append(row)

        # Optional: assert uniform NB to keep batching fast
        NB0 = self.rves[0][0][0].NB
        for i in range(M):
            for j in range(N):
                for k in range(O):
                    assert self.rves[i][j][k].NB == NB0, (
                        "All RVEs must have the same (order) NB for batched constraints."
                    )
        return self

    def precompute_faces(self, N_face: int | None = None, thresh: float | None = None):
        # keep N uniform (batched path assumes same N_face), but threshold per RVE
        r0 = self.rves[0][0][0]
        N  = r0.nq + 1 if N_face is None else N_face
    
        self.N_face = N
        self.face   = {}
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    rve = self.rves[i][j][k]
                    thr_local = 0.5*(rve.E_in + rve.E_out) if thresh is None else thresh
                    for axis, side in (("x","lo"),("x","hi"),("y","lo"),("y","hi"),("z","lo"),("z","hi")):
                        Bf, pts, grids = rve.face_basis(axis, side, N)
                        E_face = rve.material_E(pts)         # or rve.material_E(pts)
                        mask   = (E_face > thr_local).astype(np.float64)
                        self.face[(i,j,k,axis,side)] = {
                            "B": Bf, "pts": pts, "mask": mask, "grids": grids, "N": N
                        }
                                   
    def build_packing(self):
        """Compute flat packing order once and remember slices for each RVE."""
        start = 0
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    rve = self.rves[i][j][k]
                    nblock = 3 * rve.NB
                    self._slices[(i,j,k)] = slice(start, start + nblock)
                    start += nblock
        self._total_dofs = start
    
    def pack_coeffs(self) -> np.ndarray:
        """Concatenate all rve.coeffs into one (total_dofs,) vector."""
        if not self._slices:
            self.build_packing()
        chunks = []
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    A = self.rves[i][j][k].coeffs.reshape(-1)
                    chunks.append(A)
        return np.concatenate(chunks, axis=0)
                        
    def unpack_coeffs(self, A_flat: np.ndarray):
        """Write a flat vector back into each rve.coeffs (view, no copy when possible)."""
        if not self._slices:
            self.build_packing()
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    rve = self.rves[i][j][k]
                    sl = self._slices[(i,j,k)]
                    self.rves[i][j][k].coeffs = A_flat[sl].reshape(3, rve.NB)
                
    def internal_misfits_ms(self, A_flat: np.ndarray) -> np.ndarray:
        assert hasattr(self, "face"), "Call precompute_faces() first."
        if not self._slices:
            self.build_packing()
    
        vals = []
        for (ia,ja,ka),(ib,jb,kb), rA, rB, ax in self.iter_neighbor_pairs():
            sideA, sideB = "hi", "lo"
            cA = self.face[(ia,ja,ka,ax,sideA)]
            cB = self.face[(ib,jb,kb,ax,sideB)]
    
            BfA, mA = cA["B"], cA["mask"][:, None]   # (N^2,NB), (N^2,1)
            BfB, mB = cB["B"], cB["mask"][:, None]
    
            A = A_flat[self._slices[(ia,ja,ka)]].reshape(3, rA.NB)
            B = A_flat[self._slices[(ib,jb,kb)]].reshape(3, rB.NB)
    
            uA = BfA @ A.T
            uB = BfB @ B.T
    
            w    = (mA * mB)                         # (N^2,1)
            num  = np.sum((uA - uB)**2 * w)          # mean-square (no sqrt)
            denom= np.maximum(np.sum(w), 1.0)
            vals.append(num / denom)
    
        return np.stack(vals) if vals else np.zeros((0,))
    
    def external_bc_misfits_ms(self, A_flat: np.ndarray) -> np.ndarray:
        if not self._slices:
            self.build_packing()
        out = []
        for bc in self.dirichlet_bcs:
            if bc.axis == "x":
                i = 0 if bc.side=="lo" else self.M-1
                indices = [(i,j,k) for j in range(self.N) for k in range(self.O)]
            elif bc.axis == "y":
                j = 0 if bc.side=="lo" else self.N-1
                indices = [(i,j,k) for i in range(self.M) for k in range(self.O)]
            else:
                k = 0 if bc.side=="lo" else self.O-1
                indices = [(i,j,k) for i in range(self.M) for j in range(self.N)]
    
            for (i,j,k) in indices:
                rve = self.rves[i][j][k]
                N   = bc.N if bc.N is not None else (rve.nq + 1)
                key = (i,j,k,bc.axis,bc.side)
                cache = self.face.get(key, None)
    
                if cache is None or cache["N"] != N:
                    # build on the fly if cache missing/mismatched
                    Bf, pts, _ = rve.face_basis(bc.axis, bc.side, N)
                    t = 0.5*(rve.E_in + rve.E_out)
                    mask = (rve.material_E(pts) > t).astype(np.float64)
                else:
                    Bf, pts, mask = cache["B"], cache["pts"], cache["mask"]
    
                A = A_flat[self._slices[(i,j,k)]].reshape(3, rve.NB)
                u = Bf @ A.T
    
                pts_global = self._local_to_global_face_pts(i,j,k, pts)
                u_tgt = bc.u_fun(pts_global)
    
                w = mask[:,None] if getattr(bc, "solid_only", True) else np.ones_like(u[:, :1])
                if bc.w_fun is not None:
                    extra = bc.w_fun(pts_global)
                    extra = extra.reshape((-1,1)) if extra.ndim == 1 else extra
                    w = w * extra
    
                num   = np.sum((u - u_tgt)**2 * w)
                denom = np.maximum(np.sum(w), 1.0)
                out.append(num / denom)
    
        return np.stack(out) if out else np.zeros((0,))

    def add_bc(self, bc):
        self.dirichlet_bcs.append(bc)
           
    # Convenience: initialise each RVE (geometry, filtering, basis, jit)
    def initialise_all(self, filter_quadrature: bool = True):
        for rve in self:
            rve.GeomAndQuadrature()
            if filter_quadrature:
                rve.filtered_quadrature()
            rve.build_basis_and_grads(rve.pts)
            rve.prepare_jit()
            
    def _local_to_global_face_pts(self, i,j,k, pts: np.ndarray) -> np.ndarray:
        """
        Map local [0,1]^3 face points of RVE(i,j,k) to component-global [0,1]^3.
        Assumes each RVE spans a unit cell and component spans MxNxO cells.
        """
        gx = (i + pts[:,0]) / self.M
        gy = (j + pts[:,1]) / self.N
        gz = (k + pts[:,2]) / self.O
        return np.stack([gx, gy, gz], axis=1)
    
    def total_energy_FEM(self, A_flat: np.ndarray) -> np.ndarray:
        if not self._slices:
            self.build_packing()
        E = 0.0
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    rve = self.rves[i][j][k]
                    sl = self._slices[(i,j,k)]
                    A = A_flat[sl].reshape(3, rve.NB)
                    E = E + rve.Energy_fn(A)  # each rve.Energy_fn is already jitted
        return E
    
    def total_energy(self, A_flat: np.ndarray, surrogate_model) -> np.ndarray:
        NB = self.rves[0][0][0].NB
        num_rves = self.M * self.N * self.O
        features_per_rve = 3 * NB
        batch_coeffs = A_flat.reshape(num_rves, features_per_rve)
        energy_batch, _ = ml_model.forward_pass(batch_coeffs)
        return jnp.sum(energy_batch)

    def build_objective(self, rho_int=1e3, rho_bc=1e3):
        def obj(A_flat):
            e     = self.total_energy(A_flat)
            ints  = self.internal_misfits_ms(A_flat)
            exts  = self.external_bc_misfits_ms(A_flat)
            return e + rho_int * np.sum(ints) + rho_bc * np.sum(exts)
        self.objective = jax.jit(obj)
        self.grad_objective = jax.jit(jax.grad(obj))

    def iter_with_indices(self) -> Iterator[Tuple[int, int, int, HyperElasticRVE]]:
        """Yield (i, j, k, rve) for every RVE in the grid."""
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    yield i, j, k, self.rves[i][j][k]
                                     
    def _nb(self) -> int:
        """Number of modal coefficients per vector component in each RVE (assume uniform)."""
        return self.rves[0][0][0].NB
    
    def total_energy_from_flat(self, flat: np.ndarray) -> np.ndarray:
        """Sum of per-RVE energies for the packed coeff vector."""
        NB = self._nb()
        tot = 0.0
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O):
                    A = self._coeff_view(flat, (i,j,k))
                    rve = self.rves[i][j][k]
                    tot = tot + rve.Energy_fn(A)  # JAX-scalar
        return tot
    

    def iter_neighbor_pairs(self) -> Iterator[
        Tuple[Tuple[int,int,int], Tuple[int,int,int], HyperElasticRVE, HyperElasticRVE, str]
    ]:
        """
        Yield each adjacent pair once:
          ((i,j,k), (ip,j,k), rve_left, rve_right, 'x')
          ((i,j,k), (i,jp,k), rve_front, rve_back, 'y')
          ((i,j,k), (i,j,kp), rve_bottom, rve_top, 'z')
        Axis flag is 'x'/'y'/'z' for the shared face.
        """
        # x-neighbors
        for i in range(self.M - 1):
            for j in range(self.N):
                for k in range(self.O):
                    yield (i, j, k), (i+1, j, k), self.rves[i][j][k], self.rves[i+1][j][k], 'x'
        # y-neighbors
        for i in range(self.M):
            for j in range(self.N - 1):
                for k in range(self.O):
                    yield (i, j, k), (i, j+1, k), self.rves[i][j][k], self.rves[i][j+1][k], 'y'
        # z-neighbors
        for i in range(self.M):
            for j in range(self.N):
                for k in range(self.O - 1):
                    yield (i, j, k), (i, j, k+1), self.rves[i][j][k], self.rves[i][j][k+1], 'z'
            
            # Objective (energy only)
    def objective_energy(self, A_flat: np.ndarray) -> np.ndarray:
        return self.total_energy(A_flat)
    
    # Internal interface constraints: signed residuals, not RMS 
    def constraints_internal(self, A_flat: np.ndarray) -> np.ndarray:
        """
        Concatenate signed displacement differences (uA - uB) on all interior faces.
        Masked by solid-solid overlap. Shape = sum_faces (N^2*3,).
        Requires precompute_faces() to have been called.
        """
        assert hasattr(self, "face"), "Call precompute_faces() first."
        if not self._slices:
            self.build_packing()
    
        pieces = []
        for (ia,ja,ka),(ib,jb,kb), rA, rB, ax in self.iter_neighbor_pairs():
            sideA, sideB = "hi", "lo"
            cA = self.face[(ia,ja,ka,ax,sideA)]
            cB = self.face[(ib,jb,kb,ax,sideB)]
            BfA, mA = cA["B"], cA["mask"][:, None]   # (N^2,NB), (N^2,1)
            BfB, mB = cB["B"], cB["mask"][:, None]
    
            A = A_flat[self._slices[(ia,ja,ka)]].reshape(3, rA.NB)
            B = A_flat[self._slices[(ib,jb,kb)]].reshape(3, rB.NB)
    
            uA = BfA @ A.T                           # (N^2,3)
            uB = BfB @ B.T                           # (N^2,3)
    
            w   = (mA * mB)                          # (N^2,1)
            res = (uA - uB) * w                      # (N^2,3), signed residual
            pieces.append(res.reshape(-1))           # flatten to (N^2*3,)
    
        return np.concatenate(pieces, axis=0) if pieces else np.zeros((0,))
    
    # External (Dirichlet) constraints: signed residuals
    def constraints_external(self, A_flat: np.ndarray) -> np.ndarray:
        """
        Concatenate signed displacement residuals to targets on all *registered* BC faces.
        Each BC is evaluated on the appropriate outer faces. Shape = sum_bcs (N^2*3,).
        """
        if not self._slices:
            self.build_packing()
        pieces = []
        for bc in self.dirichlet_bcs:
            # which RVEs lie on the exterior face?
            if bc.axis == "x":
                i = 0 if bc.side == "lo" else self.M - 1
                idxs = [(i,j,k) for j in range(self.N) for k in range(self.O)]
            elif bc.axis == "y":
                j = 0 if bc.side == "lo" else self.N - 1
                idxs = [(i,j,k) for i in range(self.M) for k in range(self.O)]
            else:
                k = 0 if bc.side == "lo" else self.O - 1
                idxs = [(i,j,k) for i in range(self.M) for j in range(self.N)]
    
            for (i,j,k) in idxs:
                rve = self.rves[i][j][k]
                N   = bc.N if bc.N is not None else (rve.nq + 1)
                key = (i,j,k,bc.axis,bc.side)
                cache = self.face.get(key, None)
    
                if cache is None or cache["N"] != N:
                    # build on the fly if cache missing/mismatch
                    Bf, pts, _ = rve.face_basis(bc.axis, bc.side, N)
                    t = 0.5 * (rve.E_in + rve.E_out)
                    mask = (rve.materials_E(pts) > t).astype(np.float64)
                else:
                    Bf, pts, mask = cache["B"], cache["pts"], cache["mask"]
    
                A   = A_flat[self._slices[(i,j,k)]].reshape(3, rve.NB)
                u   = Bf @ A.T                                          # (N^2,3)
                pts_g = self._local_to_global_face_pts(i, j, k, pts)    # global [0,1]^3
                u_tgt = bc.u_fun(pts_g)                                 # (N^2,3)
    
                w = mask[:, None] if getattr(bc, "solid_only", True) else np.ones_like(u[:, :1])
                if bc.w_fun is not None:
                    extra = bc.w_fun(pts_g)
                    extra = extra.reshape((-1,1)) if extra.ndim == 1 else extra
                    w = w * extra
    
                res = (u - u_tgt) * w
                pieces.append(res.reshape(-1))
    
        return np.concatenate(pieces, axis=0) if pieces else np.zeros((0,))
    
    # Build jitted constraint functions and their Jacobians 
    def build_constraints(self):
        """
        Produces:
          - self.eq_constraints(A): concatenated equality constraint vector
          - self.jac_eq_constraints(A): full Jacobian (dense) of the above
        """
        def c_all(A_flat):
            ci = self.constraints_internal(A_flat)
            ce = self.constraints_external(A_flat)
            return np.concatenate([ci, ce], axis=0)
    
        self.eq_constraints      = jax.jit(c_all)
        self.jac_eq_constraints  = jax.jit(jax.jacrev(c_all))
        
    def build_ineq_constraints(self, tol_int: float, tol_bc: float):
        """Build smooth inequality constraints: MS - tol^2 <= 0 per face/BC."""
        tol_int2 = float(tol_int)**2
        tol_bc2  = float(tol_bc)**2
    
        def g_all(A_flat):
            ints = self.internal_misfits_ms(A_flat)  # per interior face MS (>=0)
            exts = self.external_bc_misfits_ms(A_flat)  # per BC face MS (>=0)
            g_int = ints - tol_int2
            g_ext = exts - tol_bc2
            return np.concatenate([g_int, g_ext], axis=0)
    
        self.ineq_constraints     = jax.jit(g_all)
        self.jac_ineq_constraints = jax.jit(jax.jacrev(g_all))
        
    def internal_misfits_ms_batched(self, A_flat: np.ndarray) -> np.ndarray:
        if self._int_batch is None:
            return np.zeros((0,))
        pack = self._int_batch
        NB   = pack["NB"]
        A4   = A_flat.reshape(self.M, self.N, self.O, 3, NB)  # (M,N,O,3,NB)
    
        A_A = A4[pack["IA"], pack["JA"], pack["KA"]]  # (Fint, 3, NB)
        A_B = A4[pack["IB"], pack["JB"], pack["KB"]]  # (Fint, 3, NB)
    
        # Batched matmul: (F,N2,NB) @ (F,NB,3) -> (F,N2,3)
        U_A = np.matmul(pack["B_A"], np.swapaxes(A_A, 1, 2))
        U_B = np.matmul(pack["B_B"], np.swapaxes(A_B, 1, 2))
    
        diff  = (U_A - U_B) * pack["W"]
        num   = np.sum(diff**2, axis=(1,2))                      # (F,)
        denom = np.maximum(np.sum(pack["W"], axis=(1,2)), 1.0)   # (F,)
        return num / denom
    
    def external_bc_misfits_ms_batched(self, A_flat: np.ndarray) -> np.ndarray:
        if not self._ext_batches:
            return np.zeros((0,))
        NB = self.rves[0][0][0].NB
        A4 = A_flat.reshape(self.M, self.N, self.O, 3, NB)
    
        pieces = []
        for pack in self._ext_batches:
            A_F = A4[pack["I"], pack["J"], pack["K"]]          # (Fext, 3, NB)
            U   = np.matmul(pack["B"], np.swapaxes(A_F, 1, 2)) # (Fext, N2, 3)
            diff  = (U - pack["u_tgt"]) * pack["W"]
            num   = np.sum(diff**2, axis=(1,2))
            denom = np.maximum(np.sum(pack["W"], axis=(1,2)), 1.0)
            pieces.append(num / denom)
        return np.concatenate(pieces, axis=0)
        
    def misfit_scalar(self, A_flat: np.ndarray, w_int: float = 1.0, w_bc: float = 1.0) -> np.ndarray:
        # Prefer batched paths if prepared
        if hasattr(self, "_int_batch") or hasattr(self, "_ext_batches"):
            ints = self.internal_misfits_ms_batched(A_flat)
            exts = self.external_bc_misfits_ms_batched(A_flat)
        else:
            ints = self.internal_misfits_ms(A_flat)
            exts = self.external_bc_misfits_ms(A_flat)
        return w_int * np.sum(ints) + w_bc * np.sum(exts)   
    
    def misfits_per_face(self, A_flat: np.ndarray) -> np.ndarray:
        """Concatenate per-face mean-square misfits (no sums).
           Order: [all internal faces, then all external-BC faces]"""
        if hasattr(self, "_int_batch") and (self._int_batch is not None):
            ints = self.internal_misfits_ms_batched(A_flat)
        else:
            ints = self.internal_misfits_ms(A_flat)
        exts = (self.external_bc_misfits_ms_batched(A_flat)
                if hasattr(self, "_ext_batches") and self._ext_batches
                else self.external_bc_misfits_ms(A_flat))
        return np.concatenate([ints, exts], axis=0)
        
    def build_single_constraint(self, tol: float, w_int: float = 1.0, w_bc: float = 1.0):
        """
        Builds jitted scalar inequality and its gradient:
          g(A) = tol - (w_int*sum intMS + w_bc*sum extMS)  >= 0
        """
#        print(w_int)
        # Close over weights and tol (static scalars)
        def g(A_flat):
            return tol - self.misfit_scalar(A_flat, w_int=w_int, w_bc=w_bc)
    
        # JIT both; they’ll capture your cached face data as constants
        self.ineq_fun = jax.jit(g)
        self.ineq_jac = jax.jit(jax.grad(g))
         
    # Add to MultiRVEComponent 
    def build_face_batches(self):
        """
        Pre-stack all INTERNAL and EXTERNAL face data so constraint evals
        become a few batched matmuls instead of many tiny loops.
        Call AFTER: initialise_all(), build_packing(), precompute_faces(), add_bc(...)
        """
        assert hasattr(self, "face"), "Call precompute_faces() first."
        NB = self.rves[0][0][0].NB
        N2 = self.N_face * self.N_face
    
        # ---------- INTERNAL faces batch ----------
        int_pairs = list(self.iter_neighbor_pairs())
        Fint = len(int_pairs)
    
        if Fint > 0:
            B_A = np.zeros((Fint, N2, NB))
            B_B = np.zeros((Fint, N2, NB))
            W   = np.zeros((Fint, N2, 1))
            IA  = np.zeros((Fint,), dtype=int)
            JA  = np.zeros((Fint,), dtype=int)
            KA  = np.zeros((Fint,), dtype=int)
            IB  = np.zeros((Fint,), dtype=int)
            JB  = np.zeros((Fint,), dtype=int)
            KB  = np.zeros((Fint,), dtype=int)
    
            for f, ((ia,ja,ka),(ib,jb,kb), rA, rB, ax) in enumerate(int_pairs):
                sideA, sideB = "hi", "lo"
                cA = self.face[(ia,ja,ka,ax,sideA)]
                cB = self.face[(ib,jb,kb,ax,sideB)]
                mA = cA["mask"][:, None]  # (N^2,1)
                mB = cB["mask"][:, None]
                B_A = B_A.at[f].set(cA["B"])
                B_B = B_B.at[f].set(cB["B"])
                W   = W.at[f].set(mA * mB)
                IA = IA.at[f].set(ia);  JA = JA.at[f].set(ja); KA = KA.at[f].set(ka)
                IB = IB.at[f].set(ib);  JB = JB.at[f].set(jb); KB = KB.at[f].set(kb)
    
            self._int_batch = dict(B_A=B_A, B_B=B_B, W=W, IA=IA, JA=JA, KA=KA,
                                   IB=IB, JB=JB, KB=KB, NB=NB, N2=N2)
        else:
            self._int_batch = None
    
        # ---------- EXTERNAL faces batches (one pack per BC) ----------
        self._ext_batches = []  # list of packs, one per registered BC
        for bc in self.dirichlet_bcs:
            # Which tiles on that boundary?
            if bc.axis == "x":
                i = 0 if bc.side == "lo" else self.M - 1
                idxs = [(i,j,k) for j in range(self.N) for k in range(self.O)]
            elif bc.axis == "y":
                j = 0 if bc.side == "lo" else self.N - 1
                idxs = [(i,j,k) for i in range(self.M) for k in range(self.O)]
            else:
                k = 0 if bc.side == "lo" else self.O - 1
                idxs = [(i,j,k) for i in range(self.M) for j in range(self.N)]
    
            Fext = len(idxs)
            if Fext == 0:
                continue
    
            B   = np.zeros((Fext, N2, NB))
            W   = np.zeros((Fext, N2, 1))
            I   = np.zeros((Fext,), dtype=int)
            J   = np.zeros((Fext,), dtype=int)
            K   = np.zeros((Fext,), dtype=int)
            Pts = np.zeros((Fext, N2, 3))
    
            for f, (i,j,k) in enumerate(idxs):
                c  = self.face[(i,j,k,bc.axis,bc.side)]
                B  = B.at[f].set(c["B"])
                # default weight: solid-only
                mask = c["mask"][:, None] if getattr(bc, "solid_only", True) else np.ones((N2,1))
                # optional extra weights
                pts_local = c["pts"]               # (N^2,3) in [0,1]^3 local
                pts_glob  = self._local_to_global_face_pts(i, j, k, pts_local)
                if bc.w_fun is not None:
                    extra = bc.w_fun(pts_glob)
                    extra = extra.reshape((-1,1)) if extra.ndim == 1 else extra
                    mask = mask * extra
                W   = W.at[f].set(mask)
                I   = I.at[f].set(i);  J = J.at[f].set(j);  K = K.at[f].set(k)
                Pts = Pts.at[f].set(pts_glob)
    
            # Precompute target once (constant wrt A)
            # Vectorize u_fun over faces: (Fext, N2, 3)
            u_tgt = jax.vmap(bc.u_fun)(Pts)
    
            self._ext_batches.append(dict(B=B, W=W, I=I, J=J, K=K, Pts=Pts, u_tgt=u_tgt, NB=NB, N2=N2))
            
            self.Fint = 0 if (self._int_batch is None) else int(self._int_batch["B_A"].shape[0])
            self.Fext = sum(pack["B"].shape[0] for pack in self._ext_batches) if self._ext_batches else 0
            self.Ftot = self.Fint + self.Fext
            
            # handy slices to split later if you want
            self._sl_int = slice(0, self.Fint)
            self._sl_ext = slice(self.Fint, self.Ftot)


    


def _tile_slice(idx: int, n: int, ntiles: int) -> tuple[slice, slice]:
    """
    Return (global_slice, local_slice) for tile `idx` along one axis when
    stitching `ntiles` blocks of size `n`, skipping duplicate interior faces.
    - Tile 0 writes [0 : n) using local [0 : n)
    - Tile k>0 writes [k*(n-1)+1 : k*(n-1)+n) using local [1 : n)
    """
    if idx == 0:
        g_sl = slice(0, n)
        l_sl = slice(0, n)
    else:
        start = idx * (n - 1)
        g_sl = slice(start + 1, start + n)  # <-- start+1 so we *skip* the shared interior face
        l_sl = slice(1, n)                  # <-- drop the first local plane
    return g_sl, l_sl

def export_assembled_structured_vts2(
    comp,
    A_flat,
    n_vis_per_rve: int = 17,
    stacked_coords: bool = True,       # True => coordinates span [0..M]×[0..N]×[0..O]
    add_solid_mask: bool = True,
    out_dir: str = "data/assembled",
    stem: str = "final",
):
    """
    Stitches all RVEs into ONE pyvista.StructuredGrid and writes:
      - {stem}_field.vts  : original grid + fields (u, ux, uy, uz, E[, solid])
      - {stem}_warp.vts   : grid warped by u (keeps E)
    """
    if not hasattr(comp, "_slices") or not comp._slices:
        comp.build_packing()
    A_flat = np.asarray(A_flat)
    assert A_flat.shape[0] == comp._total_dofs, "A_flat length doesn't match comp packing."

    n  = int(n_vis_per_rve)
    NX = comp.M * n - (comp.M - 1)
    NY = comp.N * n - (comp.N - 1)
    NZ = comp.O * n - (comp.O - 1)

    # Allocate global arrays
    Xg = np.zeros((NX, NY, NZ))
    Yg = np.zeros((NX, NY, NZ))
    Zg = np.zeros((NX, NY, NZ))
    uX = np.zeros((NX, NY, NZ))
    uY = np.zeros((NX, NY, NZ))
    uZ = np.zeros((NX, NY, NZ))
    Eg = np.zeros((NX, NY, NZ))
    SM = np.zeros((NX, NY, NZ)) if add_solid_mask else None

    # Local structured grid (same for every RVE)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    z = np.linspace(0.0, 1.0, n)

    for i in range(comp.M):
        gx_sl, lx_sl = _tile_slice(i, n, comp.M)
        for j in range(comp.N):
            gy_sl, ly_sl = _tile_slice(j, n, comp.N)
            for k in range(comp.O):
                gz_sl, lz_sl = _tile_slice(k, n, comp.O)

                rve = comp.rves[i][j][k]
                sl  = comp._slices[(i, j, k)]
                Aij = A_flat[sl].reshape(3, rve.NB)

                # Build local grid (ni × nj × nk)
                Xl, Yl, Zl = np.meshgrid(x, y, z, indexing="ij")

                # Local points (flattened F) to evaluate basis once
                pts_local = np.stack([
                    Xl.reshape(-1, order="F"),
                    Yl.reshape(-1, order="F"),
                    Zl.reshape(-1, order="F"),
                ], axis=1)

                # Displacement on local grid from modal basis
                Bv = rve._basis_only(pts_local)             # (n^3, NB)
                u  = (Bv @ Aij.T).reshape(n, n, n, 3, order="F")
                E  = rve.materials_E(pts_local).reshape(n, n, n, order="F")

                # Map local coords into global physical coords
                if stacked_coords:
                    # stacked cubes: 0..M, 0..N, 0..O
                    Gx = i + Xl
                    Gy = j + Yl
                    Gz = k + Zl
                else:
                    # single unit box subdivided:
                    Gx = (i + Xl) / comp.M
                    Gy = (j + Yl) / comp.N
                    Gz = (k + Zl) / comp.O

                # Stitch into globals (skip duplicate interior planes)
                Xg = Xg.at[gx_sl, gy_sl, gz_sl].set(Gx[lx_sl, ly_sl, lz_sl])
                Yg = Yg.at[gx_sl, gy_sl, gz_sl].set(Gy[lx_sl, ly_sl, lz_sl])
                Zg = Zg.at[gx_sl, gy_sl, gz_sl].set(Gz[lx_sl, ly_sl, lz_sl])

                uX = uX.at[gx_sl, gy_sl, gz_sl].set(u[lx_sl, ly_sl, lz_sl, 0])
                uY = uY.at[gx_sl, gy_sl, gz_sl].set(u[lx_sl, ly_sl, lz_sl, 1])
                uZ = uZ.at[gx_sl, gy_sl, gz_sl].set(u[lx_sl, ly_sl, lz_sl, 2])

                Eg = Eg.at[gx_sl, gy_sl, gz_sl].set(E[lx_sl, ly_sl, lz_sl])
                if SM is not None:
                    thr = 0.5 * (rve.E_in + rve.E_out)
                    SM = SM.at[gx_sl, gy_sl, gz_sl].set((E[lx_sl, ly_sl, lz_sl] > thr).astype(np.float64))

    # -------- Build pyvista.StructuredGrid from points (avoid X/Y/Z order traps)
    os.makedirs(out_dir, exist_ok=True)

    # Flatten everything in Fortran order to match VTK point ordering
    Xf = onp.asarray(Xg).ravel(order="F")
    Yf = onp.asarray(Yg).ravel(order="F")
    Zf = onp.asarray(Zg).ravel(order="F")
    Ux = onp.asarray(uX).ravel(order="F")
    Uy = onp.asarray(uY).ravel(order="F")
    Uz = onp.asarray(uZ).ravel(order="F")
    Ee = onp.asarray(Eg).ravel(order="F")
    solid = onp.asarray(SM).ravel(order="F") if SM is not None else None

    points = onp.c_[Xf, Yf, Zf]

    grid = pv.StructuredGrid()
    grid.dimensions = (NX, NY, NZ)          # VTK expects (nx, ny, nz)
    grid.points = points                    # (NX*NY*NZ, 3)

    # Attach point data
    U = onp.c_[Ux, Uy, Uz]
    grid["u"]  = U
    grid["ux"] = Ux
    grid["uy"] = Uy
    grid["uz"] = Uz
    grid["E"]  = Ee
    if solid is not None:
        grid["solid"] = solid

    field_path = os.path.join(out_dir, f"{stem}.vts")
    grid.save(field_path)

    # Warped grid: move points by u
    grid_warp = pv.StructuredGrid()
    grid_warp.dimensions = (NX, NY, NZ)
    grid_warp.points = points + U
    grid_warp["E"] = Ee
    warp_path = os.path.join(out_dir, f"{stem}_warp.vts")
    grid_warp.save(warp_path)

    # Small sanity print
    print(f"[viz] dims=({NX},{NY},{NZ}) | x:[{Xf.min():.3f},{Xf.max():.3f}] "
          f"y:[{Yf.min():.3f},{Yf.max():.3f}] z:[{Zf.min():.3f},{Zf.max():.3f}]")
    print(f"Saved:\n  {field_path}\n  {warp_path}")
    
    
    
def export_assembled_structured_vts(
    comp,
    A_flat,
    n_vis_per_rve: int = 17,
    stacked_coords: bool = True,       # True => coordinates span [0..M]×[0..N]×[0..O]
    add_solid_mask: bool = True,
    out_dir: str = "data/assembled",
    stem: str = "final",
    strain: str | None = None,         # None | "small" | "green"
):
    """
    Stitches all RVEs into ONE pyvista.StructuredGrid and writes:
      - {stem}.vts        : original grid + fields (u, ux, uy, uz, E[, solid][, strains])
      - {stem}_warp.vts   : grid warped by u (keeps E and strains)
    """

    import os
    import numpy as onp
    import pyvista as pv
    import jax.numpy as np

    if not hasattr(comp, "_slices") or not comp._slices:
        comp.build_packing()
    A_flat = np.asarray(A_flat)
    assert A_flat.shape[0] == comp._total_dofs, "A_flat length doesn't match comp packing."

    n  = int(n_vis_per_rve)
    NX = comp.M * n - (comp.M - 1)
    NY = comp.N * n - (comp.N - 1)
    NZ = comp.O * n - (comp.O - 1)

    # Allocate global arrays
    Xg = np.zeros((NX, NY, NZ))
    Yg = np.zeros((NX, NY, NZ))
    Zg = np.zeros((NX, NY, NZ))
    uX = np.zeros((NX, NY, NZ))
    uY = np.zeros((NX, NY, NZ))
    uZ = np.zeros((NX, NY, NZ))
    Eg = np.zeros((NX, NY, NZ))
    SM = np.zeros((NX, NY, NZ)) if add_solid_mask else None

    # Optional strain containers
    export_small = (strain == "small")
    export_green = (strain == "green")
    if export_small:
        exx = np.zeros((NX, NY, NZ)); eyy = np.zeros((NX, NY, NZ)); ezz = np.zeros((NX, NY, NZ))
        exy = np.zeros((NX, NY, NZ)); exz = np.zeros((NX, NY, NZ)); eyz = np.zeros((NX, NY, NZ))
    if export_green:
        # Green–Lagrange 6 comps + full F (9 comps)
        Eg_xx = np.zeros((NX, NY, NZ)); Eg_yy = np.zeros((NX, NY, NZ)); Eg_zz = np.zeros((NX, NY, NZ))
        Eg_xy = np.zeros((NX, NY, NZ)); Eg_xz = np.zeros((NX, NY, NZ)); Eg_yz = np.zeros((NX, NY, NZ))
        Ften = np.zeros((NX, NY, NZ, 9))  # row-major F00..F22

    # Local structured grid (same for every RVE)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    z = np.linspace(0.0, 1.0, n)

    # ---- helper: basis & grads at arbitrary points (no mutation) ----
    def basis_and_grads_at(rve, points):
        # This mirrors HyperElasticRVE.build_basis_and_grads but returns arrays instead of setting attrs
        Px, Py, Pz = rve.order
        xp, yp, zp = points[:,0], points[:,1], points[:,2]
        xh, yh, zh = 2.0*xp - 1.0, 2.0*yp - 1.0, 2.0*zp - 1.0

        def legendre_vals(n, xh):
            V0 = np.ones_like(xh)
            if n == 0: return V0[:, None]
            V1 = xh
            vals = [V0, V1]
            for k in range(1, n):
                Vk1 = ((2*k+1)*xh*vals[-1] - k*vals[-2])/(k+1)
                vals.append(Vk1)
            return np.stack(vals, axis=1)

        def legendre_derivs_from_vals(vals, xhat):
            N, m = vals.shape
            derivs = np.zeros_like(vals)
            if m >= 2:
                n_idx = np.arange(1, m)
                numer = vals[:, :-1] - xhat[:, None] * vals[:, 1:]
                denom = (1 - xhat**2)[:, None]
                derivs = derivs.at[:, 1:].set(n_idx[None, :] * numer / denom)
            return derivs

        Vx = legendre_vals(Px, xh)
        Vy = legendre_vals(Py, yh)
        Vz = legendre_vals(Pz, zh)
        dVx_hat = legendre_derivs_from_vals(Vx, xh)
        dVy_hat = legendre_derivs_from_vals(Vy, yh)
        dVz_hat = legendre_derivs_from_vals(Vz, zh)
        dVx = 2.0 * dVx_hat
        dVy = 2.0 * dVy_hat
        dVz = 2.0 * dVz_hat

        cols = []
        dx_cols, dy_cols, dz_cols = [], [], []
        for p in range(Px+1):
            for q in range(Py+1):
                for r in range(Pz+1):
                    phi = Vx[:, p] * Vy[:, q] * Vz[:, r]
                    dphidx = dVx[:, p] * Vy[:, q] * Vz[:, r]
                    dphidy = Vx[:, p] * dVy[:, q] * Vz[:, r]
                    dphidz = Vx[:, p] * Vy[:, q] * dVz[:, r]
                    cols.append(phi)
                    dx_cols.append(dphidx)
                    dy_cols.append(dphidy)
                    dz_cols.append(dphidz)

        B    = np.stack(cols, axis=1)
        dBdx = np.stack(dx_cols, axis=1)
        dBdy = np.stack(dy_cols, axis=1)
        dBdz = np.stack(dz_cols, axis=1)
        return B, dBdx, dBdy, dBdz

    # ---- stitch all tiles ----
    for i in range(comp.M):
        gx_sl, lx_sl = _tile_slice(i, n, comp.M)
        for j in range(comp.N):
            gy_sl, ly_sl = _tile_slice(j, n, comp.N)
            for k in range(comp.O):
                gz_sl, lz_sl = _tile_slice(k, n, comp.O)

                rve = comp.rves[i][j][k]
                sl  = comp._slices[(i, j, k)]
                Aij = A_flat[sl].reshape(3, rve.NB)

                # Build local grid (ni × nj × nk)
                Xl, Yl, Zl = np.meshgrid(x, y, z, indexing="ij")

                # Local points (flattened F) to evaluate basis once
                pts_local = np.stack([
                    Xl.reshape(-1, order="F"),
                    Yl.reshape(-1, order="F"),
                    Zl.reshape(-1, order="F"),
                ], axis=1)

                # Basis at vis points
                Bv, dBdx, dBdy, dBdz = basis_and_grads_at(rve, pts_local)

                # Displacement on local grid from modal basis
                u  = (Bv @ Aij.T).reshape(n, n, n, 3, order="F")
                E  = rve.material_E(pts_local).reshape(n, n, n, order="F")

                # Grad u (each component shape (n^3,))
                du_dx = (dBdx @ Aij.T)  # (n^3, 3)
                du_dy = (dBdy @ Aij.T)  # (n^3, 3)
                du_dz = (dBdz @ Aij.T)  # (n^3, 3)

                # Map local coords into global physical coords
                if stacked_coords:
                    Gx = i + Xl; Gy = j + Yl; Gz = k + Zl
                else:
                    Gx = (i + Xl) / comp.M
                    Gy = (j + Yl) / comp.N
                    Gz = (k + Zl) / comp.O

                # Stitch into globals (skip duplicate interior planes)
                Xg = Xg.at[gx_sl, gy_sl, gz_sl].set(Gx[lx_sl, ly_sl, lz_sl])
                Yg = Yg.at[gx_sl, gy_sl, gz_sl].set(Gy[lx_sl, ly_sl, lz_sl])
                Zg = Zg.at[gx_sl, gy_sl, gz_sl].set(Gz[lx_sl, ly_sl, lz_sl])

                uX = uX.at[gx_sl, gy_sl, gz_sl].set(u[lx_sl, ly_sl, lz_sl, 0])
                uY = uY.at[gx_sl, gy_sl, gz_sl].set(u[lx_sl, ly_sl, lz_sl, 1])
                uZ = uZ.at[gx_sl, gy_sl, gz_sl].set(u[lx_sl, ly_sl, lz_sl, 2])

                Eg = Eg.at[gx_sl, gy_sl, gz_sl].set(E[lx_sl, ly_sl, lz_sl])
                if SM is not None:
                    thr = 0.5 * (rve.E_in + rve.E_out)
                    SM = SM.at[gx_sl, gy_sl, gz_sl].set((E[lx_sl, ly_sl, lz_sl] > thr).astype(np.float64))

                # ---- strains (optional) ----
                if export_small or export_green:
                    # reshape grads to (n,n,n,3)
                    dudx = du_dx.reshape(n, n, n, 3, order="F")
                    dudy = du_dy.reshape(n, n, n, 3, order="F")
                    dudz = du_dz.reshape(n, n, n, 3, order="F")

                    # grad u as 3x3 at each point (row = component of x, col = component of u)
                    # Here: ∂u_i/∂x_j => rows are x,y,z; columns are ux,uy,uz
                    # We'll build F = I + grad_u with F_ij = δ_ij + ∂u_i / ∂x_j
                    # Extract components:
                    ux_x, uy_x, uz_x = dudx[...,0], dudx[...,1], dudx[...,2]
                    ux_y, uy_y, uz_y = dudy[...,0], dudy[...,1], dudy[...,2]
                    ux_z, uy_z, uz_z = dudz[...,0], dudz[...,1], dudz[...,2]

                    if export_small:
                        # ε = 0.5*(∇u + ∇u^T)
                        e_xx = ux_x
                        e_yy = uy_y
                        e_zz = uz_z
                        e_xy = 0.5*(ux_y + uy_x)
                        e_xz = 0.5*(ux_z + uz_x)
                        e_yz = 0.5*(uy_z + uz_y)

                        exx = exx.at[gx_sl, gy_sl, gz_sl].set(e_xx[lx_sl, ly_sl, lz_sl])
                        eyy = eyy.at[gx_sl, gy_sl, gz_sl].set(e_yy[lx_sl, ly_sl, lz_sl])
                        ezz = ezz.at[gx_sl, gy_sl, gz_sl].set(e_zz[lx_sl, ly_sl, lz_sl])
                        exy = exy.at[gx_sl, gy_sl, gz_sl].set(e_xy[lx_sl, ly_sl, lz_sl])
                        exz = exz.at[gx_sl, gy_sl, gz_sl].set(e_xz[lx_sl, ly_sl, lz_sl])
                        eyz = eyz.at[gx_sl, gy_sl, gz_sl].set(e_yz[lx_sl, ly_sl, lz_sl])

                    if export_green:
                        # F = I + grad u
                        F00 = 1.0 + ux_x; F01 = ux_y;       F02 = ux_z
                        F10 = uy_x;       F11 = 1.0 + uy_y; F12 = uy_z
                        F20 = uz_x;       F21 = uz_y;       F22 = 1.0 + uz_z

                        # C = F^T F
                        C00 = F00*F00 + F10*F10 + F20*F20
                        C01 = F00*F01 + F10*F11 + F20*F21
                        C02 = F00*F02 + F10*F12 + F20*F22
                        C11 = F01*F01 + F11*F11 + F21*F21
                        C12 = F01*F02 + F11*F12 + F21*F22
                        C22 = F02*F02 + F12*F12 + F22*F22

                        # Green–Lagrange E = 0.5*(C - I)
                        Eg_xx_loc = 0.5*(C00 - 1.0)
                        Eg_yy_loc = 0.5*(C11 - 1.0)
                        Eg_zz_loc = 0.5*(C22 - 1.0)
                        Eg_xy_loc = 0.5*(C01)  # C is symmetric
                        Eg_xz_loc = 0.5*(C02)
                        Eg_yz_loc = 0.5*(C12)

                        Eg_xx = Eg_xx.at[gx_sl, gy_sl, gz_sl].set(Eg_xx_loc[lx_sl, ly_sl, lz_sl])
                        Eg_yy = Eg_yy.at[gx_sl, gy_sl, gz_sl].set(Eg_yy_loc[lx_sl, ly_sl, lz_sl])
                        Eg_zz = Eg_zz.at[gx_sl, gy_sl, gz_sl].set(Eg_zz_loc[lx_sl, ly_sl, lz_sl])
                        Eg_xy = Eg_xy.at[gx_sl, gy_sl, gz_sl].set(Eg_xy_loc[lx_sl, ly_sl, lz_sl])
                        Eg_xz = Eg_xz.at[gx_sl, gy_sl, gz_sl].set(Eg_xz_loc[lx_sl, ly_sl, lz_sl])
                        Eg_yz = Eg_yz.at[gx_sl, gy_sl, gz_sl].set(Eg_yz_loc[lx_sl, ly_sl, lz_sl])

                        # Pack F into 9-comp vector for VTK (row-major)
                        Fpack = np.stack([
                            F00, F01, F02,
                            F10, F11, F12,
                            F20, F21, F22
                        ], axis=-1)  # (n,n,n,9)
                        Ften = Ften.at[gx_sl, gy_sl, gz_sl, :].set(Fpack[lx_sl, ly_sl, lz_sl, :])

    # -------- Build pyvista.StructuredGrid from points (avoid X/Y/Z order traps)
    os.makedirs(out_dir, exist_ok=True)

    # Flatten everything in Fortran order to match VTK point ordering
    Xf = onp.asarray(Xg).ravel(order="F")
    Yf = onp.asarray(Yg).ravel(order="F")
    Zf = onp.asarray(Zg).ravel(order="F")
    Ux = onp.asarray(uX).ravel(order="F")
    Uy = onp.asarray(uY).ravel(order="F")
    Uz = onp.asarray(uZ).ravel(order="F")
    Ee = onp.asarray(Eg).ravel(order="F")
    solid = onp.asarray(SM).ravel(order="F") if SM is not None else None

    points = onp.c_[Xf, Yf, Zf]

    grid = pv.StructuredGrid()
    grid.dimensions = (NX, NY, NZ)          # VTK expects (nx, ny, nz)
    grid.points = points                    # (NX*NY*NZ, 3)

    # Attach point data
    U = onp.c_[Ux, Uy, Uz]
    grid["u"]  = U
    grid["ux"] = Ux
    grid["uy"] = Uy
    grid["uz"] = Uz
    grid["E"]  = Ee
    if solid is not None:
        grid["solid"] = solid

    # Strain output
    if export_small:
        grid["exx"] = onp.asarray(exx).ravel(order="F")
        grid["eyy"] = onp.asarray(eyy).ravel(order="F")
        grid["ezz"] = onp.asarray(ezz).ravel(order="F")
        grid["exy"] = onp.asarray(exy).ravel(order="F")
        grid["exz"] = onp.asarray(exz).ravel(order="F")
        grid["eyz"] = onp.asarray(eyz).ravel(order="F")

    if export_green:
        grid["E_xx"] = onp.asarray(Eg_xx).ravel(order="F")
        grid["E_yy"] = onp.asarray(Eg_yy).ravel(order="F")
        grid["E_zz"] = onp.asarray(Eg_zz).ravel(order="F")
        grid["E_xy"] = onp.asarray(Eg_xy).ravel(order="F")
        grid["E_xz"] = onp.asarray(Eg_xz).ravel(order="F")
        grid["E_yz"] = onp.asarray(Eg_yz).ravel(order="F")
        grid["F"]    = onp.asarray(Ften).reshape(-1, 9, order="F")

    field_path = os.path.join(out_dir, f"{stem}.vts")
    grid.save(field_path)

    # Warped grid: move points by u (reuse same data arrays)
    grid_warp = pv.StructuredGrid()
    grid_warp.dimensions = (NX, NY, NZ)
    grid_warp.points = points + U
    grid_warp["E"] = Ee
    if export_small:
        grid_warp["exx"] = grid["exx"]; grid_warp["eyy"] = grid["eyy"]; grid_warp["ezz"] = grid["ezz"]
        grid_warp["exy"] = grid["exy"]; grid_warp["exz"] = grid["exz"]; grid_warp["eyz"] = grid["eyz"]
    if export_green:
        grid_warp["E_xx"] = grid["E_xx"]; grid_warp["E_yy"] = grid["E_yy"]; grid_warp["E_zz"] = grid["E_zz"]
        grid_warp["E_xy"] = grid["E_xy"]; grid_warp["E_xz"] = grid["E_xz"]; grid_warp["E_yz"] = grid["E_yz"]
        grid_warp["F"]    = grid["F"]

    warp_path = os.path.join(out_dir, f"{stem}_warp.vts")
    grid_warp.save(warp_path)

    print(f"[viz] dims=({NX},{NY},{NZ}) | x:[{Xf.min():.3f},{Xf.max():.3f}] "
          f"y:[{Yf.min():.3f},{Yf.max():.3f}] z:[{Zf.min():.3f},{Zf.max():.3f}]")
    print(f"Saved:\n  {field_path}\n  {warp_path}")
                
if __name__ == "__main__":
    # index map: shape (M,N,O)
    index_map = np.array([
        [[0,1,1,0,2],
         [0,0,1,2,2]],
    ], dtype=int)  # (1,2,5)
    
    # catalog: each id -> kwargs for HyperElasticRVE
    catalog = {
        0: dict(order=(6,6,6), nq=50, geom="gyroid", wall_thickness=0.30,
                E_in=12.0, E_out=1e-3, nu=0.33),
        1: dict(order=(6,6,6), nq=50, geom="gyroid", wall_thickness=0.18,
                E_in=15.0, E_out=1e-3, nu=0.30),
        2: dict(order=(6,6,6), nq=50, geom="solid",  wall_thickness=1.00,
                E_in=10.0, E_out=1e-3, nu=0.30),
    }
    
    comp = MultiRVEComponent.from_index_map(index_map, catalog)
    comp.initialise_all(filter_quadrature=True)
    comp.build_packing()
    comp.precompute_faces(N_face=50)  # defaults to rve.nq+1
    
    # Define BC targets (global coords are normalized to [0,1]^3 by _local_to_global_face_pts)
    def zero_disp(pts):
        # Fully constrained: ux = uy = uz = 0
        return np.zeros((pts.shape[0], 3))
    
    def top_compression(pts, mag=1.0):
        # Uniform compression in +z direction ⇒ negative uz
        N = pts.shape[0]
        return np.stack([
            np.zeros((N,)),     # ux = 0 (tangential fixed; change if you want them free)
            np.zeros((N,)),     # uy = 0
            -mag * np.ones((N,))  # uz = -mag
        ], axis=1)
    
    # Register BCs (clear existing if you’re reusing the object)
    comp.dirichlet_bcs = []
    comp.add_bc(DirichletBC(axis="z", side="lo", u_fun=zero_disp))       # z = 0 face fixed
    comp.add_bc(DirichletBC(axis="z", side="hi", u_fun=top_compression)) # z = top compressed
    
        # Build energy-only obj + constraints
    comp.build_constraints()  # creates comp.eq_constraints and comp.jac_eq_constraints
    
    A0 = comp.pack_coeffs()
    
    # Energy-only
    E = comp.total_energy(A0)  # or comp.objective_energy(A0) if you kept that wrapper
    print("Energy-only objective:", float(E))
    
    # Equality constraints vector (internal + BCs), and its norm
    c = comp.eq_constraints(A0)
    print("‖constraints‖₂:", float(np.linalg.norm(c)))
    
    # If you still want to *inspect* the penalty pieces without using them in the obj:
    ints = comp.internal_misfits_ms(A0)   # mean-square per interior face
    exts = comp.external_bc_misfits_ms(A0)  # mean-square per BC patch
    print("sum(intMS)=", float(np.sum(ints)), "sum(extMS)=", float(np.sum(exts)))