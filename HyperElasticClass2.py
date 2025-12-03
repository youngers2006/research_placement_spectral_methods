import os

import jax
import jax.numpy as np
# Ensure double precision everywhere for SciPy Fortran SLSQP
jax.config.update("jax_enable_x64", True)
from jax import grad
import time

class HyperElasticRVE():
    def __init__(self, *, order=(4,4,4), nq=50, barrier=1e-6, alpha=1e-6,
                 geom="gyroid", wall_thickness=0.3,
                 E_in=10.0, E_out=1e-3, nu=0.3):
        self.order = order
        self.nq = nq
        self.NB = (order[0]+1)*(order[1]+1)*(order[2]+1)

        # material / geometry per-instance
        self.geom = geom
        self.wall_thickness = wall_thickness
        self.E_in = E_in
        self.E_out = E_out
        self.nu = nu

        self.coeffs = np.zeros((3, self.NB))
        self.barrier = barrier
        self.alpha   = alpha

    @staticmethod
    def _face_axis_check(axis: str):
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be one of {'x','y','z'}")

    @staticmethod
    def _face_side_check(side: str):
        if side not in ("lo", "hi"):
            raise ValueError("side must be 'lo' or 'hi'")

    def face_param_grid(self, N: int):
        """Return 2D parameter grid S, T in [0,1] (N x N), and stacked st points (N^2,2)."""
        g = np.linspace(0.0, 1.0, N)
        S, T = np.meshgrid(g, g, indexing="ij")      # (N,N)
        st = np.stack([S.ravel(), T.ravel()], axis=1)  # (N^2,2)
        return S, T, st

    def face_points(self, axis: str, side: str, N: int = None):
        """
        Build face point coordinates on the unit cube for a given axis/side.
        axis: 'x'|'y'|'z', side: 'lo' (0) or 'hi' (1)
        Returns:
          pts_face : (N^2,3) JAX array
          (Xf, Yf, Zf) : each (N,N) grid, useful for structured output
        """
        if N is None:
            N = self.nq + 1
        self._face_axis_check(axis)
        self._face_side_check(side)

        S, T, st = self.face_param_grid(N)  # S,T are (N,N) JAX arrays
        zero = np.zeros_like(S)
        one  = np.ones_like(S)

        if axis == "x":
            Xf = zero if side == "lo" else one
            Yf = S
            Zf = T
        elif axis == "y":
            Xf = S
            Yf = zero if side == "lo" else one
            Zf = T
        else:  # axis == "z"
            Xf = S
            Yf = T
            Zf = zero if side == "lo" else one

        pts_face = np.stack([Xf.ravel(), Yf.ravel(), Zf.ravel()], axis=1)  # (N^2,3)
        return pts_face, (Xf, Yf, Zf)
       
    @staticmethod
    def _face_normal(axis: str, side: str):
        # Only needed if you want normal-only misfit.
        # Sign doesn’t matter if you compare magnitudes.
        if axis == "x":
            return np.array([1.0, 0.0, 0.0])
        if axis == "y":
            return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])  # axis == "z"
        
    def _basis_only(self, points):
        """Build tensor-product Legendre basis (no grads) at arbitrary points."""
        Px, Py, Pz = self.order
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        xh, yh, zh = self.map_to_hat(x), self.map_to_hat(y), self.map_to_hat(z)

        Vx = self.legendre_vals(Px, xh)  # (N, Px+1)
        Vy = self.legendre_vals(Py, yh)  # (N, Py+1)
        Vz = self.legendre_vals(Pz, zh)  # (N, Pz+1)

        cols = []
        for p in range(Px + 1):
            for q in range(Py + 1):
                for r in range(Pz + 1):
                    cols.append(Vx[:, p] * Vy[:, q] * Vz[:, r])
        return np.stack(cols, axis=1)  # (N, NB)

    def face_basis(self, axis: str, side: str, N: int = None):
        """
        Basis matrix on a specified face.
        Returns:
          Bf      : (N^2, NB)
          pts_face, (Xf, Yf, Zf)
        """
        if N is None:
            N = self.nq + 1
        pts_face, grids = self.face_points(axis, side, N)
        Bf = self._basis_only(pts_face)
        return Bf, pts_face, grids

    def face_solid_mask(self, axis: str, side: str, N: int = None, thresh: float = None):
        """
        Boolean mask on a face using the same geometry definition as the volume.
        thresh defaults to the midpoint of (E_in, E_out).
        """
        if N is None:
            N = self.nq + 1
        if thresh is None:
            thresh = 0.5 * (self.E_in + self.E_out)
    
        pts_face, _ = self.face_points(axis, side, N)
        E_face = self.gyroid_E(pts_face)      # pure eval, no side-effects
        return E_face > thresh                # (N^2,)

    def face_displacement(self, coeffs=None, axis: str = "x", side: str = "lo", N: int = None):
        """
        Displacement on a face u = Bf @ coeffs.T.
        """
        if N is None:
            N = self.nq + 1
        if coeffs is None:
            coeffs = self.coeffs
        Bf, pts_face, grids = self.face_basis(axis, side, N)
        uf = Bf @ coeffs.T  # (N^2, 3)
        return uf, pts_face, grids

    @staticmethod
    def legendre_vals(n, xhat):
        """Return array V of shape (N, n+1) with V[:,k] = P_k(xhat)."""
        xhat = np.asarray(xhat)
        V0 = np.ones_like(xhat)
        if n == 0:
            return V0[:, None]
        V1 = xhat
        vals = [V0, V1]
        for k in range(1, n):
            Vk1 = ((2*k+1)*xhat*vals[-1] - k*vals[-2])/(k+1)
            vals.append(Vk1)
        return np.stack(vals, axis=1)

    @staticmethod
    def legendre_derivs_from_vals2(vals, xhat):
        """Given vals (N, n+1) and xhat (N,), return d/dxhat of each P_k at xhat.
        Uses formula: P'_n(x) = n/(x^2-1) * (P_{n-1}(x) - x P_n(x)) for n>=1, P'_0=0.
        """
        xhat = np.asarray(xhat)
        N, m = vals.shape  # m = n+1
        derivs = np.zeros_like(vals)
        # n=0 -> 0
        if m >= 2:
            n_idx = np.arange(1, m)
            # broadcast-safe
            numer = vals[:, :-1] - xhat[:, None] * vals[:, 1:]
            denom = (1-xhat**2)[:, None]
            derivs = derivs.at[:, 1:].set(n_idx[None, :] * numer / denom)
        return derivs

    @staticmethod
    def legendre_derivs_from_vals(vals, xhat, eps=1e-12):
        """
        Given vals (N, n+1) and xhat (N,), return d/dxhat P_k(xhat).
        Stable at endpoints using exact limits:
          P'_n(1)  = n(n+1)/2
          P'_n(-1) = (-1)^(n-1) * n(n+1)/2
        """
        xhat = np.asarray(xhat)
        N, m = vals.shape  # m = n+1
        derivs = np.zeros_like(vals)
    
        if m >= 2:
            n_idx = np.arange(1, m)                 # 1..n
            numer = vals[:, :-1] - xhat[:, None] * vals[:, 1:]
            denom = (1.0 - xhat**2)[:, None]
    
            # core formula away from endpoints
            safe = np.abs(denom) > eps
            core = np.zeros((N, m-1), dtype=vals.dtype)
            core = core.at[safe.squeeze(), :].set(
                n_idx[None, :] * numer[safe.squeeze(), :] / denom[safe.squeeze(), :]
            )
    
            # endpoint fixes
            # at +1
            at_p1 = xhat > (1.0 - eps)
            if np.any(at_p1):
                v = 0.5 * n_idx * (n_idx + 1.0)  # P'_n(1)
                core = core.at[at_p1, :].set(v[None, :])
    
            # at -1
            at_m1 = xhat < (-1.0 + eps)
            if np.any(at_m1):
                # (-1)^(n-1) = +1 for n odd, -1 for n even
                sgn = np.where((n_idx % 2) == 1, 1.0, -1.0)
                v = 0.5 * n_idx * (n_idx + 1.0) * sgn
                core = core.at[at_m1, :].set(v[None, :])
    
            derivs = derivs.at[:, 1:].set(core)
    
        return derivs

    @staticmethod
    def map_to_hat(p):
        return 2.0 * p - 1.0
    
    @staticmethod
    def neo_hookean_energy_core(coeffs, B, dBdx, dBdy, dBdz, E, nu, w, barrier, alpha):
        du_dx = dBdx @ coeffs.T
        du_dy = dBdy @ coeffs.T
        du_dz = dBdz @ coeffs.T
    
        I = np.eye(3)
        F = np.stack([
            np.stack([I[0,0] + du_dx[:,0],     du_dy[:,0],         du_dz[:,0]], axis=1),
            np.stack([du_dx[:,1],             I[1,1] + du_dy[:,1], du_dz[:,1]], axis=1),
            np.stack([du_dx[:,2],             du_dy[:,2],         I[2,2] + du_dz[:,2]], axis=1)
        ], axis=1)
    
        # after you build F
        J = np.linalg.det(F)
        J_safe = np.maximum(J, 1e-12)          # or use your barrier epsilon
        
        C  = np.einsum('...ji,...jk->...ik', F, F)
        I1 = np.trace(C, axis1=1, axis2=2)
        
        mu    = E / (2.0*(1.0+nu))
        kappa = E / (3.0*(1.0-2.0*nu))
        
        Jm23 = J_safe**(-2.0/3.0)              # <= clamp avoids NaN
        W0   = 0.5 * mu * (Jm23 * I1 - 3.0) + 0.5 * kappa * (J - 1.0)**2
        
        # keep your barrier as-is (it already branches safely on J)
        W_bar = np.where(J > barrier, -alpha * np.log(J),
                         alpha * ((barrier - J)**2 / barrier**2))
        W = W0 + W_bar
        return np.sum(W * w)
    
    @staticmethod
    def boundary_disp(coeffs, B):
        return B @ coeffs.T
   
    def unit_cube_quadrature(self):
        """Return (points, weights) for a midpoint tensor grid with nq per axis."""
    #    nq = nq
        grid = (np.arange(self.nq) + 0.5)/self.nq
        X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')
        self.pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        # print(X.ravel().shape)
        # print(nq)
        self.w = np.full((self.pts.shape[0],), (1.0/self.nq)**3)
    
    def gyroid_E(self, points):
        # use instance params (no kwargs)
        x, y, z = points[:,0], points[:,1], points[:,2]
        g = (np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
           + np.sin(2*np.pi*y)*np.cos(2*np.pi*z)
           + np.sin(2*np.pi*z)*np.cos(2*np.pi*x))
        inside = (np.abs(g) < self.wall_thickness)
        return np.where(inside, self.E_in, self.E_out)

    def cross_E(self, points):
        """
        Three orthogonal 'webs' (slabs) through the center of the unit cell:
          - plane x = 0.5 with thickness `wall_thickness`
          - plane y = 0.5 with thickness `wall_thickness`
          - plane z = 0.5 with thickness `wall_thickness`
    
        `wall_thickness` is interpreted in unit-cell coordinates [0,1].
        Typical useful values: ~0.01–0.15 (much smaller than for gyroid).
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
        # half-thickness in [0, 0.5]
        t = 0.5 * self.wall_thickness
    
        # distance to central planes
        dx = np.abs(x - 0.5)
        dy = np.abs(y - 0.5)
        dz = np.abs(z - 0.5)
    
        # inside solid if within any of the three slabs
        inside = (dx <= t) | (dy <= t) | (dz <= t)
        return np.where(inside, self.E_in, self.E_out)
    
    # Optional extension if you want other unit-cell families later:
    def material_E(self, points):
        if self.geom == "gyroid":
            return self.gyroid_E(points)
        if self.geom == "cross":
            return self.cross_E(points)
        elif self.geom == "solid":
            return self.E_in * np.ones(points.shape[0])
        elif self.geom == "void":
            return self.E_out * np.ones(points.shape[0])
        else:
            return self.gyroid_E(points)  # fallback
            
    def GeomAndQuadrature(self):
        self.unit_cube_quadrature()
        self.E = self.gyroid_E(self.pts)
        
    def filtered_quadrature(self):
        """
        Build tensor grid in NumPy, evaluate geometry in NumPy, slice in NumPy,
        then convert to JAX arrays with fixed shapes.
        """

        inside_np = self.E > 0.5*(self.E_in + self.E_out)    # boolean mask (NumPy)
        self.pts = self.pts[inside_np]
        self.w   = self.w[inside_np]
        self.E   = self.E[inside_np] 
        
        
    def build_basis_and_grads(self,points):
        """Build tensor-product Legendre basis and its physical gradients at points.
        points : (N,3) in [0,1]^3
        orders : (Px,Py,Pz)
        Returns:
          B      : (N, NB)
          dB_dx  : (N, NB)
          dB_dy  : (N, NB)
          dB_dz  : (N, NB)
        where NB=(Px+1)(Py+1)(Pz+1). Chain rule accounts for xhat=2x-1 => d/dx = 2 d/dxhat, etc.
        """
        Px, Py, Pz = self.order
        x, y, z = points[:,0], points[:,1], points[:,2]
        xh, yh, zh = self.map_to_hat(x), self.map_to_hat(y), self.map_to_hat(z)

        Vx = self.legendre_vals(Px, xh)  # (N, Px+1)
        Vy = self.legendre_vals(Py, yh)
        Vz = self.legendre_vals(Pz, zh)

        dVx_hat = self.legendre_derivs_from_vals(Vx, xh)
        dVy_hat = self.legendre_derivs_from_vals(Vy, yh)
        dVz_hat = self.legendre_derivs_from_vals(Vz, zh)

        # physical derivatives: d/dx = 2 * d/dxhat
        dVx = 2.0 * dVx_hat
        dVy = 2.0 * dVy_hat
        dVz = 2.0 * dVz_hat

        # assemble basis lists then stack
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
        self.B      = np.stack(cols, axis=1)
        self.dBdx  = np.stack(dx_cols, axis=1)
        self.dBdy  = np.stack(dy_cols, axis=1)
        self.dBdz  = np.stack(dz_cols, axis=1)
           
    def prepare_jit(self):
         # JIT the core energy; other args are treated as constants at call time
        self.Energy_fn = jax.jit(lambda A: self.neo_hookean_energy_core(A, self.B, self.dBdx, self.dBdy,
                                                                        self.dBdz, self.E, self.nu, self.w, 
                                                                        self.barrier, self.alpha))

        # Gradient w.r.t. coeffs (first argument) — also JIT
        self.dEnergy_dA_fn = jax.jit(
            grad(lambda A:
                 self.neo_hookean_energy_core(A, self.B, self.dBdx, self.dBdy, 
                                              self.dBdz, self.E, self.nu, self.w, 
                                              self.barrier, self.alpha)))
              
        self.Disp = jax.jit(lambda coeffs, B: self.boundary_disp(coeffs, B))
           
if __name__ == "__main__":
    T1 = time.time()
    rve = HyperElasticRVE(order=(8,8,8))
    rve.GeomAndQuadrature()
    rve.filtered_quadrature()
    rve.build_basis_and_grads(rve.pts)
    rve.prepare_jit()
    T2 = time.time()
    
    E   = rve.Energy_fn(rve.coeffs)
    dE  = rve.dEnergy_dA_fn(rve.coeffs)
    print("Setup: ", float(T2-T1),"s. Energy and Sens: ",float(time.time()-T2))
    ubc = rve.boundary_disp(rve.coeffs, rve.B)   # or rve.Disp(rve.coeffs, rve.B)

    print("Energy:", float(E))
    print("‖grad‖:", float(np.linalg.norm(dE)))
    print("u shape:", ubc.shape)
    
    for axis in ("x","y","z"):
        for side in ("lo","hi"):
            pts, (Xf,Yf,Zf) = rve.face_points(axis, side, N=rve.nq+1)
            if axis == "x":
                assert float(np.max(np.abs(pts[:,0] - (0.0 if side=="lo" else 1.0)))) < 1e-12
            if axis == "y":
                assert float(np.max(np.abs(pts[:,1] - (0.0 if side=="lo" else 1.0)))) < 1e-12
            if axis == "z":
                assert float(np.max(np.abs(pts[:,2] - (0.0 if side=="lo" else 1.0)))) < 1e-12
    print("Face coordinate test: OK")
    
    # Evaluate using the face helper (Bf) vs. build B on-the-fly at the same points
    for axis, side in (("z","lo"), ("z","hi")):
        Bf, pts_f, _ = rve.face_basis(axis, side, N=rve.nq+1)
        u1 = Bf @ rve.coeffs.T                         # via face_basis
        # direct: rebuild basis on those points with the internal helper
        u2 = rve.boundary_disp(rve.coeffs, rve._basis_only(pts_f))
        diff = np.linalg.norm(u1 - u2)
        print(axis, side, "||diff|| =", float(diff))
        assert float(diff) < 1e-12
    print("Face basis consistency: OK")
    
    for axis, side in (("z","lo"), ("z","hi")):
        mask = rve.face_solid_mask(axis, side, N=rve.nq+1)
        frac = float(np.mean(mask.astype(np.float64)))
        print(f"{axis}-{side}: solid fraction ~ {frac:.2f}")
    # You’re just checking it’s not identically 0 or 1 unless that’s intended.
    
    rve2 = HyperElasticRVE(order=rve.order, nq=rve.nq)
    rve2.GeomAndQuadrature(); rve2.filtered_quadrature()
    rve2.build_basis_and_grads(rve2.pts); rve2.prepare_jit()
    # Copy coeffs so they’re identical
    rve2.coeffs = rve.coeffs
    
    m0 = rve.face_misfit(rve2, axis_self="z", side_self="hi",
                         axis_other="z", side_other="lo",
                         N=rve.nq+1, solid_only=True)
    print("Misfit identical:", float(m0))
    
    # Perturb rve2 slightly
    rve2.coeffs = rve2.coeffs.at[2,0].add(1e-3)  # tweak uz-constant mode
    m1 = rve.face_misfit(rve2, axis_self="z", side_self="hi",
                         axis_other="z", side_other="lo",
                         N=rve.nq+1, solid_only=True)
    print("Misfit perturbed:", float(m1))
    assert float(m1) > 0.0