from typing import Tuple
import time
import os
import jax
import jax.numpy as np
import numpy as onp
from scipy.optimize import minimize
from RVEAssembly2 import MultiRVEComponent, export_assembled_structured_vts
from RVEAssemblyDirichletBCs import DirichletBC
from active_learning import ActiveLearningModel

def build_component(index_map,catalog,squeeze = -0.4):
    comp = MultiRVEComponent.from_index_map(index_map, catalog)
    comp.initialise_all(filter_quadrature=True)
    comp.build_packing()

    # Faces & masks (coarse sampling is often fine for constraints)
    comp.precompute_faces(N_face=50)
    
    # BCs: z=0 fixed, z=top compressed by -0.2
    def fixed_zero(pts):
        return np.zeros((pts.shape[0], 3))

    def compress_top(pts):
        N = pts.shape[0]
        return np.stack([np.zeros((N,)), squeeze*np.ones((N,)), squeeze*np.ones((N,))], axis=1)

    comp.dirichlet_bcs = []
    comp.add_bc(DirichletBC(axis="z", side="lo", u_fun=fixed_zero))
    comp.add_bc(DirichletBC(axis="z", side="hi", u_fun=compress_top))

    # Batch all constraint data once (fast path)
    comp.build_face_batches()
    return comp

def soft_hinge(x, eps=1e-12):
    # C^1 smooth max(0, x)
#    return 0.5 * (x + np.sqrt(x*x + eps))
    return np.maximum(x, 0.0)

def prep_augL_per_face(comp):
    # JIT once
    print("JIT ENERGY")
    E  = jax.jit(comp.total_energy)       # scalar
    print("JIT Constraints")
    MF = jax.jit(comp.misfits_per_face)   # (Ftot,)

    def L_only(A, lam, rho, tol_vec):
        # A: (ndof,), lam,rho,tol_vec: (Ftot,)
        f  = E(A)
        m  = MF(A)
        # C^1 hinge per face: h = softplus(m - tol) ≈ max(m - tol, 0)
        r  = m - tol_vec
        h  = 0.5*(r + jax.lax.sqrt(r*r + 1e-16))
        return f + np.dot(lam, h) + 0.5*np.dot(rho, h*h)


    print("JIT Value and Grad")
    L_valgrad = jax.jit(jax.value_and_grad(L_only, argnums=0))
    print("JIT Value and Grad Completed")
    return E, MF, L_valgrad


def run_augL_per_face(comp, A0,
                      tol_int=1e-6, tol_bc=1e-6,
                      lam0=0.0, rho0=1e3,
                      eta=0.7, gamma_up=2.0, rho_max=1e3,
                      inner_maxiter=200, inner_ftol=1e-8, inner_gtol=1e-8,
                      verbose=True,
                      inner_log_every=10,          # print/log every k inner steps
                      inner_cb=None               # optional user callback
                      ):

    # Make sure the component exposes these (set during build_face_batches):
    # comp.Fint, comp.Fext, comp.Ftot, comp._sl_int, comp._sl_ext
    assert hasattr(comp, "Fint") and hasattr(comp, "Fext") and hasattr(comp, "Ftot")
    assert hasattr(comp, "_sl_int") and hasattr(comp, "_sl_ext")

    E, MF, L_valgrad = prep_augL_per_face(comp)

    # Tolerances per-face (use MS misfit, so squares of tolerances)
    t_int   = (tol_int**2) * np.ones((comp.Fint,))
    t_ext   = (tol_bc**2)  * np.ones((comp.Fext,))
    tol_vec = np.concatenate([t_int, t_ext], axis=0)     # (Ftot,)

    lam = np.full((comp.Ftot,), lam0)                    # (Ftot,)
    rho = np.full((comp.Ftot,), rho0)                    # (Ftot,)
    A   = np.asarray(A0)

    # Warm-up compile
    _ = L_valgrad(A, lam, rho, tol_vec)

    h_prev = np.full((comp.Ftot,), np.inf)

    for it in range(100):
        cache = {"gA": None, "L": None, "f": None, "m": None}

        def fun(x):
            Ax = np.asarray(x)
            Lval, gA = L_valgrad(Ax, lam, rho, tol_vec)
            cache["gA"] = gA
            cache["L"]  = float(Lval)
            cache["f"]  = float(E(Ax))
            cache["m"]  = np.asarray(MF(Ax))
            return cache["L"]

        def jac(x):
            gA = cache["gA"]
            if gA is None:
                Ax = np.asarray(x)
                _, gA = L_valgrad(Ax, lam, rho, tol_vec)
            cache["gA"] = None
            return onp.asarray(gA)      
        
        inner_k = {"k": 0}
        prev_x  = {"x": None}
        
        def sci_cb(xk):
            # increment counter
            inner_k["k"] += 1
            if inner_k["k"] % inner_log_every != 0:
                return
        
            Ax = np.asarray(xk)
        
            # Try to reuse cache (set in 'fun'); recompute only if needed
            if cache["L"] is None or cache["gA"] is None:
                Lval, gA = L_valgrad(Ax, lam, rho, tol_vec)
                Lval = float(Lval)
            else:
                Lval = cache["L"]
                gA   = cache["gA"]
        
            f_val = cache["f"] if cache["f"] is not None else float(E(Ax))
            m_vec = cache["m"] if cache["m"] is not None else np.asarray(MF(Ax))
        
            # Current violations (same smooth hinge)
            r     = m_vec - tol_vec
            h_vec = 0.5*(r + np.sqrt(r*r + 1e-16))
        
            # Diagnostics
            g_inf  = float(np.max(np.abs(gA)))
            h_max  = float(np.max(h_vec))
            h_mean = float(np.mean(h_vec))
        
            if prev_x["x"] is None:
                dx_inf = 0.0
            else:
                dx_inf = float(np.max(np.abs(Ax - prev_x["x"])))
            prev_x["x"] = Ax
        
            stats = {
                "outer": it,
                "inner": inner_k["k"],
                "L": Lval,
                "f": f_val,
                "h_max": h_max,
                "h_mean": h_mean,
                "g_inf": g_inf,
                "dx_inf": dx_inf,
                "lam_med": float(np.median(lam)),
                "rho_med": float(np.median(rho)),
            }
        
            if inner_cb is not None:
                # hand the stats to user code
                inner_cb(stats)
            elif verbose:
                # default pretty print
                print(f"[inner {it}:{inner_k['k']:04d}] "
                      f"L={stats['L']:.6e} | f={stats['f']:.6e} | "
                      f"h_max={stats['h_max']:.3e} | h_mean={stats['h_mean']:.3e} | "
                      f"‖∇L‖∞={stats['g_inf']:.2e} | ‖Δx‖∞={stats['dx_inf']:.2e} | "
                      f"lam_med={stats['lam_med']:.2e} | rho_med={stats['rho_med']:.2e}")

        x0_host = onp.asarray(A).copy()
        res = minimize(fun=fun, x0=x0_host, jac=jac, method="L-BFGS-B",
                       options={"maxiter": inner_maxiter, "ftol": inner_ftol, "gtol": inner_gtol},
                       callback=sci_cb)
        A = np.asarray(res.x)

        f = cache["f"] if cache["f"] is not None else float(E(A))
        m = cache["m"] if cache["m"] is not None else np.asarray(MF(A))

        r = m - tol_vec
        h = 0.5*((m - tol_vec) + np.maximum(0,m)) # np.sqrt((m - tol_vec)**2 + 1e-16))  # smooth hinge
        
        # Always update multipliers
        lam = lam + rho * h
        
        # Only grow ρ on faces that didn't improve enough
        improved = (h <= eta * h_prev)
        rho = np.where(improved, rho, np.minimum(rho * gamma_up, rho_max))
        h_prev = h
        
        ms = m
        if verbose:
            print(f"[outer {it:03d}] f={f:.6e} | h_max={float(np.max(h)):.3e} "
                  f"| MS_int {np.max(ms[comp._sl_int]):.6e} | MS_ext={np.max(ms[comp._sl_ext]):.6e}"
                  f"| inner_it={res.nit} | lam_med={float(np.median(lam)):.2e} "
                  f"rho_med={float(np.median(rho)):.2e}")

        if float(np.max(ms[comp._sl_int])) <= tol_int**2 and float(np.max(ms[comp._sl_ext])) <= tol_bc**2:
            break

        h_prev = h

    return A

def main():
    index_map = np.array([
        [[0,0,1,1,2],
         [0,1,1,2,2]],
        [[0,0,1,1,2],
         [0,1,1,2,2]]
    ], dtype=int)  # (1,2,5)
    
    catalog = {
        0: dict(order=(6,6,6), nq=50, geom="gyroid", wall_thickness=0.45,
                E_in=10.0, E_out=1e-3, nu=0.30),
        1: dict(order=(6,6,6), nq=50, geom="cross", wall_thickness=0.17,
                E_in=10.0, E_out=1e-3, nu=0.30),
        2: dict(order=(6,6,6), nq=50, geom="gyroid",  wall_thickness=0.35,
                E_in=10.0, E_out=1e-3, nu=0.30),
    }
    
    comp = build_component(index_map,catalog,squeeze = -0.01)
    comp.precompute_faces(N_face=50)
    comp.build_face_batches()
    A0 = comp.pack_coeffs()
    for i in range(100):
        squeeze = -0.01*(i+1)
        comp = build_component(index_map,catalog,squeeze = squeeze)
        comp.precompute_faces(N_face=50)
        comp.build_face_batches()
        A0 = comp.pack_coeffs()
        
        A_opt = run_augL_per_face(
            comp, A0,
            tol_int=1e-4, tol_bc=1e-4,
            lam0=0.0, rho0=1e3,
            eta=0.5, gamma_up=10.0,
            inner_maxiter=100,
        )
    
        export_assembled_structured_vts(
            comp, A_opt,
            n_vis_per_rve=70, stacked_coords=True,
            add_solid_mask=True, out_dir="/Users/rhewson/PythonLocal/SED/data/assembled", stem=f"final_z={squeeze:.3f}",
            strain="green",
        )
        A0 = A_opt

if __name__ == "__main__":
    main()