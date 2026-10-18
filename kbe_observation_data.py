"""KBE observation-data layer.

Single entry point:

    dataset = generate_kbe_dataset(system, n_samples=..., seed=..., observables=None, **overrides)

Supported systems: "ou", "triple_well", "lorenz63", "lorenz96", and "sst".

For the simulated systems this module builds the observation data from the
dynamics (KBE semigroup or hidden-trajectory sensor readings); "sst" loads
the ERA5 daily sea-surface-temperature record and applies the same
weather-forecast protocol to the real observations.

Returned dataset dict keys (consumed by edmd_bso_method.run_bso_pipeline):
    system       : system name
    kind         : "grid1d" | "grid2d" | "cloud1d" | "sensor1d"
    T            : one-step horizon tau
    u0, uT, u2T  : (n_samples, ...) float64 observation arrays
    fno_grid     : positional channel for the neural operator
                   ((p,) in [0,1] for 1d kinds, (H, W, 2) for grid2d)
    basis_coords : (p, d) coordinates fed to the dictionary network
    weights      : (p,) quadrature weights, normalized to sum 1
    meta         : system parameters, nodes / generator / solver handles
"""

from __future__ import annotations

import os
import pickle
import hashlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial.hermite_e import hermeval
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial import cKDTree


def calendar_day_index(times):
    """Map dates to a leap-calendar month/day index in ``[1, 366]``.

    Unlike ``dayofyear``, March 1 and later retain the same index in leap and
    non-leap years.  Index 60 is reserved for February 29.
    """
    import pandas as pd

    dates = pd.DatetimeIndex(pd.to_datetime(times))
    days_before_month = np.asarray(
        [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
        dtype=np.int16,
    )
    return (
        days_before_month[dates.month.to_numpy() - 1]
        + dates.day.to_numpy()
    ).astype(np.int16)


def calendar_phase(times):
    """Return a leap-stable annual phase in ``[0, 1)`` for each date.

    Equal month/day pairs have equal phases in every year.  February 29 is
    placed halfway between February 28 and March 1.
    """
    import pandas as pd

    dates = pd.DatetimeIndex(pd.to_datetime(times))
    days_before_month = np.asarray(
        [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
        dtype=np.float64,
    )
    phase_day = (
        days_before_month[dates.month.to_numpy() - 1]
        + dates.day.to_numpy().astype(np.float64)
        - 1.0
    )
    february_29 = (
        (dates.month.to_numpy() == 2)
        & (dates.day.to_numpy() == 29)
    )
    phase_day[february_29] = 58.5
    return phase_day / 365.0


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck (1D SDE on a grid)
# ---------------------------------------------------------------------------

OU_PARAMS = dict(
    name="Ornstein-Uhlenbeck",
    D=0.5,
    x_range=(-5.0, 5.0),
    Nx=128,
    T=0.15,
    Nt_solve=200,
)


def ou_drift(x: np.ndarray) -> np.ndarray:
    return -1.0 * x


class KBESolver1D:
    """Finite-difference backward generator + Crank-Nicolson time stepping."""

    def __init__(self, f, D, x_min, x_max, Nx):
        self.f = f
        self.D = D
        self.Nx = Nx
        self.x = np.linspace(x_min, x_max, Nx)
        self.dx = self.x[1] - self.x[0]
        K = np.zeros((Nx, Nx), dtype=float)
        f_vals = f(self.x)
        for i in range(1, Nx - 1):
            drift = f_vals[i] / (2.0 * self.dx)
            diff = D / self.dx**2
            K[i, i] = -2.0 * diff
            K[i, i + 1] = diff + drift
            K[i, i - 1] = diff - drift
        self.K = K

    def solve(self, u0, T, Nt):
        dt = T / Nt
        u = u0.astype(float).copy()
        if not hasattr(self, "_lu_cache"):
            self._lu_cache = {}
        if dt not in self._lu_cache:
            I = np.eye(self.Nx)
            A = I - 0.5 * dt * self.K
            B = I + 0.5 * dt * self.K
            self._lu_cache[dt] = (lu_factor(A), B)
        lu_piv, B = self._lu_cache[dt]
        for _ in range(Nt):
            u = lu_solve(lu_piv, B @ u)
        return u


def random_observable_1d(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_modes = int(rng.integers(2, 5))
    L = float(x[-1] - x[0])
    func = np.zeros_like(x)
    for _ in range(n_modes):
        if rng.random() > 0.5:
            k = int(rng.integers(1, 5))
            amp = rng.uniform(-1.0, 1.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            func += amp * np.sin(2.0 * np.pi * k * (x - x[0]) / L + phase)
        else:
            center = rng.uniform(x.min() * 0.6, x.max() * 0.6)
            width = rng.uniform(0.3, 1.0)
            amp = rng.uniform(-1.5, 1.5)
            func += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
    env = np.exp(-0.1 * ((x - x.mean()) / (0.5 * L)) ** 2)
    return func * env


def _generate_ou(n_samples, seed, observables, overrides):
    params = dict(OU_PARAMS)
    params.update(overrides)
    rng = np.random.default_rng(seed)
    solver = KBESolver1D(ou_drift, params["D"], params["x_range"][0], params["x_range"][1], params["Nx"])
    x = solver.x
    sigma_s = np.sqrt(params["D"])
    L = float(x[-1] - x[0])
    env = np.exp(-0.1 * ((x - x.mean()) / (0.5 * L)) ** 2)

    trap = np.full(params["Nx"], solver.dx)
    trap[0] *= 0.5
    trap[-1] *= 0.5
    rho = np.exp(-x**2 / (2.0 * params["D"]))
    w = trap * rho
    w = w / w.sum()

    u0_list = []
    for idx in range(n_samples):
        if observables is not None:
            u0 = np.asarray(observables[idx % len(observables)](x), dtype=np.float64)
        elif idx % 10 < 3:
            # enveloped Hermite modes up to degree 17, unit norm in L2(rho)
            order = int(rng.integers(0, 18))
            coeff = np.zeros(order + 1)
            coeff[-1] = 1.0
            u0 = hermeval(x / sigma_s, coeff) * env
            u0 = u0 / (np.sqrt(np.sum(w * u0**2)) + 1e-300)
        else:
            u0 = random_observable_1d(x, rng)
        u0_list.append(u0)

    T, Nt = params["T"], params["Nt_solve"]
    print(f"[ou] generating {n_samples} KBE samples on {params['Nx']} grid points ...")
    u0_arr = np.asarray(u0_list)
    uT_arr = np.stack([solver.solve(u, T, Nt) for u in u0_arr])
    u2T_arr = np.stack([solver.solve(u, 2 * T, 2 * Nt) for u in u0_arr])

    # invariant-measure weighting: the natural L2(rho) geometry for the OU
    # BSO operator (self-adjoint there; Hermite eigenfunctions orthonormal)
    coords01 = (x - x[0]) / L
    return dict(
        system="ou",
        kind="grid1d",
        T=float(T),
        u0=u0_arr,
        uT=uT_arr,
        u2T=u2T_arr,
        fno_grid=coords01,
        basis_coords=(x / np.abs(x).max())[:, None],
        weights=w,
        meta=dict(params=params, x=x, solver=solver, envelope=env,
                  analytic_lambdas=-np.arange(0, 12, dtype=float)),
    )


# ---------------------------------------------------------------------------
# Triple well (2D SDE on a grid)
# ---------------------------------------------------------------------------

TRIPLE_WELL_PARAMS = dict(
    name="Triple Well",
    D=1.09**2 / 2,
    bounds=(-2.0, 2.0, -1.0, 2.0),
    N=64,
    T=0.30,
)


def tw_potential(x, y):
    """Three-hole potential: wells near (-1, 0), (1, 0) and (0, 5/3)."""
    return (3.0 * np.exp(-x**2 - (y - 1.0 / 3.0) ** 2)
            - 3.0 * np.exp(-x**2 - (y - 5.0 / 3.0) ** 2)
            - 5.0 * np.exp(-(x - 1.0) ** 2 - y**2)
            - 5.0 * np.exp(-(x + 1.0) ** 2 - y**2)
            + 0.2 * x**4 + 0.2 * (y - 1.0 / 3.0) ** 4)


def tw_drift_x(x, y):
    dVdx = (
        -6 * x * np.exp(-x**2 - (y - 1 / 3) ** 2)
        + 6 * x * np.exp(-x**2 - (y - 5 / 3) ** 2)
        + 10 * (x - 1) * np.exp(-(x - 1) ** 2 - y**2)
        + 10 * (x + 1) * np.exp(-(x + 1) ** 2 - y**2)
        + 0.8 * x**3
    )
    return -dVdx


def tw_drift_y(x, y):
    dVdy = (
        -6 * (y - 1 / 3) * np.exp(-x**2 - (y - 1 / 3) ** 2)
        + 6 * (y - 5 / 3) * np.exp(-x**2 - (y - 5 / 3) ** 2)
        + 10 * y * np.exp(-(x - 1) ** 2 - y**2)
        + 10 * y * np.exp(-(x + 1) ** 2 - y**2)
        + 0.8 * (y - 1 / 3) ** 3
    )
    return -dVdy


class KBESolver2D:
    """Sparse upwind backward generator with reflecting boundaries; expm stepping."""

    def __init__(self, fx, fy, D, bounds, N):
        self.D = D
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.N = N
        self.dx = (self.xmax - self.xmin) / (N - 1)
        self.dy = (self.ymax - self.ymin) / (N - 1)
        self.x = np.linspace(self.xmin, self.xmax, N)
        self.y = np.linspace(self.ymin, self.ymax, N)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.Fx = fx(self.X, self.Y)
        self.Fy = fy(self.X, self.Y)
        self.operator = self._build_operator().tocsr()

    def _index(self, i, j):
        return i * self.N + j

    def _build_operator(self):
        rows, cols, data = [], [], []

        def add(row, col, value):
            if value != 0.0:
                rows.append(row)
                cols.append(col)
                data.append(float(value))

        dx2, dy2 = self.dx**2, self.dy**2
        for i in range(self.N):
            for j in range(self.N):
                row = self._index(i, j)
                bx = float(self.Fx[i, j])
                by = float(self.Fy[i, j])
                if i == 0:
                    r = 2.0 * self.D / dx2 + max(bx, 0.0) / self.dx
                    add(row, row, -r)
                    add(row, self._index(1, j), r)
                elif i == self.N - 1:
                    r = 2.0 * self.D / dx2 + max(-bx, 0.0) / self.dx
                    add(row, row, -r)
                    add(row, self._index(self.N - 2, j), r)
                else:
                    rl = self.D / dx2 + max(-bx, 0.0) / self.dx
                    rr = self.D / dx2 + max(bx, 0.0) / self.dx
                    add(row, self._index(i - 1, j), rl)
                    add(row, row, -(rl + rr))
                    add(row, self._index(i + 1, j), rr)
                if j == 0:
                    r = 2.0 * self.D / dy2 + max(by, 0.0) / self.dy
                    add(row, row, -r)
                    add(row, self._index(i, 1), r)
                elif j == self.N - 1:
                    r = 2.0 * self.D / dy2 + max(-by, 0.0) / self.dy
                    add(row, row, -r)
                    add(row, self._index(i, self.N - 2), r)
                else:
                    rd = self.D / dy2 + max(-by, 0.0) / self.dy
                    ru = self.D / dy2 + max(by, 0.0) / self.dy
                    add(row, self._index(i, j - 1), rd)
                    add(row, row, -(rd + ru))
                    add(row, self._index(i, j + 1), ru)
        shape = (self.N * self.N, self.N * self.N)
        return sp.coo_matrix((data, (rows, cols)), shape=shape)

    def solve(self, u0, T):
        u_vec = np.asarray(u0, dtype=float).reshape(-1)
        evolved = spla.expm_multiply(self.operator * float(T), u_vec)
        return np.asarray(evolved, dtype=float).reshape(self.N, self.N)


def compute_triple_well_fd_reference_lambdas(n_eigs=6, N_ref=128, tol=1e-9, maxiter=20000):
    """Finite-difference numerical reference for the Triple-Well generator spectrum."""
    cfg = dict(TRIPLE_WELL_PARAMS)
    solver = KBESolver2D(tw_drift_x, tw_drift_y, cfg["D"], cfg["bounds"], int(N_ref))
    vals = spla.eigs(solver.operator, k=int(n_eigs), which="LR",
                     return_eigenvectors=False, tol=float(tol), maxiter=int(maxiter))
    vals = np.asarray(vals)
    return vals[np.argsort(-vals.real)]


def random_observable_2d(X, Y, rng: np.random.Generator) -> np.ndarray:
    n_modes = int(rng.integers(2, 5))
    func = np.zeros_like(X)
    for _ in range(n_modes):
        if rng.random() > 0.5:
            kx = int(rng.integers(1, 3))
            ky = int(rng.integers(1, 3))
            px, py = rng.uniform(0.0, 2.0 * np.pi, 2)
            amp = rng.uniform(-1.0, 1.0)
            func += amp * np.sin(kx * X + px) * np.cos(ky * Y + py)
        else:
            cx = rng.uniform(X.min() * 0.6, X.max() * 0.6)
            cy = rng.uniform(Y.min() * 0.6, Y.max() * 0.6)
            sx, sy = rng.uniform(0.4, 1.0, 2)
            amp = rng.uniform(-1.5, 1.5)
            func += amp * np.exp(-0.5 * ((X - cx) / sx) ** 2 - 0.5 * ((Y - cy) / sy) ** 2)
    env = np.exp(-0.1 * X**2 - 0.1 * Y**2)
    return func * env


def _generate_triple_well(n_samples, seed, observables, overrides):
    params = dict(TRIPLE_WELL_PARAMS)
    params.update(overrides)
    rng = np.random.default_rng(seed)
    solver = KBESolver2D(tw_drift_x, tw_drift_y, params["D"], params["bounds"], params["N"])
    T = params["T"]

    print(f"[triple_well] generating {n_samples} KBE samples on {params['N']}x{params['N']} grid ...")
    u0_list = []
    for idx in range(n_samples):
        if observables is not None:
            u0 = np.asarray(observables[idx % len(observables)](solver.X, solver.Y), dtype=np.float64)
        else:
            u0 = random_observable_2d(solver.X, solver.Y, rng)
        u0_list.append(u0)
    u0_arr = np.asarray(u0_list)
    uT_arr = np.stack([solver.solve(u, T) for u in u0_arr])
    u2T_arr = np.stack([solver.solve(u, 2.0 * T) for u in u0_arr])

    N = params["N"]
    gx = (solver.X - solver.xmin) / (solver.xmax - solver.xmin)
    gy = (solver.Y - solver.ymin) / (solver.ymax - solver.ymin)
    fno_grid = np.stack([gx, gy], axis=-1)
    basis_coords = np.stack([solver.X.reshape(-1), solver.Y.reshape(-1)], axis=-1)
    basis_coords = basis_coords / np.abs(basis_coords).max(axis=0, keepdims=True)
    # Uniform quadrature weights in the L2(dx) geometry.
    V = tw_potential(solver.X, solver.Y)
    w = np.ones(N * N)
    return dict(
        system="triple_well",
        kind="grid2d",
        T=float(T),
        u0=u0_arr,
        uT=uT_arr,
        u2T=u2T_arr,
        fno_grid=fno_grid,
        basis_coords=basis_coords,
        weights=w / w.sum(),
        meta=dict(params=params, X=solver.X, Y=solver.Y, V=V, solver=solver),
    )


# ---------------------------------------------------------------------------
# Shared point-cloud KBE generator (deterministic drift, k-NN least squares)
# ---------------------------------------------------------------------------

def build_cloud_kbe_operator(nodes, rhs_vals, k=48, reg_scale=1.0e-6, diffusion=0.0):
    """Collocation matrix L_h with (L_h u)_i ~ f(x_i) . grad u(x_i); zero row sums.

    With diffusion > 0, an SDE noise term is added to the backward generator:
    L = advection + diffusion * (A - I), where A is the row-normalized kNN
    kernel (a proper CTMC generator: zero row sums, non-negative off-diagonal,
    hence max Re(lambda) <= 0 for the diffusion part). This regularizes the
    spurious expanding modes of the pure-advection collocation.
    """
    pts = np.asarray(nodes, dtype=np.float64)
    rhs = np.asarray(rhs_vals, dtype=np.float64)
    dim = pts.shape[1]
    tree = cKDTree(pts)
    _, nbr_idx = tree.query(pts, k=min(int(k) + 1, len(pts)))
    rows, cols, data = [], [], []
    d_rows, d_cols, d_data = [], [], []
    eye = np.eye(dim, dtype=np.float64)
    for i in range(len(pts)):
        nbrs = np.asarray(nbr_idx[i, 1:], dtype=int)
        offsets = pts[nbrs] - pts[i]
        dist2 = np.sum(offsets * offsets, axis=1)
        width = np.median(dist2) + 1.0e-12
        w = np.exp(-dist2 / width)
        weighted_offsets = w[:, None] * offsets
        normal = offsets.T @ weighted_offsets
        reg = reg_scale * (np.trace(normal) / dim + 1.0e-12)
        coeff = weighted_offsets @ np.linalg.solve(normal + reg * eye, rhs[i])
        rows.extend([i] * (len(nbrs) + 1))
        cols.extend(nbrs.tolist() + [i])
        data.extend(coeff.tolist() + [-float(np.sum(coeff))])
        if diffusion > 0:
            a = w / (np.sum(w) + 1.0e-300)
            d_rows.extend([i] * (len(nbrs) + 1))
            d_cols.extend(nbrs.tolist() + [i])
            d_data.extend((diffusion * a).tolist() + [-float(diffusion)])
    L = sp.csr_matrix((data, (rows, cols)), shape=(len(pts), len(pts)), dtype=np.float64)
    if diffusion > 0:
        L = L + sp.csr_matrix((d_data, (d_rows, d_cols)),
                              shape=(len(pts), len(pts)), dtype=np.float64)
    return L


def solve_kbe_observables(u0_batch, horizon, operator):
    vals = np.asarray(u0_batch, dtype=np.float64)
    evolved = spla.expm_multiply(float(horizon) * operator, vals.T).T
    return np.asarray(evolved, dtype=np.float64)


def _pca_first_component_order(points):
    pts = np.asarray(points, dtype=np.float64)
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    score = centered @ vt[0]
    return np.argsort(score)


# ---------------------------------------------------------------------------
# Lorenz63 (3D chaotic attractor, sensor-curve observables on a site ring)
# ---------------------------------------------------------------------------

LORENZ63_PARAMS = dict(
    name="Lorenz63",
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    K=20,                        # sensor sites on the ring s in [0, 1)
    dt=0.005,
    tau=0.1,
    burn_steps=4000,             # dt steps discarded before recording
    long_traj_steps=300000,      # dt steps; tau-sampled length = this // (tau/dt)
    train_fraction=0.8,          # leading trajectory fraction used for training
    n_train_times=256,           # training-pair start times per sensor family
)


def lorenz63_rhs(points, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    pts = np.asarray(points, dtype=np.float64)
    squeeze = pts.ndim == 1
    if squeeze:
        pts = pts[None, :]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rhs = np.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=-1)
    return rhs[0] if squeeze else rhs


def _rk4(points, rhs_func, dt):
    k1 = rhs_func(points)
    k2 = rhs_func(points + 0.5 * dt * k1)
    k3 = rhs_func(points + 0.5 * dt * k2)
    k4 = rhs_func(points + dt * k3)
    return points + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _l63_scaled(states):
    """Standardize attractor coordinates to O(1): x/8, y/9, (z-25)/8."""
    pts = np.asarray(states, dtype=np.float64)
    return pts[:, 0] / 8.0, pts[:, 1] / 9.0, (pts[:, 2] - 25.0) / 8.0


def l63_observation_bank(states, site_param):
    """Four deterministic sensor-curve families; each value is (K, n_states).

    Every sensor site s carries a fixed measurement function of the hidden
    state (x, y, z); the site dependence is smooth and periodic in s, so the
    readings form smooth closed curves over the site ring.
    """
    xs, ys, zs = _l63_scaled(states)
    th = 2.0 * np.pi * np.asarray(site_param, dtype=np.float64)
    c1, s1 = np.cos(th), np.sin(th)
    ring = xs[:, None] * c1 + ys[:, None] * s1 + 0.5 * zs[:, None] * np.sin(2 * th)
    quad = (xs * ys)[:, None] * c1 + (zs**2 - 1.0)[:, None] * s1 \
        + 0.4 * (xs * zs)[:, None] * np.cos(2 * th + 0.7)
    nonlin = np.tanh(xs[:, None] * c1 + ys[:, None] * s1 + 0.3 * zs[:, None]) \
        + 0.25 * zs[:, None] * np.cos(2 * th + 1.0) \
        + 0.15 * (xs * ys)[:, None] * np.sin(th + 0.5)
    pts = np.asarray(states, dtype=np.float64)
    centers = np.stack([12.0 * c1, 12.0 * s1, 25.0 + 6.0 * np.sin(2 * th)], axis=1)
    d2 = ((pts[:, None, 0] - centers[None, :, 0]) / 9.0) ** 2 \
        + ((pts[:, None, 1] - centers[None, :, 1]) / 10.0) ** 2 \
        + ((pts[:, None, 2] - centers[None, :, 2]) / 9.0) ** 2
    bump = np.exp(-0.5 * d2)
    return {
        "nonlinear_sensor": nonlin.T.copy(),
        "ring_projection": ring.T.copy(),
        "quadratic_moment": quad.T.copy(),
        "radial_bump": bump.T.copy(),
    }


class RandomSensorFamilyL63:
    """Random site-smooth combinations of low-order state features."""

    def __init__(self, seed, site_param):
        self.rng = np.random.default_rng(seed)
        self.site_param = np.asarray(site_param, dtype=np.float64)

    def _envelope(self):
        rng = self.rng
        th = 2.0 * np.pi * self.site_param
        env = np.full_like(self.site_param, rng.uniform(-0.3, 0.3))
        for harm in (1, 2, 3):
            env += rng.normal(scale=0.55 / harm) * np.cos(harm * th + rng.uniform(0, 2 * np.pi))
        return env

    def sample_field(self, states):
        rng = self.rng
        xs, ys, zs = _l63_scaled(states)
        feats = [xs, ys, zs, xs * ys, xs * zs, ys * zs,
                 xs**2 - 1.0, zs**2 - 1.0,
                 np.tanh(rng.normal(scale=0.8) * xs + rng.normal(scale=0.8) * ys
                         + rng.normal(scale=0.8) * zs)]
        picks = rng.choice(len(feats), size=int(rng.integers(3, 6)), replace=False)
        out = np.zeros((len(xs), len(self.site_param)))
        for j in picks:
            out += rng.uniform(0.4, 1.0) * feats[j][:, None] * self._envelope()[None, :]
        return out.T.copy()


def _generate_lorenz63(n_samples, seed, observables, overrides):
    """Weather-forecast protocol on one long hidden Lorenz-63 trajectory.

    Mirrors the Lorenz-96 protocol: sensor readings are recorded along a
    single trajectory; training pairs are tau-shifted readings taken ONLY
    from the leading `train_fraction` (the 'past'); the trailing segment is
    held out as the forecast test window (meta["test_readings"]). The
    tau-sampled hidden trajectory itself is stored in meta["traj_tau"] for
    spectral diagnostics.
    """
    params = dict(LORENZ63_PARAMS)
    params.update(overrides)
    K = int(params["K"])
    dt, tau = float(params["dt"]), float(params["tau"])
    m = int(round(tau / dt))
    rhs = lambda p: lorenz63_rhs(p, params["sigma"], params["rho"], params["beta"])
    site_param = np.linspace(0.0, 1.0, K, endpoint=False)
    rng = np.random.default_rng(seed)

    print(f"[lorenz63] simulating hidden trajectory ...")
    x = np.array([1.0, 1.0, 1.0]) + 0.01 * rng.normal(size=3)
    for _ in range(int(params["burn_steps"])):
        x = _rk4(x, rhs, dt)
    n_tau = int(params["long_traj_steps"]) // m
    states = np.zeros((n_tau, 3))
    for i in range(n_tau):
        for _ in range(m):
            x = _rk4(x, rhs, dt)
        states[i] = x

    field_banks = dict(l63_observation_bank(states, site_param))
    if observables is not None:
        for j, fn in enumerate(observables):
            field_banks[f"custom_{j}"] = np.asarray(fn(states), dtype=np.float64)
    else:
        family = RandomSensorFamilyL63(seed, site_param)
        for j in range(int(n_samples)):
            field_banks[f"random_{j}"] = family.sample_field(states)

    train_end = int(float(params["train_fraction"]) * n_tau)
    idx = np.linspace(0, train_end - 3, int(params["n_train_times"]), dtype=int)
    print(f"[lorenz63] {len(field_banks)} sensor families; training pairs from "
          f"tau-steps [0, {train_end}) of {n_tau}; forecast test window = "
          f"{n_tau - train_end} tau-steps")
    y0_list, y1_list, y2_list = [], [], []
    test_readings = {}
    train_readings = {}
    norms = {}
    for name, field in field_banks.items():
        field = np.asarray(field, dtype=np.float64)            # (K, n_tau)
        # per-family scalar normalization from TRAINING-window statistics only
        shift = float(field[:, :train_end].mean())
        scale = float(field[:, :train_end].std()) + 1e-6
        fieldN = (field - shift) / scale
        y0_list.append(fieldN[:, idx].T)
        y1_list.append(fieldN[:, idx + 1].T)
        y2_list.append(fieldN[:, idx + 2].T)
        test_readings[name] = fieldN[:, train_end:]
        train_readings[name] = fieldN[:, :train_end].astype(np.float32)
        norms[name] = (shift, scale)

    return dict(
        system="lorenz63",
        kind="sensor1d",
        T=float(tau),
        u0=np.concatenate(y0_list, axis=0),
        uT=np.concatenate(y1_list, axis=0),
        u2T=np.concatenate(y2_list, axis=0),
        fno_grid=site_param,
        basis_coords=site_param[:, None],
        weights=np.full(K, 1.0 / K),
        meta=dict(params=params, site_param=site_param, train_end=int(train_end),
                  n_tau=int(n_tau), test_readings=test_readings,
                  train_readings=train_readings, norms=norms,
                  traj_tau=states, seed=int(seed)),
    )


# ---------------------------------------------------------------------------
# Lorenz96 (20D hidden-state cloud, spatial sensor-curve observables)
# ---------------------------------------------------------------------------

LORENZ96_PARAMS = dict(
    name="Lorenz96",
    K=20,
    F=8.0,
    dt=0.01,
    tau=0.02,
    burn_steps=4000,
    long_traj_steps=30000,       # dt steps; tau-sampled length = this // (tau/dt)
    train_fraction=0.8,          # leading trajectory fraction used for training
    n_train_times=256,           # training-pair start times per sensor family
)


def lorenz96_rhs(states, forcing=8.0):
    arr = np.asarray(states, dtype=np.float64)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]
    rhs = (np.roll(arr, -1, axis=1) - np.roll(arr, 2, axis=1)) * np.roll(arr, 1, axis=1) - arr + forcing
    return rhs[0] if squeeze else rhs


def periodic_distance(grid, center):
    raw = np.abs(grid - center)
    return np.minimum(raw, 1.0 - raw)


def spatial_observation_bank(states, forcing=8.0):
    """Four deterministic sensor-curve families; each value is (K, n_states)."""
    x = np.asarray(states, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    xm2 = np.roll(x, 2, axis=1)
    xm1 = np.roll(x, 1, axis=1)
    xp1 = np.roll(x, -1, axis=1)
    xp2 = np.roll(x, -2, axis=1)
    adv = (xp1 - xm2) * xm1
    grad = 0.5 * (xp1 - xm1)
    local_energy = x**2 + 0.25 * (xm1**2 + xp1**2)
    local_flux = x * xp1 - xm1 * xp2
    tendency = adv - x + forcing
    nonlinear_sensor = np.tanh(0.35 * x + 0.08 * (xm1 + xp1)) + 0.04 * x * xp1
    return {
        "nonlinear_sensor": nonlinear_sensor.T.copy(),
        "local_energy": local_energy.T.copy(),
        "advective_tendency": tendency.T.copy(),
        "flux_gradient": (local_flux + 0.25 * grad).T.copy(),
    }


class RandomSpatialObservationFamilyL96:
    """Random stencil / flux-energy / spectral sensor-curve fields."""

    def __init__(self, seed, site_param):
        self.rng = np.random.default_rng(seed)
        self.site_param = np.asarray(site_param, dtype=np.float64)

    def _envelope(self):
        rng = self.rng
        center = rng.uniform(0.0, 1.0)
        width = rng.uniform(0.08, 0.25)
        background = rng.uniform(0.25, 0.55)
        cos_amp = rng.uniform(-0.20, 0.20)
        cos_harm = int(rng.integers(1, 6))
        cos_phase = rng.uniform(0.0, 2.0 * np.pi)
        dist = periodic_distance(self.site_param, center)
        local = np.exp(-0.5 * (dist / max(width, 1.0e-3)) ** 2)
        shaped = background + (1.0 - background) * local
        shaped += cos_amp * np.cos(2.0 * np.pi * cos_harm * self.site_param + cos_phase)
        return shaped

    def sample_field(self, states):
        rng = self.rng
        x = np.asarray(states, dtype=np.float64)
        xm2 = np.roll(x, 2, axis=1)
        xm1 = np.roll(x, 1, axis=1)
        xp1 = np.roll(x, -1, axis=1)
        xp2 = np.roll(x, -2, axis=1)
        grad = 0.5 * (xp1 - xm1)
        curv = xp1 - 2.0 * x + xm1
        adv = (xp1 - xm2) * xm1
        pair = x * xp1 - xm1 * xp2
        local_energy = x**2 + 0.35 * (xm1**2 + xp1**2)
        env = self._envelope()[None, :]

        branch = str(rng.choice(["stencil_field", "flux_energy_field", "spectral_field"],
                                p=[0.45, 0.35, 0.20]))
        if branch == "stencil_field":
            w = rng.normal(loc=[1.0, 0.45, 0.45, -0.15, -0.15], scale=[0.22, 0.18, 0.18, 0.10, 0.10])
            core = w[0] * x + w[1] * xm1 + w[2] * xp1 + w[3] * xm2 + w[4] * xp2
            val = core + rng.normal(scale=0.22) * grad + rng.normal(scale=0.16) * curv
            val += rng.normal(scale=0.12) * np.tanh(rng.uniform(0.20, 0.80) * core)
            out = env * val
        elif branch == "flux_energy_field":
            val = rng.uniform(0.45, 1.05) * rng.choice([-1.0, 1.0]) * adv
            val += rng.uniform(0.12, 0.34) * local_energy
            val += rng.normal(scale=0.18) * pair
            val += rng.normal(scale=0.16) * grad
            val += rng.normal(scale=0.14) * x
            out = env * val
        else:
            n_modes = int(rng.integers(2, 5))
            spectral_weight = np.zeros_like(self.site_param)
            for _ in range(n_modes):
                mode = int(rng.integers(1, 7))
                angle = 2.0 * np.pi * mode * self.site_param + rng.uniform(0.0, 2.0 * np.pi)
                spectral_weight += rng.normal(scale=0.22) * np.cos(angle) + rng.normal(scale=0.22) * np.sin(angle)
            out = spectral_weight[None, :] * x
            out += rng.normal(scale=0.18) * env * x**2
            out += rng.normal(scale=0.14) * env * x * xp1
            out += rng.normal(scale=0.08) * env * np.tanh(0.25 * x) ** 3
        out = out + 0.04 * (env - env.mean(axis=1, keepdims=True)) * grad
        return out.T.copy()


def lorenz96_diagnostic_field(states, site_param):
    x = np.asarray(states, dtype=np.float64)
    xp1 = np.roll(x, -1, axis=1)
    env = 0.55 + 0.30 * np.cos(2.0 * np.pi * site_param - 0.35) + 0.10 * np.sin(4.0 * np.pi * site_param + 0.25)
    return (env[None, :] * (x + 0.12 * x * xp1 - 0.05 * x**2)).T.copy()


def _generate_lorenz96(n_samples, seed, observables, overrides):
    """Weather-forecast protocol on one long hidden trajectory.

    Sensor readings are recorded along a single trajectory. Training pairs
    are tau-shifted readings taken ONLY from the leading `train_fraction` of
    the trajectory (the 'past'); the trailing segment is held out as the
    forecast test window and stored in meta["test_readings"].
    n_samples = number of random sensor-curve families (added to the 4
    deterministic ones).
    """
    params = dict(LORENZ96_PARAMS)
    params.update(overrides)
    K, F = params["K"], params["F"]
    dt, tau = float(params["dt"]), float(params["tau"])
    m = int(round(tau / dt))
    rhs = lambda s: lorenz96_rhs(s, F)
    site_param = np.linspace(0.0, 1.0, K, endpoint=False)
    rng = np.random.default_rng(seed)

    print(f"[lorenz96] simulating hidden trajectory (K={K}) ...")
    x = F * np.ones(K) + 0.05 * rng.normal(size=K)
    for _ in range(int(params["burn_steps"])):
        x = _rk4(x, rhs, dt)
    n_tau = int(params["long_traj_steps"]) // m
    states = np.zeros((n_tau, K))
    for i in range(n_tau):
        for _ in range(m):
            x = _rk4(x, rhs, dt)
        states[i] = x

    field_banks = dict(spatial_observation_bank(states, forcing=F))
    if observables is not None:
        for j, fn in enumerate(observables):
            field_banks[f"custom_{j}"] = np.asarray(fn(states), dtype=np.float64)
    else:
        family = RandomSpatialObservationFamilyL96(seed, site_param)
        for j in range(int(n_samples)):
            field_banks[f"random_{j}"] = family.sample_field(states)

    train_end = int(float(params["train_fraction"]) * n_tau)
    idx = np.linspace(0, train_end - 3, int(params["n_train_times"]), dtype=int)
    print(f"[lorenz96] {len(field_banks)} sensor families; training pairs from "
          f"tau-steps [0, {train_end}) of {n_tau}; forecast test window = "
          f"{n_tau - train_end} tau-steps")
    y0_list, y1_list, y2_list = [], [], []
    test_readings = {}
    train_readings = {}
    norms = {}
    for name, field in field_banks.items():
        field = np.asarray(field, dtype=np.float64)            # (K, n_tau)
        # per-family scalar normalization from TRAINING-window statistics only
        shift = float(field[:, :train_end].mean())
        scale = float(field[:, :train_end].std()) + 1e-6
        fieldN = (field - shift) / scale
        y0_list.append(fieldN[:, idx].T)
        y1_list.append(fieldN[:, idx + 1].T)
        y2_list.append(fieldN[:, idx + 2].T)
        test_readings[name] = fieldN[:, train_end:]
        train_readings[name] = fieldN[:, :train_end].astype(np.float32)
        norms[name] = (shift, scale)

    return dict(
        system="lorenz96",
        kind="sensor1d",
        T=float(tau),
        u0=np.concatenate(y0_list, axis=0),
        uT=np.concatenate(y1_list, axis=0),
        u2T=np.concatenate(y2_list, axis=0),
        fno_grid=site_param,
        basis_coords=site_param[:, None],
        weights=np.full(K, 1.0 / K),
        meta=dict(params=params, site_param=site_param, train_end=int(train_end),
                  n_tau=int(n_tau), test_readings=test_readings,
                  train_readings=train_readings, norms=norms),
    )


# ---------------------------------------------------------------------------
# SST: multivariate ocean/air state, SST-only verification
# ---------------------------------------------------------------------------

SST_PARAMS = dict(
    name="ERA5-SST + ocean/air state",
    data_path="data/sst_multivar_64x32_daily_1993_2022.zarr",
    tau=1.0,
    record_start="1993-01-01",
    record_end="2022-12-31",
    preprocessing_version="sst-data-v6-1993-2022",
    train_end_date="2020-01-01",  # exclusive; the evaluation period begins in 2020
    n_train_times=9859,           # all complete (t,t+1,t+2) starts
    channel_names=(
        "sst",
        "wind_stress_x",
        "wind_stress_y",
        "ssh",
        "temp_55m",
        "temp_105m",
        "current_5m_u",
        "current_5m_v",
        "mld",
        "salinity_5m",
        "q_net",
    ),
    sst_channel_weight=0.5,
    n_eof_sst=64,
    n_eof_aux=8,
)


def _generate_sst(n_samples, seed, observables, overrides):
    """Load the multivariate ocean state used for SST forecasting.

    The augmented channels are an *internal closure state*: the learned
    one-day operator advances all channels autonomously, but forecast
    verification and downstream figures use SST only.  This avoids feeding
    observed future auxiliary fields during rollout.  Every normalization
    statistic, climatology, and later EOF/dictionary fit is restricted to the
    leading training window.
    """
    import pandas as pd
    import xarray as xr

    params = dict(SST_PARAMS)
    params.update(overrides)
    module_root = Path(__file__).resolve().parent

    def _resolve_data_path(value):
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else module_root / candidate

    path = _resolve_data_path(params["data_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"SST zarr not found at '{path}'. Run python -m experiments.sst.download_sst_data "
            "before loading this dataset.")

    print(f"[sst] loading daily augmented state from {path} ...")
    ds = xr.open_zarr(str(path), consolidated=True)
    version = str(ds.attrs.get("preprocessing_version", ""))
    if version != str(params["preprocessing_version"]):
        raise ValueError(
            f"SST store preprocessing_version={version!r}; expected "
            f"{params['preprocessing_version']!r}. Rebuild it with "
            "python -m experiments.sst.download_sst_data to prepare it.")
    if str(ds.attrs.get("prediction_target", "")) != "sst":
        raise ValueError("SST store does not declare prediction_target='sst'")

    def _coord(obj, names):
        lower = {str(n).lower(): str(n) for n in list(obj.coords) + list(obj.dims)}
        for name in names:
            if name in lower:
                return lower[name]
        raise KeyError(f"none of {names} in {list(lower)}")

    def _as_lat_lon(name):
        da = ds[name]
        renames = {
            _coord(da, ["latitude", "lat"]): "lat",
            _coord(da, ["longitude", "lon"]): "lon",
        }
        da = da.rename(renames)
        return da.transpose("time", "lat", "lon").sortby("lat").sortby("lon")

    channel_names = tuple(str(v) for v in params["channel_names"])
    missing = [name for name in channel_names if name not in ds]
    if missing:
        raise KeyError(f"multivariate SST store is missing channels: {missing}")
    arrays = [_as_lat_lon(name) for name in channel_names]
    sst = arrays[0]
    lsm = ds["land_sea_mask"].rename({
        _coord(ds["land_sea_mask"], ["latitude", "lat"]): "lat",
        _coord(ds["land_sea_mask"], ["longitude", "lon"]): "lon",
    }).transpose("lat", "lon").sortby("lat").sortby("lon")

    lat = sst.lat.values.astype(np.float64)
    lon = sst.lon.values.astype(np.float64)
    times = sst.time.values
    n_time, H, W = sst.shape
    if (H, W) != (32, 64):
        raise ValueError(
            f"SST grid must be 32x64 (latitude x longitude), got {H}x{W}")
    if (
            lsm.shape != (H, W)
            or not np.allclose(lsm.lat.values, lat)
            or not np.allclose(lsm.lon.values, lon)):
        raise ValueError(
            "land_sea_mask is not aligned to the SST latitude/longitude grid")
    dt = pd.DatetimeIndex(pd.to_datetime(times)).normalize()
    expected_start = pd.Timestamp(params["record_start"])
    expected_end = pd.Timestamp(params["record_end"])
    if (
            dt[0] != expected_start
            or dt[-1] != expected_end
            or not np.all(np.diff(dt.values).astype("timedelta64[D]")
                          == np.timedelta64(1, "D"))):
        raise ValueError(
            f"SST record is {dt[0]}..{dt[-1]} and must be a complete daily "
            f"record over {expected_start.date()}..{expected_end.date()}")
    train_cutoff = pd.Timestamp(params["train_end_date"])
    train_end = int(dt.searchsorted(train_cutoff, side="left"))
    if train_end <= 2 or train_end >= n_time or dt[train_end] != train_cutoff:
        raise ValueError(
            f"train_end_date={train_cutoff.date()} is not an interior daily "
            "boundary of the SST record")
    for name, da in zip(channel_names[1:], arrays[1:]):
        if da.shape != sst.shape:
            raise ValueError(f"channel '{name}' has shape {da.shape}, expected {sst.shape}")
        if not np.array_equal(da.time.values, times):
            raise ValueError(f"channel '{name}' is not exactly aligned to the SST daily times")
        if not (np.allclose(da.lat.values, lat) and np.allclose(da.lon.values, lon)):
            raise ValueError(f"channel '{name}' is not on the SST latitude/longitude grid")

    raw_sst = sst.values.astype(np.float32)
    ocean_mask = ((lsm.values.astype(np.float64) < 0.5)
                  & (np.sum(np.isfinite(raw_sst[:train_end]), axis=0) > 0))
    n_channels = len(channel_names)
    fields = np.empty((n_time, n_channels, H, W), dtype=np.float32)
    channel_mean = np.empty(n_channels, dtype=np.float64)
    channel_std = np.empty(n_channels, dtype=np.float64)
    channel_valid_masks = np.zeros((n_channels, H, W), dtype=bool)

    for c, (name, da) in enumerate(zip(channel_names, arrays)):
        raw = da.values.astype(np.float32)
        valid = ocean_mask & (np.sum(np.isfinite(raw[:train_end]), axis=0)
                              >= max(1, train_end // 2))
        if not np.any(valid):
            raise ValueError(f"channel '{name}' has no valid training-window ocean values")
        mean_c = float(np.nanmean(raw[:train_end, valid]))
        std_c = float(np.nanstd(raw[:train_end, valid])) + 1e-6
        filled = np.where(valid[None, :, :] & np.isfinite(raw), raw, mean_c)
        fields[:, c] = ((filled - mean_c) / std_c).astype(np.float32)
        channel_mean[c], channel_std[c] = mean_c, std_c
        channel_valid_masks[c] = valid

    # SST day-of-year climatology in physical units, training window only.
    calendar_day = calendar_day_index(times)
    sst_filled = fields[:, 0] * channel_std[0] + channel_mean[0]
    clim_fallback = sst_filled[:train_end].mean(axis=0)
    climatology = np.tile(clim_fallback[None, :, :], (367, 1, 1)).astype(np.float32)
    for day in range(1, 367):
        idx_day = np.where(calendar_day[:train_end] == day)[0]
        if len(idx_day):
            climatology[day] = sst_filled[:train_end][idx_day].mean(axis=0)

    lat_w = np.clip(np.cos(np.deg2rad(lat)), 0.0, None)
    sst_weights_2d = lat_w[:, None] * ocean_mask.astype(np.float64)
    sst_weights_2d /= sst_weights_2d.sum()

    sst_channel_weight = float(params["sst_channel_weight"])
    if not 0.0 < sst_channel_weight < 1.0:
        raise ValueError("sst_channel_weight must lie strictly between 0 and 1")
    channel_weights = np.full(
        n_channels, (1.0 - sst_channel_weight) / max(1, n_channels - 1),
        dtype=np.float64)
    channel_weights[0] = sst_channel_weight
    weights_by_channel = np.zeros((n_channels, H, W), dtype=np.float64)
    for c in range(n_channels):
        wc = lat_w[:, None] * channel_valid_masks[c].astype(np.float64)
        wc /= wc.sum()
        weights_by_channel[c] = channel_weights[c] * wc
    weights_flat = weights_by_channel.reshape(-1)
    weights_flat /= weights_flat.sum()

    gx = (lat - lat.min()) / (lat.max() - lat.min() + 1e-12)
    gy = (lon - lon.min()) / (lon.max() - lon.min() + 1e-12)
    fno_grid = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1)
    spatial_coords = 2.0 * fno_grid.reshape(-1, 2) - 1.0
    channel_one_hot = np.repeat(np.eye(n_channels, dtype=np.float64), H * W, axis=0)
    basis_coords = np.concatenate(
        [np.tile(spatial_coords, (n_channels, 1)), channel_one_hot], axis=1)

    n_train_times = int(params["n_train_times"])
    if not 2 <= n_train_times <= train_end - 2:
        raise ValueError(
            f"n_train_times={n_train_times} must lie in "
            f"[2, {train_end - 2}] to avoid duplicate training pairs")
    pair_idx = np.linspace(
        0, train_end - 3, n_train_times, dtype=int)
    if n_train_times == train_end - 2:
        # The formal SST protocols use every complete (t,t+1,t+2) start.
        # Slices keep these three large arrays as views of ``fields`` instead
        # of allocating almost three additional gigabytes on the host.
        u0 = fields[:train_end - 2]
        uT = fields[1:train_end - 1]
        u2T = fields[2:train_end]
    else:
        u0 = fields[pair_idx]
        uT = fields[pair_idx + 1]
        u2T = fields[pair_idx + 2]
    fingerprint = str(ds.attrs.get("data_fingerprint", ""))
    if not fingerprint:
        payload = (
            f"{path.name}|{str(times[0])}|{str(times[-1])}|"
            f"{n_time}x{n_channels}x{H}x{W}|{','.join(channel_names)}")
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    training_hash = hashlib.sha256()
    training_hash.update(fingerprint.encode("utf-8"))
    training_hash.update(pair_idx.astype(np.int64).tobytes())
    training_hash.update(np.asarray(
        [train_end, n_time, n_channels, H, W], dtype=np.int64).tobytes())
    training_hash.update(channel_weights.astype(np.float64).tobytes())
    training_hash.update(channel_mean.astype(np.float64).tobytes())
    training_hash.update(channel_std.astype(np.float64).tobytes())
    training_hash.update(memoryview(
        np.ascontiguousarray(channel_valid_masks)).cast("B"))
    training_hash.update(memoryview(
        np.ascontiguousarray(fields[:train_end])).cast("B"))
    training_hash.update(b"train-window-channel-zscore-v1")
    training_fingerprint = training_hash.hexdigest()[:20]

    print(
        f"[sst] {n_time} days, {n_channels} channels at {W}x{H}; "
        f"training pairs from days [0, {train_end}) "
        f"({str(times[0])[:10]} .. {str(times[train_end - 1])[:10]}); "
        f"forecast test window = {n_time - train_end} days")
    return dict(
        system="sst",
        kind="grid2d_multichannel",
        T=float(params["tau"]),
        u0=u0,
        uT=uT,
        u2T=u2T,
        fno_grid=fno_grid,
        basis_coords=basis_coords,
        weights=weights_flat,
        meta=dict(
            params=params,
            lat=lat,
            lon=lon,
            times=times,
            n_time=int(n_time),
            train_end=int(train_end),
            n_channels=int(n_channels),
            channel_names=channel_names,
            sst_index=0,
            channel_mean=channel_mean.astype(np.float32),
            channel_std=channel_std.astype(np.float32),
            channel_valid_masks=channel_valid_masks,
            channel_weights=channel_weights.astype(np.float32),
            state_fields=fields,
            sst_fields=fields[:, 0],
            ocean_mask=ocean_mask,
            weights_2d=sst_weights_2d,
            climatology=climatology,
            climatology_calendar="month-day-leap-v1",
            calendar_day_index=calendar_day,
            data_path=str(path),
            data_fingerprint=fingerprint,
            training_fingerprint=training_fingerprint,
            train_pair_indices=pair_idx,
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_GENERATORS = {
    "ou": _generate_ou,
    "triple_well": _generate_triple_well,
    "lorenz63": _generate_lorenz63,
    "lorenz96": _generate_lorenz96,
    "sst": _generate_sst,
}

_DEFAULT_N_SAMPLES = {
    "ou": 256,
    "triple_well": 64,
    "lorenz63": 12,
    "lorenz96": 12,
    "sst": 0,
}


def generate_kbe_dataset(system, n_samples=None, seed=2026, observables=None,
                         cache_path=None, **overrides):
    """Generate KBE observation data for one of the built-in systems.

    :param system: "ou" | "triple_well" | "lorenz63" | "lorenz96" | "sst"
    :param n_samples: number of observable samples (lorenz63/lorenz96: number
        of random sensor-curve families in addition to the 4 deterministic ones)
    :param observables: optional list of callables replacing the built-in
        random observable bank (signature depends on the system: f(x) for ou,
        f(X, Y) for triple_well, f(states) for lorenz63/lorenz96)
    :param cache_path: if given, the generated dataset dict is pickled there
        and loaded on later calls instead of being regenerated
    :param overrides: overrides for the system parameter dict
    """
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            dataset = pickle.load(f)
        print(f"[data] loaded dataset from {cache_path}")
        return dataset
    key = str(system).lower()
    if key not in _GENERATORS:
        raise ValueError(f"Unknown system '{system}'. Available: {sorted(_GENERATORS)}")
    if n_samples is None:
        n_samples = _DEFAULT_N_SAMPLES[key]
    dataset = _GENERATORS[key](int(n_samples), seed, observables, overrides)
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"[data] saved dataset to {cache_path}")
    return dataset
