"""Methodology layer: neural-operator BSO learning with pluggable EDMD solvers.

Single entry point:

    result = run_bso_pipeline(dataset, config=None, solver_file="solver_edmd.py")

`dataset` comes from kbe_observation_data.generate_kbe_dataset. `solver_file`
names a python file exposing a `DictionaryTrainer` class (see solver_edmd.py /
solver_resdmd.py); changing the file name switches the dictionary-training
method while everything else stays identical.

Pipeline:
    Phase A  train a neural operator (FNO) on the one-step KBE pairs u0 -> uT
             (with an optional two-step semigroup consistency term using u2T).
    Phase B  jointly train an orthonormal neural dictionary Psi (and optionally
             fine-tune the operator) against the solver's dictionary loss,
             refreshing K = (G + a*I)^{-1} A from the current (Psi, U Psi)
             at every outer epoch.
    Final    weighted Galerkin EDMD in float64: K, discrete eigenvalues mu,
             generator eigenvalues log(mu)/T, eigenfunctions, r_eff, cond(G).

Returns a BSOResult carrying the trained operator, the dictionary network,
the evaluated dictionary matrix Psi, and the BSO matrix K.
"""

from __future__ import annotations

import copy
import importlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG: dict[str, Any] = dict(
    seed=2026,
    operator="fno",              # fno | sfno | pino | deeponet | kernel
    physics_weight=1.0,          # PINO only: weight of the KBE residual loss
    model_kwargs=dict(modes=24, width=96, n_layers=4),
    train_kwargs=dict(epochs=1500, lr=6e-4, bs=24, weight_decay=7.5e-5,
                      sg_weight=0.25, val_fraction=0.10, print_every=200),
    n_basis=18,
    reg_alpha=1e-5,
    projection_steps=1,          # compose the operator m times for the Galerkin projection
    training_projection_steps=None,  # defaults to projection_steps
    projection_operator="joint",    # joint | phase_a
    symmetrize_K=False,
    pin_constant=True,           # K[:,0]=e0: the constant mode carries mu=1 exactly
    stabilize_max_radius=None,   # e.g. 0.999 clips |mu| for stable long rollouts
    basis_kwargs=dict(hidden=192, depth=4, num_freq=6),
    joint_kwargs=dict(epochs=150, model_lr=5e-5, basis_lr=2e-4, basis_steps=2,
                      inv_weight=2.0, data_weight=1.0, print_every=25),
)


def set_seed(seed: int = 2026) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _merge_config(config: dict | None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


# ---------------------------------------------------------------------------
# Operator backends (all neural operators come from existing packages)
#   fno      neuralop.models.FNO                      (default)
#   sfno     neuralop.models.SFNO                     (needs torch_harmonics)
#   pino     neuralop.models.FNO + KBE physics residual loss
#   deeponet deepxde.nn.DeepONetCartesianProd
#   kernel   classical kernel ridge regression operator (closed form, no NN)
# ---------------------------------------------------------------------------

OPERATOR_CHOICES = ("fno", "sfno", "pino", "deeponet", "kernel")


class _NeuralOpWrapper(nn.Module):
    """Adapt a neuralop model to the repository's field interface.

    Scalar grid data use ``(B, H, W)`` at the wrapper boundary, while an
    augmented geophysical state uses ``(B, C, H, W)``.  In both cases the
    wrapped neuralop model sees an explicit channel dimension.
    """

    def __init__(self, core, kind, n_channels=1, residual=False):
        super().__init__()
        self.core = core
        self.kind = kind
        self.n_channels = int(n_channels)
        self.residual = bool(residual)

    def forward(self, u, grid):
        scalar = self.n_channels == 1 and u.ndim == self.core.n_dim + 1
        inputs = u.unsqueeze(1) if scalar else u
        out = self.core(inputs)
        if self.residual:
            out = inputs + out
        return out.squeeze(1) if scalar else out


def _spherical_lat_basis(lat_deg, modes_l, modes_m):
    """Orthonormalized spherical-harmonic latitude bases, one block per
    longitudinal wavenumber m: for each m the columns span
    span{P_l^m(cos theta), l = m..m+modes_l-1} on the given latitude grid,
    orthonormalized under the sin(theta) quadrature weights (weighted QR).
    Returns (Bs, sqw): Bs is (modes_m, n_lat, modes_l) with orthonormal
    columns in the sqrt-weighted inner product, sqw the sqrt quadrature
    weights. Forward coefficients: a_m = Bs[m]^T (sqw * F_m); inverse:
    F_m = Bs[m] a_m / sqw."""
    try:
        from scipy.special import sph_harm
    except ImportError:                      # scipy >= 1.17: renamed, args reordered
        from scipy.special import sph_harm_y

        def sph_harm(m, n, theta_az, phi_polar):
            return sph_harm_y(n, m, phi_polar, theta_az)
    theta = np.deg2rad(90.0 - np.asarray(lat_deg, dtype=np.float64))
    w = np.sin(theta)
    w = np.clip(w / w.sum(), 1e-8, None)
    sqw = np.sqrt(w)
    H = len(theta)
    Bs = np.zeros((modes_m, H, modes_l), dtype=np.float64)
    for m in range(modes_m):
        cols = [sph_harm(m, l, 0.0, theta).real
                for l in range(m, m + modes_l)]
        A = np.stack(cols, axis=1) * sqw[:, None]
        Q, _ = np.linalg.qr(A)
        Bs[m] = Q
    return Bs, sqw


class _SphericalConv(nn.Module):
    """Learned spectral convolution in the spherical-harmonic basis:
    rFFT in longitude, projection onto the per-m latitude bases, complex
    channel-mixing weights per (l, m) mode, inverse transforms back."""

    def __init__(self, ch_in, ch_out, Bs, sqw):
        super().__init__()
        modes_m, H, modes_l = Bs.shape
        self.modes_l, self.modes_m = modes_l, modes_m
        self.register_buffer("Bs", torch.as_tensor(Bs, dtype=torch.float32))
        self.register_buffer("sqw", torch.as_tensor(sqw, dtype=torch.float32))
        scale = 1.0 / max(1, ch_in * ch_out)
        self.weight = nn.Parameter(
            scale * torch.randn(ch_in, ch_out, modes_l, modes_m, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        F = torch.fft.rfft(x, dim=-1)                       # (B, C, H, W//2+1)
        Fm = F[..., :self.modes_m] * self.sqw[None, None, :, None]
        a = torch.einsum("mhl,bchm->bclm", self.Bs.to(F.dtype), Fm)
        b = torch.einsum("bilm,iolm->bolm", a, self.weight)
        Fh = torch.einsum("mhl,bolm->bohm", self.Bs.to(F.dtype), b)
        Fh = Fh / self.sqw[None, None, :, None]
        out = torch.zeros(B, b.shape[1], H, F.shape[-1], dtype=F.dtype, device=F.device)
        out[..., :self.modes_m] = Fh
        return torch.fft.irfft(out, n=W, dim=-1)


class _SFNOWrapper(nn.Module):
    """In-house spherical FNO for lat/lon grids (torch_harmonics-free):
    the FNO template with the 2-D Fourier layers replaced by spherical
    spectral convolutions, giving the spherical bandlimiting inductive bias
    appropriate for fields on the sphere."""

    def __init__(self, lat_deg, modes_l=12, modes_m=24, width=64, n_layers=4,
                 n_channels=1, residual=False):
        super().__init__()
        self.n_channels = int(n_channels)
        self.residual = bool(residual)
        Bs, sqw = _spherical_lat_basis(lat_deg, int(modes_l), int(modes_m))
        self.lift = nn.Conv2d(self.n_channels, width, kernel_size=1)
        self.convs = nn.ModuleList(
            [_SphericalConv(width, width, Bs, sqw) for _ in range(int(n_layers))])
        self.ws = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(int(n_layers))])
        self.proj = nn.Sequential(nn.Conv2d(width, 2 * width, kernel_size=1),
                                  nn.GELU(),
                                  nn.Conv2d(2 * width, self.n_channels, kernel_size=1))

    def forward(self, u, grid):
        scalar = self.n_channels == 1 and u.ndim == 3
        inputs = u.unsqueeze(1) if scalar else u
        z = self.lift(inputs)
        for conv, w in zip(self.convs, self.ws):
            z = z + nn.functional.gelu(conv(z) + w(z))
        out = self.proj(z)
        if self.residual:
            out = inputs + out
        return out.squeeze(1) if scalar else out


class _DeepONetWrapper(nn.Module):
    """deepxde DeepONetCartesianProd: branch = field values, trunk = coords."""

    def __init__(self, coords, p, width=128, depth=3, **_):
        super().__init__()
        import deepxde as dde
        coords = torch.as_tensor(coords, dtype=torch.float32)
        if coords.ndim == 1:
            coords = coords[:, None]
        self.register_buffer("trunk_x", coords)
        self.core = dde.nn.DeepONetCartesianProd(
            layer_sizes_branch=[p] + [width] * depth,
            layer_sizes_trunk=[coords.shape[1]] + [width] * depth,
            activation="tanh",
            kernel_initializer="Glorot normal",
        )

    def forward(self, u, grid):
        shape = u.shape
        flat = u.reshape(shape[0], -1)
        out = self.core((flat, self.trunk_x))
        return out.reshape(shape)


class KernelRidgeOperator(nn.Module):
    """Classical numerical baseline: RBF kernel ridge regression in function space.

    Fitted in closed form on the training pairs; forward is differentiable in
    torch so the joint dictionary phase can backprop through it.
    """

    def __init__(self, sigma=None, reg=1e-6, **_):
        super().__init__()
        self.sigma = sigma
        self.reg = reg
        self.register_buffer("X", torch.zeros(1, 1))
        self.register_buffer("alpha", torch.zeros(1, 1))
        self.fitted = False

    @torch.no_grad()
    def fit(self, u0, uT):
        X = u0.detach()
        d2 = torch.cdist(X, X) ** 2
        if self.sigma is None:
            self.sigma = float(torch.sqrt(0.5 * torch.median(d2[d2 > 0])))
        Kmat = torch.exp(-d2 / (2.0 * self.sigma**2))
        n = X.shape[0]
        eye = torch.eye(n, device=X.device, dtype=X.dtype)
        self.alpha = torch.linalg.solve(Kmat + self.reg * n * eye, uT.detach())
        self.X = X
        self.fitted = True

    def forward(self, u, grid):
        shape = u.shape
        flat = u.reshape(shape[0], -1)
        d2 = torch.cdist(flat, self.X) ** 2
        k = torch.exp(-d2 / (2.0 * self.sigma**2))
        return (k @ self.alpha).reshape(shape)


def _attach_state_mask(model, dataset):
    """Pin structurally unavailable multichannel grid points to normalized 0.

    Some auxiliary products do not cover the full SST ocean grid.  Their
    missing points have zero loss weight and are filled by the training mean;
    projecting those points back to normalized zero after every operator
    application prevents unconstrained land/high-latitude outputs from
    feeding back into later autoregressive steps through spectral convolutions.
    """
    if dataset["kind"] == "grid2d_multichannel":
        masks = dataset.get("meta", {}).get("channel_valid_masks")
        if masks is None:
            raise ValueError("multichannel grid dataset is missing valid masks")
        model.register_buffer(
            "_state_valid_mask",
            torch.as_tensor(np.asarray(masks), dtype=torch.float32))
    return model


def build_operator(operator_kind, dataset, model_kwargs):
    """Instantiate one of the packaged operator backends for this dataset."""
    kind = dataset["kind"]
    is_grid2d = kind in ("grid2d", "grid2d_multichannel")
    n_channels = int(dataset.get("meta", {}).get("n_channels", 1))
    key = str(operator_kind).lower()
    if key not in OPERATOR_CHOICES:
        raise ValueError(f"Unknown operator '{operator_kind}'. Choices: {OPERATOR_CHOICES}")
    if kind == "grid2d_multichannel" and key in ("deeponet", "pino"):
        raise ValueError(
            f"Operator '{key}' is not implemented for multichannel grid states; "
            "use FNO, SFNO, or kernel.")
    kw = dict(model_kwargs)
    p = int(np.prod(np.asarray(dataset["u0"]).shape[1:]))
    coords = dataset["fno_grid"] if not is_grid2d else dataset["fno_grid"].reshape(-1, 2)

    if key in ("fno", "pino"):
        from neuralop.models import FNO
        from neuralop.layers.embeddings import GridEmbeddingND
        residual = bool(kw.pop("residual", False))
        modes = kw.pop("modes", 16)
        n_modes = (modes, modes) if (is_grid2d and isinstance(modes, int)) else (
            tuple(modes) if not isinstance(modes, int) else (modes,))
        core = FNO(n_modes=n_modes, in_channels=n_channels, out_channels=n_channels,
                   hidden_channels=kw.pop("width", 64), n_layers=kw.pop("n_layers", 4),
                   positional_embedding=GridEmbeddingND(
                       in_channels=n_channels,
                       dim=len(n_modes),
                       grid_boundaries=[[0.0, 1.0]] * len(n_modes),
                   ))
        return _attach_state_mask(
            _NeuralOpWrapper(
                core, kind, n_channels=n_channels, residual=residual), dataset)
    if key == "sfno":
        if not is_grid2d:
            raise ValueError("SFNO only applies to 2D grid datasets.")
        lat = dataset.get("meta", {}).get("lat")
        if lat is None:
            raise ValueError("SFNO needs latitude values in dataset['meta']['lat'].")
        modes = kw.pop("modes", 16)
        residual = bool(kw.pop("residual", False))
        n_modes = (modes, modes) if isinstance(modes, int) else tuple(modes)
        return _attach_state_mask(
            _SFNOWrapper(
                lat, modes_l=n_modes[0], modes_m=n_modes[1],
                width=kw.pop("width", 64), n_layers=kw.pop("n_layers", 4),
                n_channels=n_channels, residual=residual),
            dataset)
    if key == "deeponet":
        return _DeepONetWrapper(coords, p, **kw)
    if key == "kernel":
        model = KernelRidgeOperator(**kw)
        model.n_channels = n_channels
        return _attach_state_mask(model, dataset)
    raise AssertionError(key)


def _pino_generator(dataset):
    """Discrete KBE generator L as a torch tensor for the PINO physics loss."""
    kind = dataset["kind"]
    meta = dataset["meta"]
    if kind == "grid1d":
        L = np.asarray(meta["solver"].K, dtype=np.float32)
        return torch.as_tensor(L, device=device)
    if kind == "grid2d":
        L = meta["solver"].operator.tocoo()
    elif kind == "cloud1d":
        L = meta["generator"].tocoo()
    else:
        raise ValueError(
            "PINO physics loss is not available for kind='sensor1d': the Lorenz96 "
            "generator acts across hidden nodes, not on individual sensor curves.")
    idx = torch.as_tensor(np.stack([L.row, L.col]), dtype=torch.long)
    val = torch.as_tensor(L.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=L.shape, device=device).coalesce()


def _apply_generator(L, v):
    """L @ v^T for dense or sparse L; v is (B, p)."""
    if L.is_sparse:
        return torch.sparse.mm(L, v.T).T
    return v @ L.T


def apply_operator(model, fields, grid, kind):
    """Apply the operator to a batch of fields; fields are (B, p) flat."""
    if kind in ("grid2d", "grid2d_multichannel"):
        H, W = grid.shape[0], grid.shape[1]
        C = int(getattr(model, "n_channels", 1))
        shape = (-1, H, W) if C == 1 else (-1, C, H, W)
        out = model(fields.reshape(shape), grid)
        out = out.reshape(fields.shape[0], -1)
        state_mask = getattr(model, "_state_valid_mask", None)
        if state_mask is not None:
            out = out * state_mask.reshape(1, -1).to(
                device=out.device, dtype=out.dtype)
        return out
    return model(fields, grid)


def apply_operator_n(model, fields, grid, kind, n_steps=1):
    """Compose the operator n_steps times (effective horizon n_steps * tau)."""
    out = fields
    for _ in range(int(n_steps)):
        out = apply_operator(model, out, grid, kind)
    return out


# ---------------------------------------------------------------------------
# Dictionary network + weighted orthonormalization
# ---------------------------------------------------------------------------

class BasisNet(nn.Module):
    """MLP over Fourier coordinate features -> n_learn raw basis columns.

    An optional fixed envelope (p,) multiplies the outputs; use it to confine
    learned columns to the bulk of the sampling measure so that near-zero-weight
    boundary regions cannot be exploited.
    """

    def __init__(self, coord_dim, n_learn, hidden=192, depth=4, num_freq=6,
                 envelope=None):
        super().__init__()
        self.num_freq = num_freq
        in_dim = coord_dim * (1 + 2 * num_freq)
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, n_learn)]
        self.net = nn.Sequential(*layers)
        if envelope is not None:
            self.register_buffer(
                "envelope",
                torch.as_tensor(np.asarray(envelope), dtype=torch.float32).reshape(-1, 1))
        else:
            self.envelope = None

    def features(self, coords):
        feats = [coords]
        for k in range(1, self.num_freq + 1):
            feats.append(torch.sin(np.pi * k * coords))
            feats.append(torch.cos(np.pi * k * coords))
        return torch.cat(feats, dim=-1)

    def forward(self, coords):
        out = self.net(self.features(coords))
        if self.envelope is not None:
            out = out * self.envelope
        return out


class FeatureMixBasis(nn.Module):
    """Trainable linear mixture over a fixed feature bank (p, F).

    Exact-capacity dictionary parametrization: any function in the span of the
    features is representable, including polynomially growing ones that
    saturating MLPs cannot express.
    """

    def __init__(self, features, n_learn):
        super().__init__()
        self.register_buffer("feats", torch.as_tensor(np.asarray(features),
                                                      dtype=torch.float32))
        self.mix = nn.Linear(self.feats.shape[1], n_learn, bias=False)

    def forward(self, _coords):
        return self.mix(self.feats)


def distill_basis(basis_net, coords, weights, targets, *, epochs=800, lr=1e-2,
                  print_every=200):
    """Supervised distillation of the dictionary onto target columns (p, k).

    For a FeatureMixBasis the weighted least-squares solution initializes the
    mixture exactly; gradient steps then only polish.
    """
    targets_t = torch.as_tensor(np.asarray(targets), dtype=torch.float32,
                                device=device).T                    # (k, p)
    wrow = weights.reshape(1, -1)
    if isinstance(basis_net, FeatureMixBasis):
        wsqrt = torch.sqrt(weights).reshape(-1, 1)
        sol = torch.linalg.lstsq(basis_net.feats * wsqrt,
                                 targets_t.T * wsqrt).solution      # (F, k)
        with torch.no_grad():
            basis_net.mix.weight.copy_(sol.T)
    opt = torch.optim.Adam(basis_net.parameters(), lr=lr)
    history = []
    for ep in range(epochs):
        out = basis_net(coords).T
        num = torch.sum(wrow * (out - targets_t) ** 2, dim=1)
        den = torch.sum(wrow * targets_t**2, dim=1) + 1e-12
        loss = torch.mean(num / den)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(dict(epoch=ep, fit=float(loss)))
        if print_every and (ep % print_every == 0 or ep == epochs - 1):
            print(f"  [distill] epoch {ep:5d}  fit {float(loss):.4e}")
    return history


def build_basis(basis_net, coords, weights, fixed=None):
    """Orthonormal dictionary Psi (n_basis, p) wrt the weighted inner product.

    Leading row is the constant function (QR preserves the span order); any
    fixed analytic columns come next, then the learned ones.
    """
    raw = basis_net(coords)                                   # (p, n_learn)
    ones = torch.ones(raw.shape[0], 1, dtype=raw.dtype, device=raw.device)
    cols = [ones, raw] if fixed is None else [ones, fixed, raw]
    R = torch.cat(cols, dim=1)                                 # (p, n_basis)
    wsqrt = torch.sqrt(weights).reshape(-1, 1)
    Q, Rmat = torch.linalg.qr(R * wsqrt, mode="reduced")
    diag = torch.diagonal(Rmat)
    sign = torch.where(diag >= 0, torch.ones_like(diag), -torch.ones_like(diag))
    Q = Q * sign.reshape(1, -1)
    # zero-weight points (e.g. masked land) carry no geometry: their basis
    # values are pinned to 0 instead of dividing by sqrt(0)
    return (Q / wsqrt.clamp_min(1e-12) * (weights > 0).reshape(-1, 1)).T


# ---------------------------------------------------------------------------
# Phase A: operator training
# ---------------------------------------------------------------------------

def _rel_l2(pred, target, w=None):
    if w is None:
        num = torch.linalg.vector_norm(pred - target, dim=-1)
        den = torch.linalg.vector_norm(target, dim=-1) + 1e-10
    else:
        num = torch.sqrt(torch.sum(w * (pred - target) ** 2, dim=-1))
        den = torch.sqrt(torch.sum(w * target**2, dim=-1)) + 1e-10
    return torch.mean(num / den)


def _pacing_start(max_duty_cycle):
    duty = float(max_duty_cycle)
    if not 0.0 < duty <= 1.0:
        raise ValueError("max_duty_cycle must lie in (0, 1]")
    if duty >= 1.0 or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return time.perf_counter()


def _pace_cuda(started, max_duty_cycle):
    """Limit average CUDA duty cycle by adding a short idle gap.

    This is a thermal/system-stability control only; it does not alter model
    updates.  CUDA is synchronized solely when pacing is requested so
    experiments retain their original throughput.
    """
    if started is None:
        return
    torch.cuda.synchronize()
    active = time.perf_counter() - started
    idle = active * (1.0 / float(max_duty_cycle) - 1.0)
    if idle > 0:
        time.sleep(idle)


def _batched_operator_rel_l2(
        model, inputs, targets, grid, kind, *, bs, loss_weights=None,
        max_duty_cycle=1.0):
    """Evaluate relative L2 in paced batches instead of one full GPU batch."""
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), int(bs)):
            pacing_started = _pacing_start(max_duty_cycle)
            stop = min(start + int(bs), len(inputs))
            value = _rel_l2(
                apply_operator(model, inputs[start:stop], grid, kind),
                targets[start:stop],
                loss_weights)
            _pace_cuda(pacing_started, max_duty_cycle)
            n_batch = stop - start
            total += float(value) * n_batch
            count += n_batch
    return total / max(count, 1)


def train_operator(model, u0, uT, u2T, grid, kind, *, epochs, lr, bs, weight_decay,
                   sg_weight, val_fraction, print_every, seed=2026,
                   physics=None, physics_weight=1.0, horizon=None, const_weight=0.0,
                   loss_weights=None, mix_augment=0.0, mix_terms=4,
                   max_duty_cycle=1.0, progress_path=None,
                   progress_every=10, progress_metadata=None,
                   training_stop_path=None):
    n = u0.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(val_fraction * n))
    val_idx = torch.as_tensor(perm[:n_val], device=device)
    tr_idx = torch.as_tensor(perm[n_val:], device=device)

    if isinstance(model, KernelRidgeOperator):
        model.fit(u0[tr_idx], uT[tr_idx])
        val = _batched_operator_rel_l2(
            model, u0[val_idx], uT[val_idx], grid, kind, bs=bs,
            max_duty_cycle=max_duty_cycle)
        print(f"  [operator] kernel ridge fitted in closed form "
              f"(sigma={model.sigma:.4f}), val rel-L2 = {val:.4e}")
        return dict(best_val=val, history=[dict(epoch=0, train=val, val=val)])

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val, best_state = float("inf"), None
    history = []
    start_epoch = 0
    progress_path = (
        None if progress_path is None else Path(progress_path))
    training_stop_path = (
        None if training_stop_path is None else Path(training_stop_path))
    progress_signature = dict(
        n_samples=int(n),
        kind=str(kind),
        epochs=int(epochs),
        lr=float(lr),
        bs=int(bs),
        weight_decay=float(weight_decay),
        sg_weight=float(sg_weight),
        val_fraction=float(val_fraction),
        seed=int(seed),
        mix_augment=float(mix_augment),
        mix_terms=int(mix_terms),
        metadata=progress_metadata,
    )
    if progress_path is not None and progress_path.is_file():
        progress = torch.load(
            progress_path, map_location=device, weights_only=False)
        if progress.get("signature") == progress_signature:
            model.load_state_dict(progress["model"], strict=False)
            opt.load_state_dict(progress["optimizer"])
            sched.load_state_dict(progress["scheduler"])
            best_val = float(progress["best_val"])
            best_state = progress.get("best_state")
            history = list(progress.get("history", []))
            start_epoch = int(progress["next_epoch"])
            if progress.get("torch_rng_state") is not None:
                torch.set_rng_state(progress["torch_rng_state"].cpu())
            if (
                    torch.cuda.is_available()
                    and progress.get("cuda_rng_states") is not None):
                torch.cuda.set_rng_state_all([
                    state.cpu() for state in progress["cuda_rng_states"]
                ])
            print(
                f"  [operator] resumed at epoch {start_epoch}/{epochs} "
                f"from {progress_path}")
        else:
            raise RuntimeError(
                "Refusing to overwrite an incompatible progress checkpoint: "
                f"{progress_path}. Move or explicitly remove it only after "
                "verifying the data, model, and training configuration.")

    def save_progress(next_epoch):
        if progress_path is None:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = progress_path.with_suffix(
            progress_path.suffix + ".tmp")
        payload = dict(
            signature=progress_signature,
            next_epoch=int(next_epoch),
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),
            best_val=float(best_val),
            best_state=best_state,
            history=history,
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_states=(
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None),
        )
        torch.save(payload, temporary)
        temporary.replace(progress_path)

    completed_epochs = int(start_epoch)
    stopped_early = False
    if (
            training_stop_path is not None
            and training_stop_path.is_file()
            and best_state is not None):
        stopped_early = start_epoch < int(epochs)
        print(
            f"  [operator] training-stop request found; promoting the "
            f"best state from {start_epoch}/{epochs} completed epochs")

    for ep in range(start_epoch, epochs):
        if stopped_early:
            break
        model.train()
        order = tr_idx[torch.randperm(len(tr_idx), device=device)]
        ep_loss, n_batches = 0.0, 0
        for s in range(0, len(order), bs):
            pacing_started = _pacing_start(max_duty_cycle)
            idx = order[s:s + bs]
            u0_b, uT_b = u0[idx], uT[idx]
            u2T_b = u2T[idx] if u2T is not None else None
            if mix_augment > 0 and float(torch.rand(())) < mix_augment:
                # The KBE/BSO map is linear, so any linear combination of
                # training pairs is an exact new pair. Mixing teaches the
                # network to act linearly on the whole sample span (plain
                # per-sample fitting does not constrain combinations, and
                # cancellation amplifies the error on contrast directions).
                n_mix = int(mix_terms)
                jdx = tr_idx[torch.randint(0, len(tr_idx),
                                           (len(idx), n_mix), device=device)]
                C = torch.randn(len(idx), n_mix, 1, device=device)
                C = C / (torch.linalg.vector_norm(C, dim=1, keepdim=True) + 1e-8)
                mix_u0 = torch.sum(C * u0[jdx], dim=1)
                mix_uT = torch.sum(C * uT[jdx], dim=1)

                def _wnorm(f):
                    if loss_weights is None:
                        return torch.linalg.vector_norm(f, dim=-1)
                    return torch.sqrt(torch.sum(loss_weights * f**2, dim=-1))

                # drop extreme cancellations (tiny targets destabilize rel-L2)
                comp = _wnorm(uT[jdx].reshape(-1, uT.shape[1]))
                comp = comp.reshape(len(idx), n_mix).min(dim=1).values
                ok = (_wnorm(mix_uT) > 0.25 * comp)[:, None]
                u0_b = torch.where(ok, mix_u0, u0_b)
                uT_b = torch.where(ok, mix_uT, uT_b)
                if u2T is not None:
                    u2T_b = torch.where(ok, torch.sum(C * u2T[jdx], dim=1), u2T_b)
            pred = apply_operator(model, u0_b, grid, kind)
            loss = _rel_l2(pred, uT_b, loss_weights)
            if sg_weight > 0 and u2T_b is not None:
                pred2 = apply_operator(model, pred, grid, kind)
                loss = loss + sg_weight * _rel_l2(pred2, u2T_b, loss_weights)
            if physics is not None:
                # Crank-Nicolson KBE residual: (u_T - u_0)/T ~= L (u_T + u_0)/2
                mid = 0.5 * (pred + u0[idx])
                lmid = _apply_generator(physics, mid)
                resid = (pred - u0[idx]) / horizon - lmid
                loss = loss + physics_weight * torch.linalg.vector_norm(resid) / (
                    torch.linalg.vector_norm(lmid) + 1e-8)
            if const_weight > 0:
                # the KBE semigroup preserves constants: U_tau 1 = 1
                ones = torch.ones(1, u0.shape[1], device=device)
                pred_const = apply_operator(model, ones, grid, kind)
                loss = loss + const_weight * torch.mean((pred_const - 1.0) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            _pace_cuda(pacing_started, max_duty_cycle)
            ep_loss += float(loss)
            n_batches += 1
        sched.step()
        val = _batched_operator_rel_l2(
            model, u0[val_idx], uT[val_idx], grid, kind, bs=bs,
            loss_weights=loss_weights, max_duty_cycle=max_duty_cycle)
        history.append(dict(epoch=ep, train=ep_loss / max(n_batches, 1), val=val))
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
        completed_epochs = ep + 1
        stop_requested = (
            training_stop_path is not None
            and training_stop_path.is_file())
        if print_every and (ep % print_every == 0 or ep == epochs - 1):
            print(f"  [operator] epoch {ep:5d}  train {ep_loss / max(n_batches, 1):.4e}  val {val:.4e}")
        if (
                progress_path is not None
                and (
                    stop_requested
                    or ep == 0
                    or (ep + 1) % max(1, int(progress_every)) == 0
                    or ep == epochs - 1)):
            save_progress(ep + 1)
        if stop_requested:
            if best_state is None or not np.isfinite(best_val):
                raise RuntimeError(
                    "training-stop request arrived before a finite validation "
                    "state was available")
            stopped_early = ep + 1 < int(epochs)
            print(
                f"  [operator] stopping after epoch {ep + 1}/{epochs}; "
                f"best validation rel-L2 = {best_val:.4e}")
            break
    if best_state is not None:
        # strict=False: neuralop models add a non-tensor "_metadata" entry
        model.load_state_dict(best_state, strict=False)
    return dict(
        best_val=best_val,
        history=history,
        completed_epochs=int(completed_epochs),
        stopped_early=bool(stopped_early),
    )


# ---------------------------------------------------------------------------
# Phase B: joint dictionary (+ operator) training via a pluggable solver
# ---------------------------------------------------------------------------

def _load_trainer_class(solver_file):
    path = Path(solver_file)
    if not path.is_absolute() and path.parent == Path("."):
        path = Path(__file__).resolve().parent / path
    parent = str(path.resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module(path.stem)
    return module.DictionaryTrainer


def joint_train(model, basis_net, trainer, u0, uT, u2T, grid, coords, weights, kind, *,
                epochs, model_lr, basis_lr, basis_steps, inv_weight, data_weight,
                print_every, smooth_weight=0.0, projection_steps=1, fixed=None,
                trace_weight=0.0, symmetrize_target=False, two_step_weight=0.0,
                cap_mu=None, sg_weight=0.0, max_duty_cycle=1.0):
    opt_basis = torch.optim.Adam(basis_net.parameters(), lr=basis_lr)
    model_params = [p for p in model.parameters() if p.requires_grad]
    opt_model = (torch.optim.Adam(model_params, lr=model_lr)
                 if model_lr > 0 and model_params else None)
    n = u0.shape[0]
    history = []
    for ep in range(epochs):
        pacing_started = _pacing_start(max_duty_cycle)
        with torch.no_grad():
            psi_d = build_basis(basis_net, coords, weights, fixed)
            ev_d = apply_operator_n(model, psi_d, grid, kind, projection_steps)
            K = trainer.compute_K(psi_d, ev_d, weights)
            if symmetrize_target:
                K = 0.5 * (K + K.T)
            if cap_mu is not None:
                # Markov semigroups are non-expansive (|mu| <= 1): clip the
                # target spectrum so spurious operator eigendirections with
                # |mu| > 1 cannot self-reinforce during dictionary training.
                # Requires symmetrize_target (eigh on the symmetric target).
                evals, V = torch.linalg.eigh(0.5 * (K + K.T))
                evals = torch.clamp(evals, min=-float(cap_mu), max=float(cap_mu))
                K = V @ torch.diag(evals) @ V.T
        target = trainer.prepare_target(K)

        last = {}
        for _ in range(basis_steps):
            psi = build_basis(basis_net, coords, weights, fixed)
            evolved = apply_operator_n(model, psi, grid, kind, projection_steps)
            inv_loss = trainer.dictionary_loss(psi, evolved, weights, target)
            loss = inv_weight * inv_loss
            if two_step_weight > 0 and "K" in target:
                # second-horizon consistency: U^2 Psi ~= (K^2)^T Psi
                evolved2 = apply_operator_n(model, evolved, grid, kind, projection_steps)
                K2 = target["K"] @ target["K"]
                pred2 = K2.T @ psi
                wrow2 = weights.reshape(1, -1)
                num2 = torch.sum(wrow2 * (evolved2 - pred2) ** 2, dim=1)
                den2 = torch.sum(wrow2 * evolved2**2, dim=1) + 1e-12
                loss = loss + two_step_weight * torch.mean(torch.sqrt(num2 / den2))
            if trace_weight > 0:
                # VAMP-style reward: trace(K) = sum of eigenvalue real parts;
                # maximizing it steers the subspace toward the slowest modes
                K_live = trainer.compute_K(psi, evolved, weights)
                loss = loss - trace_weight * torch.trace(K_live) / K_live.shape[0]
            rough = torch.zeros((), device=device)
            if smooth_weight > 0:
                if kind == "grid2d":
                    f = psi.reshape(psi.shape[0], grid.shape[0], grid.shape[1])
                    rough = (torch.mean((f[:, 1:, :] - f[:, :-1, :]) ** 2)
                             + torch.mean((f[:, :, 1:] - f[:, :, :-1]) ** 2))
                else:
                    rough = torch.mean((psi[:, 1:] - psi[:, :-1]) ** 2)
                loss = loss + smooth_weight * rough
            data_loss = torch.zeros((), device=device)
            if opt_model is not None and data_weight > 0:
                idx = torch.randint(0, n, (min(24, n),), device=device)
                pred1 = apply_operator(model, u0[idx], grid, kind)
                data_loss = _rel_l2(pred1, uT[idx])
                if sg_weight > 0 and u2T is not None:
                    # two-step semigroup consistency anchors the decay RATES:
                    # a rate error is amplified quadratically over 2 steps,
                    # so the invariance loss cannot drift the slow modes
                    pred2 = apply_operator(model, pred1, grid, kind)
                    data_loss = data_loss + sg_weight * _rel_l2(pred2, u2T[idx])
                loss = loss + data_weight * data_loss
            opt_basis.zero_grad()
            if opt_model is not None:
                opt_model.zero_grad()
            loss.backward()
            opt_basis.step()
            if opt_model is not None:
                opt_model.step()
            last = dict(epoch=ep, inv=float(inv_loss), data=float(data_loss),
                        rough=float(rough))
        history.append(last)
        _pace_cuda(pacing_started, max_duty_cycle)
        if print_every and (ep % print_every == 0 or ep == epochs - 1):
            print(f"  [joint:{trainer.name}] epoch {ep:4d}  inv {last['inv']:.4e}  "
                  f"data {last['data']:.4e}  rough {last['rough']:.4e}")
    return history


# ---------------------------------------------------------------------------
# Deflation: sequential per-mode dictionary learning
# ---------------------------------------------------------------------------

def build_data_operator(u0, uT, weights, reg_rel=1e-8):
    """Linear estimate L of U_tau built from the training pairs.

    L phi = uT^T c(phi) with c = argmin_c ||u0^T c - phi||_rho^2 + eps||c||^2
    (rho-weighted ridge regression onto the sample fields). L serves as a
    data-consistency guard during deflation stages: search directions on which
    the neural operator disagrees with the sample evolution are penalized, so
    spurious operator eigendirections cannot be exploited.
    """
    S = np.asarray(u0, dtype=np.float64).reshape(len(u0), -1)
    ST = np.asarray(uT, dtype=np.float64).reshape(len(uT), -1)
    w = np.asarray(weights, dtype=np.float64).reshape(1, -1)
    SW = S * w
    G = SW @ S.T
    eps = reg_rel * np.trace(G) / G.shape[0]
    return ST.T @ np.linalg.solve(G + eps * np.eye(G.shape[0]), SW)


def deflation_stages(model, grid, coords, weights, kind, L, *, n_stages, epochs,
                     lr=3e-4, res_weight=5.0, cons_weight=5.0, cap_weight=10.0,
                     smooth_weight=0.05, res_tol=2e-3, cons_tol=5e-3, mu_max=1.005,
                     seed0=2033, basis_kwargs=None, print_stages=True):
    """Sequential per-mode dictionary learning (deflation), 1D grids only.

    Each stage trains a single function, rho-orthogonalized against the
    constant and all previously accepted columns, to maximize the Rayleigh
    quotient <phi, U phi>_rho (the slowest remaining mode) subject to
      - an eigenpair residual penalty ||U phi - <phi, U phi> phi||_rho,
      - a non-expansiveness cap (Markov semigroups satisfy |mu| <= 1),
      - consistency with the data operator L (see build_data_operator).
    A stage column is frozen into the dictionary only if its final residual
    and data consistency are below tolerance -- a purely data-driven
    acceptance rule; no reference spectrum is involved. Returns the accepted
    columns (p, k) (or None if none) and a per-stage log.
    """
    basis_kwargs = dict(basis_kwargs or {})
    wrow = weights.reshape(1, -1)
    e0 = torch.ones_like(weights)
    e0 = e0 / torch.sqrt(torch.sum(weights * e0**2))
    found = [e0]
    log = []
    L_t = torch.as_tensor(L, dtype=torch.float32, device=weights.device)

    def deflate(net):
        raw = net(coords).squeeze(-1)
        E = torch.stack(found)
        phi = raw - (E * wrow @ raw) @ E
        return phi / (torch.sqrt(torch.sum(weights * phi**2)) + 1e-12)

    def terms(phi):
        Uphi = apply_operator(model, phi.unsqueeze(0), grid, kind).squeeze(0)
        ray = torch.sum(weights * phi * Uphi)
        den = torch.sum(weights * Uphi**2) + 1e-12
        res = torch.sum(weights * (Uphi - ray * phi) ** 2) / den
        cons = torch.sum(weights * (Uphi - L_t @ phi) ** 2) / den
        return ray, res, cons

    for k in range(1, n_stages + 1):
        set_seed(seed0 + k)
        net = BasisNet(coord_dim=coords.shape[1], n_learn=1,
                       **basis_kwargs).to(weights.device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(epochs):
            for _ in range(2):
                phi = deflate(net)
                ray, res, cons = terms(phi)
                cap = torch.relu(ray - 0.999) ** 2
                rough = torch.mean((phi[1:] - phi[:-1]) ** 2)
                loss = (-ray + res_weight * res + cons_weight * cons
                        + cap_weight * cap + smooth_weight * rough)
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            phi = deflate(net)
            ray, res, cons = (float(t) for t in terms(phi))
        # semigroup non-expansiveness: a stage claiming |mu| > 1 is spurious
        ok = res < res_tol and cons < cons_tol and abs(ray) <= mu_max
        log.append(dict(stage=k, mu=ray, res=res, cons=cons, accepted=ok))
        if print_stages:
            print(f"  [deflation] stage {k}: mu={ray:+.4f}  res={res:.4f}  "
                  f"cons={cons:.4f}  -> {'accepted' if ok else 'rejected'}")
        if ok:
            found.append(phi.detach())
    if len(found) == 1:
        return None, log
    return torch.stack(found[1:], dim=1), log


# ---------------------------------------------------------------------------
# Result container and spectral analysis
# ---------------------------------------------------------------------------

@dataclass
class BSOResult:
    system: str
    kind: str
    T: float
    solver_name: str
    operator: nn.Module          # operator used for the BSO projection
    operator_pure: nn.Module     # phase-A operator
    basis_net: nn.Module
    psi: np.ndarray                    # (n_basis, p) orthonormal dictionary
    evolved: np.ndarray                # U_tau Psi via the operator
    K: np.ndarray                      # BSO matrix, U Psi ~= K^T Psi
    mu: np.ndarray                     # discrete eigenvalues of K
    lam: np.ndarray                    # generator eigenvalues log(mu)/T
    eigenfunctions: np.ndarray         # (n_basis, p), rows = eigenfunctions
    r_eff: int
    cond_G: float
    weights: np.ndarray
    grid: np.ndarray
    operator_val_rel_l2: float
    history: dict = field(default_factory=dict)
    operator_joint: nn.Module | None = None  # jointly refined operator

    def summary(self) -> str:
        lines = [
            f"system={self.system}  solver={self.solver_name}  n_basis={self.psi.shape[0]}",
            f"operator val rel-L2 = {self.operator_val_rel_l2:.4e}",
            f"r_eff = {self.r_eff}/{self.psi.shape[0]}  cond(G) = {self.cond_G:.3e}",
            "top generator eigenvalues (log mu / T):",
        ]
        for lam in self.lam[:8]:
            lines.append(f"  {lam.real:+.6f} {lam.imag:+.6f}j")
        return "\n".join(lines)


def _final_spectral_analysis(psi, evolved, weights, reg_alpha, T, max_radius=None,
                             pin_constant=True, symmetrize=False, pin_eigs=None):
    psi = np.asarray(psi, dtype=np.float64)
    evolved = np.asarray(evolved, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(1, -1)
    G = (psi * w) @ psi.T
    A = (psi * w) @ evolved.T
    K = np.linalg.solve(G + reg_alpha * np.eye(G.shape[0]), A)
    if pin_constant:
        # the first dictionary row is the constant function and U_tau 1 = 1,
        # so the constant mode carries mu = 1 exactly
        K[:, 0] = 0.0
        K[0, 0] = 1.0
    if symmetrize:
        # for reversible diffusions the BSO operator is self-adjoint in
        # L2(rho); enforcing symmetry gives perfectly conditioned eigenpairs
        K = 0.5 * (K + K.T)
        if pin_constant:
            K[:, 0] = 0.0
            K[0, :] = 0.0
            K[0, 0] = 1.0
        if pin_eigs:
            # dictionary columns carrying a data-driven eigenpair certificate
            # (deflation stages accepted on residual/consistency tolerances)
            # are pinned like the constant: their measured Rayleigh quotient
            # becomes the eigenvalue, so the surrounding columns cannot mix
            # them away through operator noise
            for i, mu_i in pin_eigs:
                K[int(i), :] = 0.0
                K[:, int(i)] = 0.0
                K[int(i), int(i)] = float(mu_i)
        mu, V = np.linalg.eigh(K)
        mu = mu.astype(complex)
    else:
        mu, V = np.linalg.eig(K)
    if max_radius is not None:
        # only genuinely expanding modes (|mu| > 1) are spurious for a Markov
        # semigroup; clip them just inside the unit circle. Modes with
        # |mu| <= 1 (including the pinned constant at exactly 1) are left as is.
        radii = np.abs(mu)
        over = radii > 1.0
        n_clipped = int(np.sum(over))
        if n_clipped:
            mu = np.where(over, mu * (max_radius / radii), mu)
            K = np.real(V @ np.diag(mu) @ np.linalg.inv(V))
            print(f"[pipeline] spectral stabilization: clipped {n_clipped}/{len(mu)} "
                  f"eigenvalues to |mu| <= {max_radius}")
    lam = np.log(mu.astype(complex)) / T
    order = np.argsort(-lam.real)
    mu, lam, V = mu[order], lam[order], V[:, order]
    efuncs = V.T @ psi
    svals = np.linalg.svd(G, compute_uv=False)
    r_eff = int(np.sum(svals > 1e-8 * svals.max()))
    cond_G = float(svals.max() / max(svals.min(), 1e-300))
    return K, mu, lam, efuncs, r_eff, cond_G


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_bso_pipeline(dataset, config=None, solver_file="solver_edmd.py", operator=None,
                         checkpoint_dir=None, force_retrain=False,
                         training_stop_path=None):
    """Train operator + dictionary on a KBE dataset; return the BSO model.

    :param dataset: dict from kbe_observation_data.generate_kbe_dataset
    :param config: overrides merged into DEFAULT_CONFIG
    :param solver_file: python file exposing DictionaryTrainer (the EDMD-class
        method used for K and the dictionary loss); switch the file name to
        switch the method, e.g. "solver_resdmd.py"
    :param operator: operator backend, one of OPERATOR_CHOICES
        ("fno" default | "sfno" | "pino" | "deeponet" | "kernel");
        overrides config["operator"]
    :param checkpoint_dir: if given, long computations are cached there and
        loaded on later runs: "<operator>.pt" (trained neural operator, named
        after the operator backend, e.g. "fno.pt") and the dictionary file
        cfg["bso_file"] (trained dictionary + projection results; only written
        when the dictionary has trainable columns)
    :param force_retrain: ignore existing checkpoints and retrain
    """
    cfg = _merge_config(config)
    set_seed(cfg["seed"])
    kind = dataset["kind"]
    T = float(dataset["T"])

    def flat(u):
        return u.reshape(u.shape[0], -1)

    u0 = torch.as_tensor(flat(dataset["u0"]), dtype=torch.float32, device=device)
    uT = torch.as_tensor(flat(dataset["uT"]), dtype=torch.float32, device=device)
    u2T = None
    if dataset.get("u2T") is not None:
        u2T = torch.as_tensor(flat(dataset["u2T"]), dtype=torch.float32, device=device)
    grid = torch.as_tensor(dataset["fno_grid"], dtype=torch.float32, device=device)
    coords = torch.as_tensor(dataset["basis_coords"], dtype=torch.float32, device=device)
    weights = torch.as_tensor(dataset["weights"], dtype=torch.float32, device=device)

    operator_kind = str(operator or cfg.get("operator", "fno")).lower()
    model = build_operator(operator_kind, dataset, cfg["model_kwargs"]).to(device)
    physics = _pino_generator(dataset) if operator_kind == "pino" else None

    op_path = bso_path = op_progress_path = None
    if checkpoint_dir is not None:
        ckdir = Path(checkpoint_dir)
        ckdir.mkdir(parents=True, exist_ok=True)
        op_path = ckdir / f"{operator_kind}.pt"
        bso_path = ckdir / cfg.get("bso_file", "bso.pt")
        if kind == "grid2d_multichannel":
            op_progress_path = ckdir / f".{operator_kind}.training.pt"
            if force_retrain:
                op_progress_path.unlink(missing_ok=True)
                op_progress_path.with_suffix(
                    op_progress_path.suffix + ".tmp").unlink(missing_ok=True)
    op_meta = dict(operator=operator_kind, model_kwargs=dict(cfg["model_kwargs"]),
                   system=dataset.get("system"), n_points=int(u0.shape[1]))
    if kind == "grid2d_multichannel":
        op_meta.update(
            state_shape=tuple(int(v) for v in np.asarray(dataset["u0"]).shape[1:]),
            channels=tuple(dataset.get("meta", {}).get("channel_names", ())),
            data_fingerprint=dataset.get("meta", {}).get("data_fingerprint"),
            training_fingerprint=dataset.get("meta", {}).get("training_fingerprint"),
            seed=int(cfg["seed"]),
            train_kwargs=dict(cfg["train_kwargs"]),
        )

    train_kwargs = dict(cfg["train_kwargs"])
    loss_w = weights if train_kwargs.pop("weighted_loss", False) else None

    op_loaded = False
    if op_path is not None and op_path.exists() and not force_retrain:
        payload = torch.load(op_path, map_location=device, weights_only=False)
        if payload.get("meta") == op_meta:
            model.load_state_dict(payload["state"], strict=False)
            model = model.to(device).eval()
            val0 = _batched_operator_rel_l2(
                model, u0, uT, grid, kind,
                bs=int(train_kwargs.get("bs", 32)),
                loss_weights=loss_w,
                max_duty_cycle=float(
                    train_kwargs.get("max_duty_cycle", 1.0)))
            train_info = dict(best_val=val0, history=[])
            op_loaded = True
            if op_progress_path is not None:
                op_progress_path.unlink(missing_ok=True)
            print(f"[pipeline] phase A: loaded operator checkpoint {op_path} "
                  f"(val rel-L2 = {val0:.4e})")
        else:
            print("[pipeline] operator checkpoint exists but its configuration "
                  "differs; retraining")
    if not op_loaded:
        print(f"[pipeline] phase A: training operator '{operator_kind}' "
              f"({model.__class__.__name__}) on {u0.shape[0]} samples ...")
        t0 = time.time()
        train_info = train_operator(model, u0, uT, u2T, grid, kind,
                                    seed=cfg["seed"], physics=physics,
                                    physics_weight=cfg.get("physics_weight", 1.0),
                                     horizon=T, loss_weights=loss_w,
                                     progress_path=op_progress_path,
                                     progress_metadata=op_meta,
                                     training_stop_path=training_stop_path,
                                     **train_kwargs)
        print(f"[pipeline] phase A done in {time.time() - t0:.1f}s, "
              f"best val rel-L2 = {train_info['best_val']:.4e}")
        if op_path is not None:
            temporary_op_path = op_path.with_suffix(op_path.suffix + ".tmp")
            torch.save(
                {
                    "state": model.state_dict(),
                    "meta": op_meta,
                    "best_val": train_info.get("best_val"),
                    "completed_epochs": train_info.get("completed_epochs"),
                    "stopped_early": train_info.get("stopped_early", False),
                },
                temporary_op_path,
            )
            temporary_op_path.replace(op_path)
            if op_progress_path is not None:
                op_progress_path.unlink(missing_ok=True)
            print(f"[pipeline] saved operator checkpoint to {op_path}")
    model_pure = copy.deepcopy(model).eval()

    trainer_cls = _load_trainer_class(solver_file)
    trainer = trainer_cls(reg_alpha=cfg["reg_alpha"])
    fixed = cfg.get("fixed_basis")
    if fixed is not None:
        fixed = torch.as_tensor(np.asarray(fixed), dtype=torch.float32, device=device)
        if fixed.ndim == 1:
            fixed = fixed[:, None]

    defl_cfg = cfg.get("deflation")
    defl_cols = None
    defl_log = []
    defl_mus = []
    n_user_fixed = 0 if fixed is None else fixed.shape[1]
    if defl_cfg is not None:
        if bso_path is not None and bso_path.exists() and not force_retrain:
            peek = torch.load(bso_path, map_location=device, weights_only=False)
            if (peek.get("meta", {}).get("deflation") == dict(defl_cfg)
                    and peek.get("deflation_columns") is not None):
                defl_cols = torch.as_tensor(peek["deflation_columns"],
                                            dtype=torch.float32, device=device)
                defl_mus = list(peek.get("deflation_mus", []))
                print(f"[pipeline] deflation: restored {defl_cols.shape[1]} "
                      f"accepted stage columns from {bso_path}")
        if defl_cols is None:
            print(f"[pipeline] deflation: running "
                  f"{defl_cfg.get('n_stages', 5)} stages ...")
            t0 = time.time()
            L = build_data_operator(u0.detach().cpu().numpy(),
                                    uT.detach().cpu().numpy(),
                                    dataset["weights"])
            defl_cols, defl_log = deflation_stages(
                model, grid, coords, weights, kind, L,
                basis_kwargs=cfg["basis_kwargs"],
                **{k: v for k, v in defl_cfg.items() if k != "pin_stages"})
            defl_mus = [e["mu"] for e in defl_log if e["accepted"]]
            n_acc = 0 if defl_cols is None else defl_cols.shape[1]
            print(f"[pipeline] deflation stages done in {time.time() - t0:.1f}s "
                  f"({n_acc} columns accepted)")
            set_seed(cfg["seed"])
        if defl_cols is not None:
            fixed = defl_cols if fixed is None else torch.cat([fixed, defl_cols], dim=1)

    n_fixed = 0 if fixed is None else fixed.shape[1]
    n_learn = cfg["n_basis"] - 1 - n_fixed
    if cfg.get("basis_features") is not None:
        basis_net = FeatureMixBasis(cfg["basis_features"], n_learn).to(device)
        basis_desc = "feature-mix"
    else:
        basis_net = BasisNet(coord_dim=coords.shape[1], n_learn=n_learn,
                             envelope=cfg.get("basis_envelope"),
                             **cfg["basis_kwargs"]).to(device)
        basis_desc = "mlp"
    training_projection_steps = int(
        cfg.get("training_projection_steps") or cfg["projection_steps"]
    )
    projection_operator = str(
        cfg.get("projection_operator", "joint")
    ).lower()
    if projection_operator not in {"joint", "phase_a"}:
        raise ValueError(
            "projection_operator must be 'joint' or 'phase_a', "
            f"got {projection_operator!r}"
        )
    bso_meta = dict(n_basis=int(cfg["n_basis"]), n_fixed=int(n_fixed),
                    solver=trainer.name, basis=basis_desc,
                    distill=bool(cfg.get("distill_targets") is not None))
    if kind == "grid2d":
        bso_meta.update(
            system=dataset.get("system"),
            seed=int(cfg["seed"]),
            basis_kwargs=dict(cfg["basis_kwargs"]),
            joint_kwargs=dict(cfg["joint_kwargs"]),
            reg_alpha=float(cfg["reg_alpha"]),
            projection_steps=int(cfg["projection_steps"]),
            training_projection_steps=training_projection_steps,
            projection_operator=projection_operator,
            symmetrize_K=bool(cfg.get("symmetrize_K", False)),
            stabilize_max_radius=cfg.get("stabilize_max_radius"),
        )
    if kind == "grid2d_multichannel":
        bso_meta.update(
            system=dataset.get("system"),
            state_shape=tuple(int(v) for v in np.asarray(dataset["u0"]).shape[1:]),
            channels=tuple(dataset.get("meta", {}).get("channel_names", ())),
            data_fingerprint=dataset.get("meta", {}).get("data_fingerprint"),
            training_fingerprint=dataset.get("meta", {}).get("training_fingerprint"),
            operator=operator_kind,
            model_kwargs=dict(cfg["model_kwargs"]),
            train_kwargs=dict(cfg["train_kwargs"]),
            seed=int(cfg["seed"]),
            basis_kwargs=dict(cfg["basis_kwargs"]),
            joint_kwargs=dict(cfg["joint_kwargs"]),
            reg_alpha=float(cfg["reg_alpha"]),
            stabilize_max_radius=cfg.get("stabilize_max_radius"),
        )
    if defl_cfg is not None:
        bso_meta["deflation"] = dict(defl_cfg)

    basis_loaded = False
    if (bso_path is not None and bso_path.exists() and not force_retrain
            and n_learn > 0):
        payload = torch.load(bso_path, map_location=device, weights_only=False)
        if payload.get("meta") == bso_meta:
            basis_net.load_state_dict(payload["basis_net"])
            basis_net = basis_net.to(device)
            if payload.get("joint_model") is not None:
                # joint training refined the operator too; restore it so the
                # recomputed spectrum matches the stored one exactly
                model.load_state_dict(payload["joint_model"], strict=False)
                model = model.to(device)
                print("[pipeline] phase B: restored jointly refined operator state")
            joint_history = []
            basis_loaded = True
            print(f"[pipeline] phase B: loaded dictionary checkpoint {bso_path}")
        else:
            print("[pipeline] bso checkpoint exists but its configuration "
                  "differs; retraining the dictionary")
    if not basis_loaded:
        t0 = time.time()
        if cfg.get("distill_targets") is not None:
            print("[pipeline] phase B: distilling the dictionary onto target columns ...")
            joint_history = distill_basis(basis_net, coords, weights,
                                          cfg["distill_targets"],
                                          **cfg.get("distill_kwargs", {}))
        else:
            print(f"[pipeline] phase B: joint training with solver '{trainer.name}' ({solver_file}) ...")
            joint_history = joint_train(model, basis_net, trainer, u0, uT, u2T, grid, coords,
                                        weights, kind,
                                        projection_steps=training_projection_steps,
                                        fixed=fixed, **cfg["joint_kwargs"])
        print(f"[pipeline] phase B done in {time.time() - t0:.1f}s")

    model.eval()
    model_pure.eval()
    basis_net.eval()
    projection_model = (
        model_pure if projection_operator == "phase_a" else model
    )
    proj_steps = int(cfg.get("projection_steps", 1))
    validation_bs = int(train_kwargs.get("bs", 32))
    validation_duty = float(train_kwargs.get("max_duty_cycle", 1.0))
    with torch.no_grad():
        pacing_started = _pacing_start(validation_duty)
        psi_t = build_basis(basis_net, coords, weights, fixed)
        evolved_t = apply_operator_n(
            projection_model, psi_t, grid, kind, proj_steps
        )
        _pace_cuda(pacing_started, validation_duty)
    val_rel = _batched_operator_rel_l2(
        projection_model, u0, uT, grid, kind, bs=validation_bs,
        max_duty_cycle=validation_duty)
    psi = psi_t.detach().cpu().numpy().astype(np.float64)
    evolved = evolved_t.detach().cpu().numpy().astype(np.float64)
    pin_eigs = None
    if defl_cfg is not None and defl_cfg.get("pin_stages") and defl_mus:
        pin_eigs = [(1 + n_user_fixed + i, mu_i)
                    for i, mu_i in enumerate(defl_mus)]
    K, mu, lam, efuncs, r_eff, cond_G = _final_spectral_analysis(
        psi, evolved, dataset["weights"], cfg["reg_alpha"], proj_steps * T,
        max_radius=cfg.get("stabilize_max_radius"),
        pin_constant=cfg.get("pin_constant", True),
        symmetrize=cfg.get("symmetrize_K", False),
        pin_eigs=pin_eigs)

    if bso_path is not None and n_learn > 0 and not basis_loaded:
        payload = {"basis_net": basis_net.state_dict(), "meta": bso_meta,
                   "K": K, "psi": psi, "mu": mu, "lam": lam,
                   "eigenfunctions": efuncs,
                   "weights": np.asarray(dataset["weights"]),
                   "grid": np.asarray(dataset["fno_grid"]), "T": T}
        if defl_cols is not None:
            payload["deflation_columns"] = defl_cols.detach().cpu().numpy()
            payload["deflation_mus"] = list(defl_mus)
        if float(cfg["joint_kwargs"].get("model_lr", 0.0)) > 0:
            payload["joint_model"] = model.state_dict()
        torch.save(payload, bso_path)
        print(f"[pipeline] saved dictionary + projection checkpoint to {bso_path}")

    result = BSOResult(
        system=dataset.get("system", "unknown"),
        kind=kind,
        T=T,
        solver_name=trainer.name,
        operator=projection_model,
        operator_pure=model_pure,
        basis_net=basis_net,
        psi=psi,
        evolved=evolved,
        K=K,
        mu=mu,
        lam=lam,
        eigenfunctions=efuncs,
        r_eff=r_eff,
        cond_G=cond_G,
        weights=np.asarray(dataset["weights"], dtype=np.float64),
        grid=np.asarray(dataset["fno_grid"], dtype=np.float64),
        operator_val_rel_l2=val_rel,
        history=dict(operator=train_info["history"], joint=joint_history,
                     deflation=defl_log),
        operator_joint=model,
    )
    print("[pipeline] " + result.summary().replace("\n", "\n[pipeline] "))
    return result


# ---------------------------------------------------------------------------
# Rollout helpers (shared by the experiment scripts)
# ---------------------------------------------------------------------------

def spectral_rollout(result: BSOResult, u0: np.ndarray, n_steps: int,
                     n_modes: int | None = None,
                     drop_at_radius: float | None = None) -> np.ndarray:
    """Eigendecomposition rollout: u_n = Psi^T V diag(mu^n) V^{-1} Psi W u0.

    The prediction is a linear combination of BSO eigenpairs. With
    n_modes set, only the n_modes slowest eigenpairs (largest |mu|) are kept;
    the discarded fast tail is typically dominated by spectral pollution.
    Accepts a single field (p,) or a batch (n_cases, p); returns
    (n_steps+1, ...) matching the input layout.
    """
    psi, w = result.psi, result.weights
    single = np.asarray(u0).ndim == 1
    U0 = np.atleast_2d(np.asarray(u0, dtype=np.float64))       # (n_cases, p)
    C0 = psi @ (w[:, None] * U0.T)                             # (n_b, n_cases)
    mu, V = np.linalg.eig(result.K)
    order = np.argsort(-np.abs(mu))
    mu, V = mu[order], V[:, order]
    if drop_at_radius is not None:
        # modes sitting exactly on the stabilization radius were clipped for
        # violating semigroup non-expansiveness (|mu| > 1): treat them as
        # spectral pollution and exclude them from the reconstruction
        keep = np.abs(np.abs(mu) - float(drop_at_radius)) > 1e-6
        mu, V = mu[keep], V[:, keep]
    r = len(mu) if n_modes is None else max(1, int(n_modes))
    Vr, mu_r = V[:, :r], mu[:r]
    # least-squares fit of the initial coefficients in the KEPT eigenspace
    # (plain truncation of the full expansion would corrupt the t=0 state,
    # since the discarded fast/pollution modes carry part of u0)
    A0 = np.linalg.lstsq(Vr, C0.astype(complex), rcond=None)[0]
    traj = []
    for n in range(n_steps + 1):
        Cn = Vr @ ((mu_r ** n)[:, None] * A0)
        traj.append((psi.T @ Cn).real.T)
    traj = np.asarray(traj)
    return traj[:, 0] if single else traj


def bso_rollout(result: BSOResult, u0: np.ndarray, n_steps: int) -> np.ndarray:
    """Linear rollout in dictionary coefficients: c <- K c, u = Psi^T c."""
    psi = result.psi
    w = result.weights
    c = psi @ (w * np.asarray(u0, dtype=np.float64).reshape(-1))
    traj = [psi.T @ c]
    for _ in range(n_steps):
        c = result.K @ c
        traj.append(psi.T @ c)
    return np.asarray(traj)


def operator_rollout(result: BSOResult, u0: np.ndarray, n_steps: int,
                     operator: nn.Module | None = None) -> np.ndarray:
    """Autoregressive rollout of a trained neural operator.

    By default uses the joint-trained operator; pass result.operator_pure to
    roll out the phase-A operator instead.
    """
    op = result.operator if operator is None else operator
    grid = torch.as_tensor(result.grid, dtype=torch.float32, device=device)
    u = torch.as_tensor(np.asarray(u0, dtype=np.float64).reshape(1, -1),
                        dtype=torch.float32, device=device)
    traj = [u0.reshape(-1).copy()]
    op.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            u = apply_operator(op, u, grid, result.kind)
            traj.append(u.squeeze(0).detach().cpu().numpy().astype(np.float64))
    return np.asarray(traj)
