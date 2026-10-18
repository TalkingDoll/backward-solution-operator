"""EDMD dictionary solver (default backend for edmd_bso_method).

Interface contract (shared with solver_resdmd.py):

    class DictionaryTrainer:
        name : str
        compute_K(psi, evolved, weights)          -> K (torch tensor, k x k)
        prepare_target(K)                          -> opaque target for the loss
        dictionary_loss(psi, evolved, weights, target) -> scalar torch tensor

Conventions: psi and evolved are (n_basis, n_points) with rows = basis
functions; weights is (n_points,); evolved = U_tau applied to each basis row.
K is the weighted Galerkin least-squares matrix K = (G + a*I)^{-1} A with
G = Psi W Psi^T, A = Psi W (U Psi)^T, so that U Psi ~= K^T Psi.

The dictionary training loss here is the plain EDMD/invariance residual
|| U Psi - K^T Psi ||_w with K frozen.
"""

import torch


class DictionaryTrainer:
    name = "edmd"

    def __init__(self, reg_alpha=1.0e-5):
        self.reg_alpha = float(reg_alpha)

    def compute_K(self, psi, evolved, weights):
        w = weights.reshape(1, -1)
        G = (psi * w) @ psi.T
        A = (psi * w) @ evolved.T
        eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        return torch.linalg.solve(G + self.reg_alpha * eye, A)

    def prepare_target(self, K):
        return {"K": K.detach()}

    def dictionary_loss(self, psi, evolved, weights, target):
        # row-wise relative residual: every basis function must be fitted,
        # not just the few rows that dominate a global norm
        K = target["K"]
        pred = K.T @ psi
        w = weights.reshape(1, -1)
        num = torch.sum(w * (evolved - pred) ** 2, dim=1)
        den = torch.sum(w * evolved**2, dim=1) + 1.0e-12
        return torch.mean(torch.sqrt(num / den))
