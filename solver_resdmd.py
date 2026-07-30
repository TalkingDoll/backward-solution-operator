"""ResDMD-style dictionary solver (drop-in alternative to solver_edmd).

Same interface as solver_edmd.DictionaryTrainer; pass
solver_file="solver_resdmd.py" to edmd_bso_method.run_bso_pipeline
to switch.

K is still the EDMD least-squares matrix (identical to solver_edmd). The
difference is the dictionary training loss: an eigenpair (spectral) residual

    || V^T (U Psi) - Lambda V^T Psi ||_w

where (Lambda, V) are the eigenvalues / right eigenvectors of the current K
(U Psi ~= K^T Psi and K c = mu c give U (c^T Psi) ~= mu (c^T Psi), so each row
is one eigenpair residual). V and Lambda are frozen during dictionary training
and refreshed at every outer epoch via prepare_target().
"""

import torch


class DictionaryTrainer:
    name = "resdmd"

    def __init__(self, reg_alpha=1.0e-5):
        self.reg_alpha = float(reg_alpha)

    def compute_K(self, psi, evolved, weights):
        w = weights.reshape(1, -1)
        G = (psi * w) @ psi.T
        A = (psi * w) @ evolved.T
        eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        return torch.linalg.solve(G + self.reg_alpha * eye, A)

    def prepare_target(self, K):
        lam, V = torch.linalg.eig(K.detach())
        return {"V": V.detach(), "lam": lam.detach()}

    def dictionary_loss(self, psi, evolved, weights, target):
        V, lam = target["V"], target["lam"]
        # .to(...) keeps autograd intact (torch.tensor() would detach)
        psi_c = psi.to(V.dtype)
        evolved_c = evolved.to(V.dtype)
        resid = V.T @ evolved_c - lam.reshape(-1, 1) * (V.T @ psi_c)
        target_norm = lam.reshape(-1, 1) * (V.T @ psi_c)
        w = weights.reshape(1, -1)
        # row-wise: one relative residual per eigenpair
        num = torch.sum(w * resid.abs() ** 2, dim=1)
        den = torch.sum(w * target_norm.abs() ** 2, dim=1) + 1.0e-12
        return torch.mean(torch.sqrt(num / den))
