"""
models/egmdm.py
===============
Ensemble of Gaussian Mixture Density Models (EGMDM) survival head.

Takes a patient embedding vector and outputs a mixture-of-Gaussians
distribution over log-transformed survival time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Numerically Stable Helpers ───────────────────────────────────────────────

def _safe_inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Stable inverse-softplus: y = log(exp(x) - 1)."""
    x = torch.clamp(x, min=1e-6, max=20.0)
    return torch.where(x > 10.0, x, x + torch.log(-torch.expm1(-x) + 1e-6))


def _logsumexp_weighted(
    a:       torch.Tensor,
    weights: torch.Tensor,
    dim:     int = -1,
) -> torch.Tensor:
    """Stable log(sum(weights * exp(a))) along `dim`."""
    a_max = torch.max(a, dim=dim, keepdim=True)[0]
    return a_max + torch.log(
        torch.sum(weights * torch.exp(a - a_max), dim=dim, keepdim=True) + 1e-8
    )


# ─── MLP building block ───────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── EGMDM Head ───────────────────────────────────────────────────────────────

class EGMDMHead(nn.Module):
    """
    Parameters
    ----------
    input_size   : dimension of the patient embedding
    hidden_size  : MLP hidden width
    E            : number of experts
    K            : number of Gaussian components per expert
    param_share  : which of ('w', 'mu', 'sigma') are shared across experts
    dropout      : dropout in MLP blocks
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int   = 256,
        E:           int   = 3,
        K:           int   = 10,
        param_share: tuple = ("sigma",),
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.E = E
        self.K = K
        self.param_share = param_share

        # Gating network
        self.gate = _MLP(input_size, hidden_size, E, dropout)

        # Expert networks — only output non-shared params
        params_per_expert = sum(
            K for name in ("w", "mu", "sigma") if name not in param_share
        )
        self.heads = nn.ModuleList(
            [_MLP(input_size, hidden_size, params_per_expert, dropout) for _ in range(E)]
        )

        # Shared parameters
        shared: dict = {}
        if "mu"    in param_share:
            shared["mu"]    = nn.Parameter(torch.linspace(-2, 2, K).unsqueeze(0))
        if "sigma" in param_share:
            shared["sigma"] = nn.Parameter(torch.ones(1, K) * 0.5)
        if "w"     in param_share:
            shared["w"]     = nn.Parameter(torch.ones(1, K))
        self.shared = nn.ParameterDict(shared)

        # Per-component linear transforms for mu and sigma
        self.layer_mu    = nn.Linear(K, K, bias=True)
        self.layer_sigma = nn.Linear(K, K, bias=True)

    def _get_params(self, h: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor]:
        B = h.size(0)
        h = F.normalize(h, dim=-1)
        G = self.gate(h).softmax(-1)   # (B, E)

        w_list, mu_list, sigma_list = [], [], []

        for e in range(self.E):
            out    = self.heads[e](h)
            offset = 0

            if "w" in self.param_share:
                w_raw = self.shared["w"].expand(B, -1)
            else:
                w_raw  = out[:, offset:offset + self.K]; offset += self.K

            if "mu" in self.param_share:
                mu_raw = self.layer_mu(self.shared["mu"]).expand(B, -1)
            else:
                mu_raw = self.layer_mu(out[:, offset:offset + self.K]); offset += self.K

            if "sigma" in self.param_share:
                sigma_raw = self.layer_sigma(self.shared["sigma"]).expand(B, -1)
            else:
                sigma_raw = self.layer_sigma(out[:, offset:offset + self.K]); offset += self.K

            w_list.append(w_raw.softmax(-1))
            mu_list.append(mu_raw.clamp(-10, 10))
            sigma_list.append(F.softplus(sigma_raw).clamp(1e-3, 5.0))

        w_stack   = torch.stack(w_list,     dim=1)   # (B, E, K)
        mu_stack  = torch.stack(mu_list,    dim=1)
        sig_stack = torch.stack(sigma_list, dim=1)

        # Flatten experts × components and renormalise weights
        w_flat  = (w_stack * G.unsqueeze(-1)).reshape(B, -1)
        w_flat  = w_flat / w_flat.sum(-1, keepdim=True).clamp_min(1e-8)
        mu_flat = mu_stack.reshape(B, -1)
        sig_flat = sig_stack.reshape(B, -1)

        params = {"w": w_flat, "mu": mu_flat, "sigma": sig_flat}
        return params, G, mu_stack

    # ── Distribution interface ────────────────────────────────────────────

    def cdf(self, params: dict, t: torch.Tensor) -> torch.Tensor:
        """CDF at time t (years). Returns (B,)."""
        y      = _safe_inverse_softplus(t).unsqueeze(-1)
        normal = torch.distributions.Normal(params["mu"], params["sigma"])
        return (params["w"] * normal.cdf(y)).sum(-1)

    def log_prob(self, params: dict, t: torch.Tensor) -> torch.Tensor:
        """Log-density at time t (years). Returns (B,)."""
        y      = _safe_inverse_softplus(t).unsqueeze(-1)
        normal = torch.distributions.Normal(params["mu"], params["sigma"])
        log_pdf_y = _logsumexp_weighted(normal.log_prob(y), params["w"]).squeeze(-1)

        t_safe   = t.clamp(min=1e-6, max=20.0)
        log_jac  = t_safe - torch.log(torch.expm1(t_safe) + 1e-8)
        return log_pdf_y + log_jac

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, h: torch.Tensor) -> tuple[dict, dict]:
        """
        Parameters
        ----------
        h : (B, input_size) patient embedding

        Returns
        -------
        params      : {'w', 'mu', 'sigma'} mixture params
        reg_losses  : {'L_div', 'L_ent'}  regularisation terms
        """
        params, G, mu_stack = self._get_params(h)

        reg_losses: dict = {}
        if self.E > 1:
            # Expert diversity: penalise when expert mean-centres are too similar.
            centers = mu_stack.mean(dim=2)              # (B, E)
            dists = []
            for i in range(self.E):
                for j in range(i + 1, self.E):
                    dists.append((centers[:, i] - centers[:, j]).pow(2).mean())
            mean_dist = torch.stack(dists).mean() if dists else torch.tensor(0.0)
            reg_losses["L_div"] = torch.exp(-mean_dist)   # ∈ [0,1]: 0=spread, 1=collapsed

        # Gate entropy: penalise gate certainty → all experts used equally.
        reg_losses["L_ent"] = -(G * (G + 1e-8).log()).sum(-1).mean()

        # Mixture weight entropy: penalise collapse of final mixture weights.
        # L_ent only regulates which expert is chosen; this regulates whether
        # each expert spreads weight across all K components or collapses to 1.
        # val/mixture_entropy of 0.2 (vs ideal log(K)≈1.6 for K=5) means the
        # model predicts the same narrow distribution for every patient, losing
        # discriminative power and causing C-index to fall after early epochs.
        w_flat = params["w"]                                    # (B, E*K)
        mix_ent = -(w_flat * (w_flat + 1e-8).log()).sum(-1).mean()
        reg_losses["L_mix"] = -mix_ent   # minimising -entropy = maximising entropy

        return params, reg_losses