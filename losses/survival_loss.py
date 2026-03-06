"""
losses/survival_loss.py
=======================
Negative log-likelihood loss for EGMDM survival head.

Handles right-censored data:
  - uncensored (event=1) : maximise log p(t)
  - censored   (event=0) : maximise log S(t)  =  log(1 - CDF(t))
"""

import torch
import torch.nn as nn

from models.egmdm import EGMDMHead


class EGMDMLoss(nn.Module):
    """
    Parameters
    ----------
    lambda_div : weight for expert-diversity regularisation (L_div)
    lambda_ent : weight for gate-entropy regularisation    (L_ent)
    """

    def __init__(self, lambda_div: float = 0.1, lambda_ent: float = 0.01):
        super().__init__()
        self.lambda_div = lambda_div
        self.lambda_ent = lambda_ent

    def forward(
        self,
        model:      EGMDMHead,
        params:     dict,           # distribution params from EGMDMHead
        reg_losses: dict,           # regularisation terms from EGMDMHead
        rfs:        torch.Tensor,   # (B,) relapse-free survival in YEARS
        event:      torch.Tensor,   # (B,) float — 1 = event, 0 = censored
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        total_loss, nll
        """
        t = rfs.clamp(min=1e-4, max=20.0)

        log_pdf  = model.log_prob(params, t)

        cdf      = model.cdf(params, t).clamp(1e-8, 1.0 - 1e-6)
        log_surv = torch.log(1.0 - cdf)

        nll        = -torch.mean(event * log_pdf + (1.0 - event) * log_surv)
        total_loss = nll

        if "L_div" in reg_losses:
            total_loss = total_loss + self.lambda_div * reg_losses["L_div"]
        if "L_ent" in reg_losses:
            total_loss = total_loss + self.lambda_ent * reg_losses["L_ent"]

        return total_loss, nll