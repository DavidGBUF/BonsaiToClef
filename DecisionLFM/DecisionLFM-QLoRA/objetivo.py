"""Entropia cruzada + Brier multiclasses, sem mudar os alvos ou a inferência."""

import torch
from torch.nn import functional as F


def decision_loss(logits, targets, brier_weight=1.0, reduction="mean"):
    """Brier soma as classes válidas; a redução final é sobre os exemplos."""
    weight = torch.as_tensor(brier_weight, device=logits.device, dtype=torch.float32)
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("brier_weight deve ser finito e >= 0")
    if weight.ndim and weight.shape != targets.shape:
        raise ValueError("Use um peso escalar ou um peso por exemplo")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction deve ser none, sum ou mean")
    logits = logits.float()
    if weight.ndim == 0 and weight == 0:
        return F.cross_entropy(logits, targets, reduction=reduction)
    ce = F.cross_entropy(logits, targets, reduction="none")
    probabilities = logits.softmax(-1)
    target = F.one_hot(targets, logits.shape[-1]).to(probabilities.dtype)
    brier = (probabilities - target).square().sum(-1)
    loss = ce + weight * brier
    return loss if reduction == "none" else getattr(loss, reduction)()

