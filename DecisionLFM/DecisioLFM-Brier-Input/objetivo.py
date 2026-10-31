"""Entropia cruzada + Brier multiclasses, sem mudar os alvos ou a inferência."""

import math

from torch.nn import functional as F


def decision_loss(logits, targets, brier_weight=1.0, reduction="mean"):
    """Brier soma as classes válidas; a redução final é sobre os exemplos."""
    if not math.isfinite(brier_weight) or brier_weight < 0:
        raise ValueError("brier_weight deve ser finito e >= 0")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction deve ser none, sum ou mean")
    logits = logits.float()
    if brier_weight == 0:
        return F.cross_entropy(logits, targets, reduction=reduction)
    ce = F.cross_entropy(logits, targets, reduction="none")
    probabilities = logits.softmax(-1)
    target = F.one_hot(targets, logits.shape[-1]).to(probabilities.dtype)
    brier = (probabilities - target).square().sum(-1)
    loss = ce + brier_weight * brier
    return loss if reduction == "none" else getattr(loss, reduction)()
