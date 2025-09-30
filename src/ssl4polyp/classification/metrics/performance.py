from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, average_precision_score, roc_auc_score


_PROB_ATOL = 1e-6
_PROB_RTOL = 1e-4


def _looks_like_probability_vector(tensor: torch.Tensor) -> bool:
    """Return ``True`` if ``tensor`` appears to contain probabilities."""

    if tensor.numel() == 0:
        return True
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    return min_val >= -_PROB_ATOL and max_val <= 1.0 + _PROB_ATOL


def _looks_like_probability_matrix(tensor: torch.Tensor) -> bool:
    """Return ``True`` if rows of ``tensor`` resemble probability distributions."""

    if tensor.numel() == 0:
        return True
    if not _looks_like_probability_vector(tensor):
        return False
    row_sums = tensor.sum(dim=1)
    ones = torch.ones_like(row_sums)
    return torch.allclose(row_sums, ones, atol=1e-3, rtol=_PROB_RTOL)


def _as_binary_positive_scores(tensor: torch.Tensor) -> torch.Tensor:
    """Return probabilities for the positive class from binary predictions."""

    if tensor.ndim == 1:
        if tensor.dtype.is_floating_point:
            if _looks_like_probability_vector(tensor):
                return tensor
            return torch.sigmoid(tensor)
        return tensor.to(dtype=torch.float32)
    if tensor.ndim == 2:
        if tensor.size(1) == 1:
            return _as_binary_positive_scores(tensor.squeeze(1))
        if tensor.size(1) != 2:
            raise ValueError("Binary probability extraction expects tensors with shape (N,), (N, 1) or (N, 2)")
        if tensor.dtype.is_floating_point and _looks_like_probability_matrix(tensor):
            probs = tensor
        else:
            probs = torch.softmax(tensor.to(dtype=torch.float32), dim=1)
        return probs[:, 1]
    raise ValueError("Binary probability extraction expects tensors with 1 or 2 dimensions")


def _as_class_probabilities(tensor: torch.Tensor, n_class: int) -> torch.Tensor:
    """Return class probabilities for multi-class predictions."""

    if tensor.ndim != 2 or tensor.size(1) != n_class:
        raise ValueError(
            f"Expected tensor with shape (N, {n_class}) for multi-class probabilities; got {tuple(tensor.shape)}"
        )
    if tensor.dtype.is_floating_point and _looks_like_probability_matrix(tensor):
        return tensor
    return torch.softmax(tensor.to(dtype=torch.float32), dim=1)


def _as_label_predictions(
    tensor: torch.Tensor,
    n_class: int,
    tau: Optional[float] = None,
) -> torch.Tensor:
    """Convert ``tensor`` containing logits/probabilities into discrete predictions."""

    if tensor.ndim == 1:
        if tensor.dtype.is_floating_point and n_class == 2:
            probs = tensor if _looks_like_probability_vector(tensor) else torch.sigmoid(tensor)
            threshold = 0.5 if tau is None else tau
            return (probs >= threshold).to(dtype=torch.long)
        if tensor.dtype.is_floating_point and n_class != 2:
            raise ValueError("1D probability tensors are only supported for binary problems")
        return tensor.to(dtype=torch.long)
    if tensor.ndim == 2:
        if tensor.size(1) == 1:
            return _as_label_predictions(tensor.squeeze(1), n_class, tau)
        if n_class == 2:
            probs = _as_binary_positive_scores(tensor)
            threshold = 0.5 if tau is None else tau
            return (probs >= threshold).to(dtype=torch.long)
        probs = _as_class_probabilities(tensor, n_class)
        return torch.argmax(probs, dim=1)
    raise ValueError("Prediction tensors must be 1D or 2D")


class meanF1Score(nn.Module):
    def __init__(self, n_class, smooth=1e-8):
        super(meanF1Score, self).__init__()
        self.n_class = n_class
        self.smooth = smooth

    def forward(self, preds, targets, tau: Optional[float] = None):
        labels = _as_label_predictions(preds.detach(), self.n_class, tau)
        score = 0
        for i in range(self.n_class):
            m1 = labels == i
            m2 = targets == i
            intersection = m1 * m2

            score += (
                2.0
                * (intersection.sum() + self.smooth)
                / (m1.sum() + m2.sum() + self.smooth)
            )
        return score / self.n_class


class meanPrecision(nn.Module):
    def __init__(self, n_class, smooth=1e-8):
        super(meanPrecision, self).__init__()
        self.n_class = n_class
        self.smooth = smooth

    def forward(self, preds, targets, tau: Optional[float] = None):
        labels = _as_label_predictions(preds.detach(), self.n_class, tau)
        score = 0
        for i in range(self.n_class):
            m1 = labels == i
            m2 = targets == i
            intersection = m1 * m2

            score += (intersection.sum() + self.smooth) / (m1.sum() + self.smooth)
        return score / self.n_class


class meanRecall(nn.Module):
    def __init__(self, n_class, smooth=1e-8):
        super(meanRecall, self).__init__()
        self.n_class = n_class
        self.smooth = smooth

    def forward(self, preds, targets, tau: Optional[float] = None):
        labels = _as_label_predictions(preds.detach(), self.n_class, tau)
        score = 0
        for i in range(self.n_class):
            m1 = labels == i
            m2 = targets == i
            intersection = m1 * m2

            score += (intersection.sum() + self.smooth) / (m2.sum() + self.smooth)
        return score / self.n_class


class meanAUROC(nn.Module):
    """Compute macro-averaged AUROC for multi-class predictions."""

    def __init__(self, n_class):
        super(meanAUROC, self).__init__()
        self.n_class = n_class

    def forward(self, preds, targets):
        """Compute AUROC using one-vs-rest averaging.

        Args:
            preds: Tensor of shape (N, n_class) with class probabilities or
                logits for each sample.
            targets: Tensor of shape (N,) with integer class labels.

        Returns:
            torch.Tensor: scalar tensor containing the macro AUROC.
        """

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        if self.n_class == 2:
            score = roc_auc_score(targets_np, preds_np[:, 1])
        else:
            score = roc_auc_score(
                targets_np, preds_np, multi_class="ovr", average="macro"
            )
        return torch.tensor(score)


class meanBalancedAccuracy(nn.Module):
    """Compute balanced accuracy across classes."""

    def __init__(self, n_class: int):
        super().__init__()
        self.n_class = n_class

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, *, tau: Optional[float] = None):
        labels = _as_label_predictions(preds.detach(), self.n_class, tau)
        labels_np = labels.cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        score = balanced_accuracy_score(targets_np, labels_np)
        return torch.tensor(score, dtype=torch.float32)


class meanAUPRC(nn.Module):
    """Compute macro-averaged area under the precision-recall curve."""

    def __init__(self, n_class: int):
        super().__init__()
        self.n_class = n_class

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_np = targets.detach().cpu().numpy()
        if self.n_class == 2:
            scores = _as_binary_positive_scores(preds.detach())
            score = average_precision_score(targets_np, scores.cpu().numpy())
            return torch.tensor(score, dtype=torch.float32)
        probs = _as_class_probabilities(preds.detach(), self.n_class)
        one_hot = F.one_hot(targets.to(dtype=torch.long), num_classes=self.n_class)
        score = average_precision_score(
            one_hot.cpu().numpy(), probs.cpu().numpy(), average="macro"
        )
        return torch.tensor(score, dtype=torch.float32)

