import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class meanF1Score(nn.Module):
    def __init__(self, n_class, smooth=1e-8):
        super(meanF1Score, self).__init__()
        self.n_class = n_class
        self.smooth = smooth

    def forward(self, preds, targets):
        score = 0
        for i in range(self.n_class):
            m1 = preds == i
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

    def forward(self, preds, targets):
        score = 0
        for i in range(self.n_class):
            m1 = preds == i
            m2 = targets == i
            intersection = m1 * m2

            score += (intersection.sum() + self.smooth) / (m1.sum() + self.smooth)
        return score / self.n_class


class meanRecall(nn.Module):
    def __init__(self, n_class, smooth=1e-8):
        super(meanRecall, self).__init__()
        self.n_class = n_class
        self.smooth = smooth

    def forward(self, preds, targets):
        score = 0
        for i in range(self.n_class):
            m1 = preds == i
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

