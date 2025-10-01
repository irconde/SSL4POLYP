from __future__ import annotations

import warnings

import pytest
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score

from ssl4polyp.classification.metrics.performance import meanAUPRC, meanAUROC, meanBalancedAccuracy


def test_balanced_accuracy_binary_with_tau():
    probs = torch.tensor(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.65, 0.35],
            [0.45, 0.55],
        ]
    )
    targets = torch.tensor([0, 1, 0, 1])
    metric = meanBalancedAccuracy(n_class=2)
    score = metric(probs, targets, tau=0.6).item()

    # Apply threshold manually to verify the helper respects ``tau``.
    manual_preds = (probs[:, 1] >= 0.6).long()
    expected = balanced_accuracy_score(targets.numpy(), manual_preds.numpy())

    assert score == pytest.approx(expected)


def test_balanced_accuracy_accepts_logits():
    logits = torch.tensor(
        [
            [2.0, 1.0],
            [0.0, 3.0],
            [3.0, 0.0],
            [1.0, 2.0],
        ]
    )
    targets = torch.tensor([0, 1, 0, 1])
    metric = meanBalancedAccuracy(n_class=2)
    score = metric(logits, targets).item()

    probs = torch.softmax(logits, dim=1)
    expected = balanced_accuracy_score(targets.numpy(), probs.argmax(dim=1).numpy())

    assert score == pytest.approx(expected)


def test_balanced_accuracy_multiclass_probabilities():
    probs = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.05, 0.1, 0.85],
        ]
    )
    targets = torch.tensor([0, 1, 2, 2])
    metric = meanBalancedAccuracy(n_class=3)
    score = metric(probs, targets).item()

    expected = balanced_accuracy_score(targets.numpy(), probs.argmax(dim=1).numpy())

    assert score == pytest.approx(expected)


def test_mauprc_binary_logits():
    logits = torch.tensor(
        [
            [2.0, 1.0],
            [1.0, 2.0],
            [3.0, 0.5],
            [0.2, 1.8],
        ]
    )
    targets = torch.tensor([0, 1, 0, 1])
    metric = meanAUPRC(n_class=2)
    score = metric(logits, targets).item()

    probs = torch.softmax(logits, dim=1)[:, 1]
    expected = average_precision_score(targets.numpy(), probs.numpy())

    assert score == pytest.approx(expected)


def test_mauprc_multiclass_probabilities():
    probs = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.2, 0.2],
            [0.15, 0.7, 0.15],
        ]
    )
    targets = torch.tensor([0, 1, 2, 0, 1])
    metric = meanAUPRC(n_class=3)
    score = metric(probs, targets).item()

    one_hot = torch.nn.functional.one_hot(targets, num_classes=3).numpy()
    expected = average_precision_score(one_hot, probs.numpy(), average="macro")

    assert score == pytest.approx(expected)


def test_mean_auroc_binary_single_class_returns_nan():
    probs = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
        ]
    )
    targets = torch.tensor([0, 0, 0])
    metric = meanAUROC(n_class=2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = metric(probs, targets)

    assert caught, "Expected a warning when AUROC is undefined"
    assert torch.isnan(score).item()


def test_mean_auroc_multiclass_single_class_returns_nan():
    probs = torch.tensor(
        [
            [0.6, 0.3, 0.1],
            [0.55, 0.25, 0.2],
            [0.7, 0.2, 0.1],
        ]
    )
    targets = torch.tensor([0, 0, 0])
    metric = meanAUROC(n_class=3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = metric(probs, targets)

    assert caught, "Expected a warning when AUROC is undefined"
    assert torch.isnan(score).item()
