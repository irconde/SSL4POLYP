import json

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset

from ssl4polyp.classification.metrics import thresholds


def test_compute_youden_threshold_matches_manual_roc():
    logits = torch.tensor(
        [
            [0.0, -1.0],
            [0.0, 0.1],
            [0.0, 1.2],
            [0.0, 2.5],
            [0.0, -2.0],
        ]
    )
    targets = torch.tensor([0, 0, 1, 1, 0])

    tau = thresholds.compute_youden_j_threshold(logits, targets)

    scores = torch.softmax(logits, dim=1)[:, 1].tolist()
    targets_np = targets.tolist()
    preds = [1 if score >= tau else 0 for score in scores]

    tp = sum(int(p == 1 and t == 1) for p, t in zip(preds, targets_np))
    tn = sum(int(p == 0 and t == 0) for p, t in zip(preds, targets_np))
    fp = sum(int(p == 1 and t == 0) for p, t in zip(preds, targets_np))
    fn = sum(int(p == 0 and t == 1) for p, t in zip(preds, targets_np))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    youden = sensitivity + specificity - 1

    brute_force_scores = []
    for step in range(21):
        candidate = step / 20
        brute_preds = [1 if score >= candidate else 0 for score in scores]
        tp_b = sum(int(p == 1 and t == 1) for p, t in zip(brute_preds, targets_np))
        tn_b = sum(int(p == 0 and t == 0) for p, t in zip(brute_preds, targets_np))
        fp_b = sum(int(p == 1 and t == 0) for p, t in zip(brute_preds, targets_np))
        fn_b = sum(int(p == 0 and t == 1) for p, t in zip(brute_preds, targets_np))
        sens = tp_b / (tp_b + fn_b) if (tp_b + fn_b) else 0.0
        spec = tn_b / (tn_b + fp_b) if (tn_b + fp_b) else 0.0
        brute_force_scores.append(sens + spec - 1)

    assert youden >= max(brute_force_scores) - 1e-6


def test_threshold_serialisation_roundtrip(tmp_path):
    mapping = {"sun_val_youden": 0.42, "polypgen_val_youden": 0.55}
    out_path = tmp_path / "thresholds.json"
    thresholds.save_thresholds(out_path, mapping)
    with out_path.open() as handle:
        payload = json.load(handle)
    assert payload["thresholds"]["sun_val_youden"] == pytest.approx(0.42, rel=1e-6)

    loaded = thresholds.load_thresholds(out_path)
    for key, value in mapping.items():
        assert loaded[key] == pytest.approx(value, rel=1e-6)


def test_format_and_resolve_threshold_key():
    key = thresholds.format_threshold_key("SUN", "Val", "Youden")
    assert key == "sun_val_youden"
    mapping = {key: 0.33}
    assert thresholds.resolve_threshold(mapping, key) == pytest.approx(0.33, rel=1e-6)
    assert thresholds.resolve_threshold(mapping, "missing") is None


class _ThresholdDataset(Dataset):
    def __init__(self, logits: torch.Tensor, labels: torch.Tensor):
        self.logits = logits
        self.labels = labels

    def __len__(self) -> int:  # pragma: no cover - simple data container
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.logits[idx], self.labels[idx], {"idx": int(idx)}


class _IdentityModel(torch.nn.Module):
    def forward(self, inputs):  # pragma: no cover - exercised indirectly
        return inputs


def test_compute_threshold_from_loader_matches_direct():
    logits = torch.tensor(
        [
            [1.2, -0.4],
            [0.3, 0.1],
            [-0.2, 0.8],
            [0.1, -1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    dataset = _ThresholdDataset(logits, labels)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = _IdentityModel()
    model.train()

    tau_loader = thresholds.compute_threshold_from_loader(
        model, loader, torch.device("cpu")
    )
    tau_direct = thresholds.compute_youden_j_threshold(logits, labels)

    assert tau_loader == pytest.approx(tau_direct, rel=1e-6)
    assert model.training, "Model training mode should be restored after threshold computation"
