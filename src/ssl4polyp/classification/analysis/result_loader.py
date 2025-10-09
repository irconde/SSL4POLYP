from __future__ import annotations

import hashlib
import hashlib
import math
import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from reporting.threshold_specs import THRESHOLD_SPECS

import numpy as np

__all__ = [
    "CurveMetadata",
    "GuardrailViolation",
    "LoadedResult",
    "ResultLoader",
    "compute_file_sha256",
    "build_report_manifest",
]


class GuardrailViolation(RuntimeError):
    """Raised when a metrics payload violates a reporting guardrail."""


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _as_int(value: object) -> Optional[int]:
    numeric = _as_float(value)
    if numeric is None:
        return None
    rounded = round(numeric)
    if not math.isclose(numeric, rounded, rel_tol=0, abs_tol=1e-9):
        return None
    return int(rounded)


@dataclass(frozen=True)
class CurveMetadata:
    """Normalised description of a curve export entry."""

    key: str
    path: Path
    sha256: str
    metadata: Mapping[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "path": str(self.path),
            "sha256": self.sha256,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LoadedResult:
    """Container for a validated metrics payload."""

    metrics_path: Path
    payload: Mapping[str, Any]
    primary_metrics: Mapping[str, float]
    sensitivity_metrics: Mapping[str, float]
    curves: Mapping[str, CurveMetadata]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metrics_path": str(self.metrics_path),
            "primary_metrics": dict(self.primary_metrics),
            "sensitivity_metrics": dict(self.sensitivity_metrics),
            "curves": {key: value.as_dict() for key, value in self.curves.items()},
        }


class ResultLoader:
    def __init__(
        self,
        exp_id: str,
        *,
        required_curve_keys: Sequence[str] = (),
        strict: bool = True,
    ) -> None:
        self.exp_id = str(exp_id)
        self.required_curve_keys = tuple(required_curve_keys)
        self.strict = bool(strict)
        self._csv_digest_registry: Dict[str, str] = {}
        self._curve_digest_registry: Dict[str, str] = {}
        self._loaded_runs: list[Dict[str, Any]] = []

    @property
    def csv_digests(self) -> Dict[str, str]:
        return dict(self._csv_digest_registry)

    @property
    def curve_digests(self) -> Dict[str, str]:
        return dict(self._curve_digest_registry)

    @property
    def loaded_runs(self) -> Sequence[Mapping[str, Any]]:
        return tuple(self._loaded_runs)

    def validate(self, metrics_path: Path, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(payload, Mapping):
            raise GuardrailViolation(f"Metrics file '{metrics_path}' must contain a mapping payload")
        normalised = self.normalise_payload(payload)
        self._normalize_blocks(metrics_path, normalised)
        try:
            threshold_spec = THRESHOLD_SPECS[self.exp_id]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise GuardrailViolation(f"Unknown experiment id '{self.exp_id}'") from exc
        val_path = self._validate_schema(metrics_path, normalised)
        self._validate_thresholds(metrics_path, normalised, val_path, threshold_spec)
        if "sensitivity" in threshold_spec:
            if not isinstance(normalised.get("test_sensitivity"), Mapping):
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' is missing test_sensitivity metrics"
                )
        elif "test_sensitivity" in normalised:
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' unexpectedly defines test_sensitivity block"
            )
        self._validate_confusion(metrics_path, normalised, "test_primary")
        self._validate_confusion(metrics_path, normalised, "test_sensitivity")
        self._validate_csv_hashes(metrics_path, normalised)
        self._validate_curve_exports(metrics_path, normalised)
        self._register_run(normalised, metrics_path)
        return normalised

    def load(self, metrics_path: Path) -> LoadedResult:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return self.extract(metrics_path, payload)

    def extract(self, metrics_path: Path, payload: Mapping[str, Any]) -> LoadedResult:
        normalised = self.validate(metrics_path, payload)
        primary_metrics = self._extract_metric_block(normalised.get("test_primary"))
        sensitivity_metrics = self._extract_metric_block(normalised.get("test_sensitivity"))
        curves = self._normalise_curves(metrics_path, normalised)
        return LoadedResult(
            metrics_path=metrics_path,
            payload=MappingProxyType(dict(normalised)),
            primary_metrics=MappingProxyType(dict(primary_metrics)),
            sensitivity_metrics=MappingProxyType(dict(sensitivity_metrics)),
            curves=MappingProxyType(curves),
        )

    @staticmethod
    def normalise_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
        return {str(key): value for key, value in dict(payload).items()}

    def _normalize_blocks(self, metrics_path: Path, payload: Mapping[str, Any]) -> None:
        bad = [key for key in payload.keys() if isinstance(key, str) and key.startswith("eval_")]
        if bad:
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' contains disallowed evaluation keys: {sorted(bad)}"
            )

    def _validate_schema(self, metrics_path: Path, payload: Mapping[str, Any]) -> str:
        for key in ("thresholds", "data", "test_primary"):
            if key not in payload:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' is missing required block '{key}'"
                )
        data_block = payload.get("data")
        if not isinstance(data_block, Mapping):
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' data block must be a mapping"
            )
        required_splits = ("train", "val", "test")
        for split in required_splits:
            entry = data_block.get(split)
            if not isinstance(entry, Mapping):
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' data.{split} must be a mapping"
                )
            path_value = entry.get("path")
            sha_value = entry.get("sha256")
            if not isinstance(path_value, str) or not path_value.strip():
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' data.{split}.path is required"
                )
            if not isinstance(sha_value, str) or not sha_value.strip():
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' data.{split}.sha256 is required"
                )
        val_info = data_block["val"]
        val_path = str(val_info.get("path"))
        if not val_path:
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' data.val.path is required"
            )
        test_primary = payload.get("test_primary")
        if not isinstance(test_primary, Mapping):
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' test_primary block must be a mapping"
            )
        return val_path

    def _validate_thresholds(
        self,
        metrics_path: Path,
        payload: Mapping[str, Any],
        val_path: str,
        spec: Mapping[str, Mapping[str, Any]],
    ) -> None:
        thresholds = payload.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' is missing thresholds block"
            )

        def _check_slot(slot: str, slot_spec: Mapping[str, Any]) -> None:
            block = thresholds.get(slot)
            if not isinstance(block, Mapping):
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' does not define thresholds.{slot}"
                )
            policy = block.get("policy")
            if policy != slot_spec["policy"]:
                raise GuardrailViolation(
                    f"thresholds.{slot}.policy={policy!r} != {slot_spec['policy']!r}"
                )
            require_fields = list(slot_spec.get("require", ()))
            missing = [field for field in require_fields if field not in block]
            if missing:
                raise GuardrailViolation(
                    f"thresholds.{slot} missing fields: {missing}"
                )
            if "tau" in block:
                tau = _as_float(block.get("tau"))
                if tau is None or not (0.0 <= tau <= 1.0):
                    raise GuardrailViolation(
                        f"thresholds.{slot}.tau out of [0,1]: {block.get('tau')}"
                    )
            if "epoch" in block and _as_int(block.get("epoch")) is None:
                raise GuardrailViolation(
                    f"thresholds.{slot}.epoch must be an integer"
                )
            if "split" in slot_spec:
                expected = slot_spec["split"].replace("${val_path}", val_path)
                actual = block.get("split")
                if actual != expected:
                    raise GuardrailViolation(
                        f"thresholds.{slot}.split != data.val.path ({actual!r} vs {expected!r})"
                    )
            if block.get("policy") == "sun_val_frozen":
                source_split_expected = slot_spec.get("source_split")
                if block.get("source_split") != source_split_expected:
                    raise GuardrailViolation(
                        f"thresholds.{slot}.source_split must be {source_split_expected!r}"
                    )
                if not block.get("source_checkpoint"):
                    raise GuardrailViolation(
                        f"thresholds.{slot}.source_checkpoint missing"
                    )

        _check_slot("primary", spec["primary"])
        if "sensitivity" in spec:
            _check_slot("sensitivity", spec["sensitivity"])
        elif "sensitivity" in thresholds:
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' unexpectedly defines thresholds.sensitivity"
            )

    def _validate_confusion(
        self,
        metrics_path: Path,
        payload: Mapping[str, Any],
        block_name: str,
    ) -> None:
        block = payload.get(block_name)
        if not isinstance(block, Mapping):
            if block_name == "test_primary" and self.strict:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' is missing {block_name} metrics"
                )
            return
        required = ("tp", "fp", "tn", "fn", "n_pos", "n_neg")
        missing = [key for key in required if key not in block]
        if missing:
            if self.strict:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' is missing {block_name} fields: {', '.join(missing)}"
                )
            return
        tp = _as_int(block.get("tp"))
        fp = _as_int(block.get("fp"))
        tn = _as_int(block.get("tn"))
        fn = _as_int(block.get("fn"))
        n_pos = _as_int(block.get("n_pos"))
        n_neg = _as_int(block.get("n_neg"))
        values = [tp, fp, tn, fn, n_pos, n_neg]
        if any(value is None for value in values):
            if self.strict:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' contains non-integer confusion entries in {block_name}"
                )
            return
        confusion_total = tp + fp + tn + fn  # type: ignore[operator]
        class_total = n_pos + n_neg  # type: ignore[operator]
        if confusion_total != class_total:
            raise GuardrailViolation(
                f"Confusion totals disagree with class counts in {block_name} for '{metrics_path}'"
            )

    def _validate_csv_hashes(self, metrics_path: Path, payload: Mapping[str, Any]) -> None:
        digests: Dict[str, str] = {}
        data_block = payload.get("data")
        if isinstance(data_block, Mapping):
            for split in ("train", "val", "test"):
                entry = data_block.get(split)
                if isinstance(entry, Mapping):
                    sha_value = entry.get("sha256")
                    if isinstance(sha_value, str) and sha_value.strip():
                        digests[f"data.{split}.sha256"] = sha_value.strip().lower()
        provenance = payload.get("provenance")
        if isinstance(provenance, Mapping):
            for key, value in provenance.items():
                if isinstance(value, Mapping):
                    nested_sha = value.get("csv_sha256")
                    if isinstance(nested_sha, str) and nested_sha.strip():
                        digests[f"{key}.csv_sha256"] = nested_sha.strip().lower()
                elif isinstance(value, str) and key.endswith("_csv_sha256"):
                    text = value.strip().lower()
                    if text:
                        digests[key] = text
        if not digests:
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' does not declare any dataset sha256 digests"
            )
        for key, digest in digests.items():
            previous = self._csv_digest_registry.get(key)
            if previous is None:
                self._csv_digest_registry[key] = digest
            elif previous != digest:
                raise GuardrailViolation(
                    f"CSV digest mismatch for key '{key}' in '{metrics_path}': "
                    f"expected '{previous}', found '{digest}'."
                )

    def _validate_curve_exports(self, metrics_path: Path, payload: Mapping[str, Any]) -> None:
        required = set(self.required_curve_keys)
        if not required:
            return
        exports = payload.get("curve_exports")
        if not isinstance(exports, Mapping):
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' is missing curve_exports block"
            )
        for key in required:
            entry = exports.get(key)
            if not isinstance(entry, Mapping):
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' is missing curve_exports['{key}'] entry"
                )
            path_value = entry.get("path")
            sha_value = entry.get("sha256")
            if not isinstance(path_value, str) or not path_value.strip():
                raise GuardrailViolation(
                    f"curve_exports['{key}'] in '{metrics_path}' is missing a path"
                )
            rel_path = Path(path_value)
            curve_path = rel_path if rel_path.is_absolute() else (metrics_path.parent / rel_path)
            if not curve_path.exists():
                raise GuardrailViolation(
                    f"curve_exports['{key}'] references missing file '{curve_path}'"
                )
            computed_digest = compute_file_sha256(curve_path)
            if isinstance(sha_value, str) and sha_value.strip():
                expected_digest = sha_value.strip().lower()
            else:
                expected_digest = computed_digest
            if expected_digest != computed_digest:
                raise GuardrailViolation(
                    f"curve_exports['{key}'] digest mismatch for '{metrics_path}'"
                )
            previous = self._curve_digest_registry.get(key)
            if previous is None:
                self._curve_digest_registry[key] = expected_digest
            elif previous != expected_digest:
                raise GuardrailViolation(
                    f"curve_exports digest mismatch for key '{key}' in '{metrics_path}'"
                )

    def _register_run(self, payload: Mapping[str, Any], metrics_path: Path) -> None:
        provenance = payload.get("provenance")
        record: Dict[str, Any] = {"metrics_path": str(metrics_path)}
        if isinstance(provenance, Mapping):
            for field in ("model", "subset_percent", "train_seed", "pack_seed", "split"):
                if field in provenance:
                    record[field] = provenance[field]
        record.setdefault("seed", payload.get("seed"))
        primary = payload.get("test_primary")
        if isinstance(primary, Mapping) and "tau" in primary:
            record["primary_tau"] = primary["tau"]
        sensitivity = payload.get("test_sensitivity")
        if isinstance(sensitivity, Mapping) and "tau" in sensitivity:
            record["sensitivity_tau"] = sensitivity["tau"]
        thresholds = payload.get("thresholds")
        if isinstance(thresholds, Mapping):
            primary_thresh = thresholds.get("primary")
            if isinstance(primary_thresh, Mapping) and "policy" in primary_thresh:
                record["primary_threshold_policy"] = primary_thresh["policy"]
            sensitivity_thresh = thresholds.get("sensitivity")
            if isinstance(sensitivity_thresh, Mapping) and "policy" in sensitivity_thresh:
                record["sensitivity_threshold_policy"] = sensitivity_thresh["policy"]
        self._loaded_runs.append(record)

    def _normalise_curves(
        self, metrics_path: Path, payload: Mapping[str, Any]
    ) -> Dict[str, CurveMetadata]:
        exports = payload.get("curve_exports")
        curves: Dict[str, CurveMetadata] = {}
        if not isinstance(exports, Mapping):
            return curves
        for key, entry in exports.items():
            if not isinstance(entry, Mapping):
                continue
            raw_path = entry.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            rel_path = Path(raw_path)
            resolved_path = rel_path if rel_path.is_absolute() else (metrics_path.parent / rel_path)
            computed_digest = compute_file_sha256(resolved_path)
            sha_field = entry.get("sha256")
            if isinstance(sha_field, str) and sha_field.strip():
                digest = sha_field.strip().lower()
            else:
                digest = computed_digest
            metadata = {
                str(k): entry[k]
                for k in entry.keys()
                if k not in {"path", "sha256"}
            }
            curves[str(key)] = CurveMetadata(
                key=str(key),
                path=resolved_path.resolve(),
                sha256=digest,
                metadata=MappingProxyType(metadata),
            )
        return curves

    def _extract_metric_block(self, block: Optional[Mapping[str, Any]]) -> Dict[str, float]:
        if not isinstance(block, Mapping):
            return {}
        metrics: Dict[str, float] = {}
        for key, value in block.items():
            numeric = _as_float(value)
            if numeric is None:
                continue
            metrics[str(key)] = float(numeric)
        return metrics


def build_report_manifest(
    *,
    output_path: Optional[Path],
    loader: ResultLoader,
    runs: Iterable[Mapping[str, Any]],
    rng_seed: Optional[int] = None,
    bootstrap: Optional[int] = None,
    extra_outputs: Optional[Iterable[Path]] = None,
    metadata: Optional[MutableMapping[str, Any]] = None,
    validated_seeds: Optional[Sequence[int]] = None,
    seed_groups: Optional[Mapping[str, Sequence[int]]] = None,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap": int(bootstrap) if bootstrap is not None else None,
        "rng_seed": int(rng_seed) if rng_seed is not None else None,
        "runs": list(runs),
        "csv_digests": loader.csv_digests,
        "curve_digests": loader.curve_digests,
    }
    if validated_seeds is not None:
        manifest["validated_seeds"] = [int(seed) for seed in validated_seeds]
    if seed_groups:
        manifest["seed_groups"] = {
            str(key): [int(seed) for seed in value]
            for key, value in seed_groups.items()
        }
    if metadata:
        manifest.update(metadata)
    outputs: list[Dict[str, Any]] = []
    candidates: list[Path] = []
    if output_path is not None and output_path.exists():
        candidates.append(output_path)
    if extra_outputs:
        for item in extra_outputs:
            if item and item.exists():
                candidates.append(item)
    for path in candidates:
        outputs.append({"path": str(path), "sha256": compute_file_sha256(path)})
    manifest["outputs"] = outputs
    return manifest
