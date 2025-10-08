from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "GuardrailViolation",
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


def _canonical_policy(raw: Optional[object]) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


@dataclass
class ResultLoader:
    expected_primary_policy: Optional[str] = None
    expected_sensitivity_policy: Optional[str] = None
    require_sensitivity: bool = False
    required_curve_keys: Sequence[str] = ()
    strict: bool = True

    def __post_init__(self) -> None:
        self.expected_primary_policy = _canonical_policy(self.expected_primary_policy)
        self.expected_sensitivity_policy = _canonical_policy(self.expected_sensitivity_policy)
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

    def validate(self, metrics_path: Path, payload: Mapping[str, Any]) -> None:
        self._validate_thresholds(metrics_path, payload)
        self._validate_confusion(metrics_path, payload, "test_primary")
        self._validate_confusion(metrics_path, payload, "test_sensitivity")
        self._validate_csv_hashes(metrics_path, payload)
        self._validate_curve_exports(metrics_path, payload)
        self._register_run(payload, metrics_path)

    def _validate_thresholds(self, metrics_path: Path, payload: Mapping[str, Any]) -> None:
        thresholds = payload.get("thresholds")
        if not isinstance(thresholds, Mapping):
            if self.strict:
                raise GuardrailViolation(f"Metrics file '{metrics_path}' is missing thresholds block")
            return
        primary = thresholds.get("primary")
        if not isinstance(primary, Mapping):
            if self.strict:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' does not provide thresholds.primary record"
                )
            return
        policy = _canonical_policy(primary.get("policy"))
        if self.expected_primary_policy and policy != self.expected_primary_policy:
            raise GuardrailViolation(
                "Primary threshold policy mismatch for "
                f"'{metrics_path}'. Expected '{self.expected_primary_policy}', found '{policy or 'none'}'."
            )
        sensitivity_policy_expected = self.expected_sensitivity_policy
        sensitivity_record = thresholds.get("sensitivity") if isinstance(thresholds, Mapping) else None
        if self.require_sensitivity or sensitivity_policy_expected:
            if not isinstance(sensitivity_record, Mapping):
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' is missing thresholds.sensitivity record"
                )
            record_policy = _canonical_policy(sensitivity_record.get("policy"))
            if sensitivity_policy_expected and record_policy != sensitivity_policy_expected:
                raise GuardrailViolation(
                    "Sensitivity threshold policy mismatch for "
                    f"'{metrics_path}'. Expected '{sensitivity_policy_expected}', found '{record_policy or 'none'}'."
                )
        elif isinstance(sensitivity_record, Mapping) and self.strict:
            # Experiments without sensitivity policy should not emit it.
            if self.expected_sensitivity_policy is None:
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
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            if self.strict:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' is missing provenance section"
                )
            return
        digests: Dict[str, str] = {}
        for key, value in provenance.items():
            if isinstance(value, Mapping):
                nested_sha = value.get("csv_sha256")
                if isinstance(nested_sha, str) and nested_sha.strip():
                    digest = nested_sha.strip().lower()
                    digests[f"{key}.csv_sha256"] = digest
            elif isinstance(value, str) and key.endswith("_csv_sha256"):
                text = value.strip().lower()
                if text:
                    digests[key] = text
        if not digests:
            if self.strict:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' does not declare any *_csv_sha256 provenance hashes"
                )
            return
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
        self._loaded_runs.append(record)


def build_report_manifest(
    *,
    output_path: Optional[Path],
    loader: ResultLoader,
    runs: Iterable[Mapping[str, Any]],
    rng_seed: Optional[int] = None,
    bootstrap: Optional[int] = None,
    extra_outputs: Optional[Iterable[Path]] = None,
    metadata: Optional[MutableMapping[str, Any]] = None,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap": int(bootstrap) if bootstrap is not None else None,
        "rng_seed": int(rng_seed) if rng_seed is not None else None,
        "runs": list(runs),
        "csv_digests": loader.csv_digests,
        "curve_digests": loader.curve_digests,
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