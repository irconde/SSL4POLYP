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


_INTEGER_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "tp",
        "fp",
        "tn",
        "fn",
        "n_pos",
        "n_neg",
        "n_total",
        "count",
    }
)
_INTEGER_METRIC_PREFIXES: Tuple[str, ...] = ("n_",)


def _is_integer_metric_key(key: str) -> bool:
    """Return ``True`` when ``key`` corresponds to an integral metric."""

    normalised = key.strip().lower()
    if normalised in _INTEGER_METRIC_KEYS:
        return True
    return any(normalised.startswith(prefix) for prefix in _INTEGER_METRIC_PREFIXES)


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
        enforce_curve_exports: bool = True,
    ) -> None:
        self.exp_id = str(exp_id)
        self.required_curve_keys = tuple(required_curve_keys)
        self.strict = bool(strict)
        self.enforce_curve_exports = bool(enforce_curve_exports)
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
        expected_val_paths = self._normalise_expected_val_paths(
            threshold_spec.get("expected_val_path")
        )
        val_path = self._validate_schema(
            metrics_path,
            normalised,
            expected_val_paths=expected_val_paths,
        )
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
        return ResultLoader._strip_debug_sections(
            {str(key): value for key, value in dict(payload).items()}
        )

    @staticmethod
    def _strip_debug_sections(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            cleaned: Dict[str, Any] = {}
            for key, value in obj.items():
                key_str = str(key)
                if key_str.startswith("debug"):
                    continue
                cleaned[key_str] = ResultLoader._strip_debug_sections(value)
            return cleaned
        if isinstance(obj, list):
            return [ResultLoader._strip_debug_sections(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(ResultLoader._strip_debug_sections(item) for item in obj)
        return obj

    def _normalize_blocks(self, metrics_path: Path, payload: Mapping[str, Any]) -> None:
        bad = [key for key in payload.keys() if isinstance(key, str) and key.startswith("eval_")]
        if bad:
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' contains disallowed evaluation keys: {sorted(bad)}"
            )

    def _validate_schema(
        self,
        metrics_path: Path,
        payload: Mapping[str, Any],
        *,
        expected_val_paths: Sequence[str] = (),
    ) -> Optional[str]:
        eval_only = bool(payload.get("eval_only"))
        required_blocks = ["thresholds", "data", "test_primary"]
        if not eval_only:
            required_blocks.append("val")
        for key in required_blocks:
            if key not in payload:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' is missing required block '{key}'"
                )
        data_block = payload.get("data")
        if not isinstance(data_block, Mapping):
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' data block must be a mapping"
            )
        present_splits: Dict[str, Mapping[str, Any]] = {}
        for split in ("train", "val", "test"):
            entry = data_block.get(split)
            if entry is None:
                continue
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
            present_splits[split] = entry
        if not present_splits:
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' data block did not include any recognised splits"
            )
        if eval_only:
            if "test" not in present_splits:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' must provide data.test when eval_only is true"
                )
        else:
            missing = [split for split in ("train", "val", "test") if split not in present_splits]
            if missing:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' is missing required data splits: {missing}"
                )
        val_path: Optional[str] = None
        if "val" in present_splits:
            val_info = present_splits["val"]
            val_path_raw = val_info.get("path")
            if not isinstance(val_path_raw, str) or not val_path_raw.strip():
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' data.val.path is required"
                )
            val_path = val_path_raw.strip()
            normalised_val_path = self._normalise_manifest_path(val_path)
            if expected_val_paths:
                allowed = {
                    self._normalise_manifest_path(candidate)
                    for candidate in expected_val_paths
                }
                if normalised_val_path not in allowed:
                    options = ", ".join(sorted(allowed)) or "<unknown>"
                    raise GuardrailViolation(
                        f"Metrics file '{metrics_path}' data.val.path must reference one of "
                        f"[{options}] (found {val_path!r})"
                    )
        elif expected_val_paths:
            options = ", ".join(
                sorted(self._normalise_manifest_path(candidate) for candidate in expected_val_paths)
            ) or "<unknown>"
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' is missing data.val.path but expected one of [{options}]"
            )
        test_primary = payload.get("test_primary")
        if not isinstance(test_primary, Mapping):
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' test_primary block must be a mapping"
            )
        val_block = payload.get("val")
        if val_block is not None or not eval_only:
            if not isinstance(val_block, Mapping):
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' val block must be a mapping"
                )
        return val_path

    def _validate_thresholds(
        self,
        metrics_path: Path,
        payload: Mapping[str, Any],
        val_path: Optional[str],
        spec: Mapping[str, Any],
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
                if not val_path:
                    raise GuardrailViolation(
                        f"thresholds.{slot}.split declared but data.val.path is unavailable"
                    )
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

    @staticmethod
    def _normalise_expected_val_paths(raw: Optional[object]) -> Tuple[str, ...]:
        if raw is None:
            return ()
        if isinstance(raw, (list, tuple, set, frozenset)):
            items = list(raw)
        else:
            items = [raw]
        normalised: list[str] = []
        for item in items:
            text = str(item).strip()
            if text:
                normalised.append(text)
        return tuple(normalised)

    @staticmethod
    def _normalise_manifest_path(candidate: object) -> str:
        text = str(candidate).strip()
        if not text:
            return ""
        text = text.replace("\\", "/")
        while text.startswith("./"):
            text = text[2:]
        text = text.lstrip("/")
        if "/data_packs/" in text:
            text = text.split("/data_packs/", 1)[1]
        if text.startswith("data_packs/"):
            text = text[len("data_packs/") :]
        if text.endswith(".csv"):
            text = text[:-4]
        return text

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
        tp = int(tp)  # type: ignore[arg-type]
        fp = int(fp)  # type: ignore[arg-type]
        tn = int(tn)  # type: ignore[arg-type]
        fn = int(fn)  # type: ignore[arg-type]
        n_pos = int(n_pos)  # type: ignore[arg-type]
        n_neg = int(n_neg)  # type: ignore[arg-type]
        prevalence_value = block.get("prevalence")
        prevalence = _as_float(prevalence_value)
        if prevalence is None:
            if self.strict:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' is missing {block_name}.prevalence"
                )
            return
        if not 0.0 <= prevalence <= 1.0:
            raise GuardrailViolation(
                f"Metrics file '{metrics_path}' reports invalid {block_name}.prevalence={prevalence_value!r}"
            )
        confusion_total = tp + fp + tn + fn  # type: ignore[operator]
        class_total = n_pos + n_neg  # type: ignore[operator]
        if confusion_total != class_total:
            raise GuardrailViolation(
                f"Confusion totals disagree with class counts in {block_name} for '{metrics_path}'"
            )
        if class_total > 0:
            expected_prevalence = float(n_pos) / float(class_total)
            if not math.isclose(prevalence, expected_prevalence, rel_tol=1e-6, abs_tol=1e-6):
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' reports {block_name}.prevalence={prevalence_value!r} "
                    f"but expected approximately {expected_prevalence:.6f}"
                )

    def _contextualise_csv_digest_key(
        self,
        base_key: str,
        *,
        split: Optional[str] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        entry: Optional[Mapping[str, Any]] = None,
    ) -> str:
        scope = split
        if scope is None:
            if base_key.startswith("data.train.") or base_key.startswith("train_"):
                scope = "train"
        if scope != "train":
            return base_key

        context_parts: list[str] = []
        seen_parts: set[str] = set()

        def _add_part(part: Optional[str]) -> None:
            if part and part not in seen_parts:
                seen_parts.add(part)
                context_parts.append(part)

        if isinstance(provenance, Mapping):
            subset_value = _as_float(provenance.get("subset_percent"))
            if subset_value is not None:
                _add_part(f"subset={subset_value:g}")
            pack_seed_value = _as_int(provenance.get("pack_seed"))
            if pack_seed_value is not None:
                _add_part(f"pack_seed={pack_seed_value}")
            for field in ("train_pack_name", "train_pack"):
                pack_value = provenance.get(field)
                if isinstance(pack_value, str) and pack_value.strip():
                    _add_part(f"pack={pack_value.strip()}")
                    break

        if isinstance(entry, Mapping):
            pack_spec_value = entry.get("pack_spec")
            if isinstance(pack_spec_value, str) and pack_spec_value.strip():
                _add_part(f"pack_spec={Path(pack_spec_value.strip()).name}")
            for path_key in ("path", "csv_path"):
                path_value = entry.get(path_key)
                if isinstance(path_value, str) and path_value.strip():
                    _add_part(f"path={Path(path_value.strip()).name}")
                    break

        if not context_parts:
            return base_key
        context = ";".join(context_parts)
        return f"{base_key}[{context}]"

    def _validate_csv_hashes(self, metrics_path: Path, payload: Mapping[str, Any]) -> None:
        digests: Dict[str, str] = {}
        provenance = payload.get("provenance")
        if isinstance(provenance, Mapping):
            provenance_mapping: Mapping[str, Any] = provenance
        else:
            provenance_mapping = {}

        data_block = payload.get("data")
        if isinstance(data_block, Mapping):
            for split in ("train", "val", "test"):
                entry = data_block.get(split)
                if isinstance(entry, Mapping):
                    sha_value = entry.get("sha256")
                    if isinstance(sha_value, str) and sha_value.strip():
                        base_key = f"data.{split}.sha256"
                        contextual_key = self._contextualise_csv_digest_key(
                            base_key,
                            split=split,
                            provenance=provenance_mapping,
                            entry=entry,
                        )
                        digests[contextual_key] = sha_value.strip().lower()
        if isinstance(provenance, Mapping):
            for key, value in provenance.items():
                if isinstance(value, Mapping):
                    nested_sha = value.get("csv_sha256")
                    if isinstance(nested_sha, str) and nested_sha.strip():
                        base_key = f"{key}.csv_sha256"
                        contextual_key = self._contextualise_csv_digest_key(
                            base_key,
                            split="train" if str(key).endswith("train") else None,
                            provenance=provenance_mapping,
                            entry=value,
                        )
                        digests[contextual_key] = nested_sha.strip().lower()
                elif isinstance(value, str) and key.endswith("_csv_sha256"):
                    text = value.strip().lower()
                    if text:
                        contextual_key = self._contextualise_csv_digest_key(
                            key,
                            split="train" if key.startswith("train_") else None,
                            provenance=provenance_mapping,
                        )
                        digests[contextual_key] = text
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
            if self.enforce_curve_exports:
                raise GuardrailViolation(
                    f"Metrics file '{metrics_path}' is missing curve_exports block"
                )
            return
        for key in required:
            entry = exports.get(key)
            if not isinstance(entry, Mapping):
                if self.enforce_curve_exports:
                    raise GuardrailViolation(
                        f"Metrics file '{metrics_path}' is missing curve_exports['{key}'] entry"
                    )
                continue
            path_value = entry.get("path")
            sha_value = entry.get("sha256")
            if not isinstance(path_value, str) or not path_value.strip():
                if self.enforce_curve_exports:
                    raise GuardrailViolation(
                        f"curve_exports['{key}'] in '{metrics_path}' is missing a path"
                    )
                continue
            rel_path = Path(path_value)
            curve_path = rel_path if rel_path.is_absolute() else (metrics_path.parent / rel_path)
            if not curve_path.exists():
                if self.enforce_curve_exports:
                    raise GuardrailViolation(
                        f"curve_exports['{key}'] references missing file '{curve_path}'"
                    )
                continue
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
            key_text = str(key)
            if _is_integer_metric_key(key_text):
                numeric_int = _as_int(value)
                if numeric_int is None:
                    continue
                metrics[key_text] = int(numeric_int)
                continue
            numeric = _as_float(value)
            if numeric is None:
                continue
            metrics[key_text] = float(numeric)
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
