"""Experiment-specific threshold policies for report validation.

Authoritative τ-based validation metrics live only in ``policy_record['metrics_at_tau']``
entries (mirrored into ``metrics_at_threshold`` in the saved threshold file). Any content
under ``debug/*`` is non-canonical and may correspond to a different τ.
"""

from __future__ import annotations

THRESHOLD_SPECS = {
    "exp1": {
        "primary":     {"policy": "f1_opt_on_val", "split": "${val_path}", "require": ["tau", "split", "epoch"]},
        "sensitivity": {"policy": "youden_on_val", "split": "${val_path}", "require": ["tau", "split", "epoch"]},
    },
    "exp2": {
        "primary":     {"policy": "f1_opt_on_val", "split": "${val_path}", "require": ["tau", "split", "epoch"]},
        "sensitivity": {"policy": "youden_on_val", "split": "${val_path}", "require": ["tau", "split", "epoch"]},
    },
    "exp3b": {
        "primary":     {"policy": "f1_opt_on_val", "split": "${val_path}", "require": ["tau", "split", "epoch"]},
        "sensitivity": {"policy": "youden_on_val", "split": "${val_path}", "require": ["tau", "split", "epoch"]},
        "expected_val_path": "sun_morphology/val",
    },
    "exp4": {
        "primary":     {"policy": "f1_opt_on_val", "split": "${val_path}", "require": ["tau", "split", "epoch"]},
        "sensitivity": {"policy": "youden_on_val", "split": "${val_path}", "require": ["tau", "split", "epoch"]},
    },
    "exp5a": {
        "primary":     {"policy": "sun_val_frozen", "source_split": "sun_full/val", "require": ["tau", "source_checkpoint", "source_split"]},
    },
    "exp5b": {
        "primary":     {"policy": "sun_val_frozen", "source_split": "sun_full/val", "require": ["tau", "source_checkpoint", "source_split"]},
    },
    "exp5c": {
        "primary":     {"policy": "sun_val_frozen", "source_split": "sun_full/val", "require": ["tau", "source_checkpoint", "source_split"]},
        "sensitivity": {"policy": "val_opt_youden", "split": "${val_path}", "require": ["tau", "split", "epoch"]},
    },
}

__all__ = ["THRESHOLD_SPECS"]
