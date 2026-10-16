"""Figures (PNG): E1 learning curves, H1–H7 forest plot, E2 strata, E4 arms, reliability diagrams, E3 LOCO."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import Result  # noqa: E402
from .sap import E1_PERCENTS, E4_ARMS, E4_BUDGETS, LABEL, REGIMES  # noqa: E402

# Categorical slots 1–4 of the reference palette (fixed order = REGIMES), plus a marker per regime.
COLOR = {"sup_imnet": "#2a78d6", "ssl_imnet": "#eb6834", "ssl_colon": "#1baf7a", "ssl_imnet_matched": "#eda100"}
MARKER = {"sup_imnet": "o", "ssl_imnet": "s", "ssl_colon": "D", "ssl_imnet_matched": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 2, "lines.markersize": 6,
})


def _regimes(res: Result) -> List[str]:
    return [g for g in REGIMES if g in set(res.run_metrics["regime"])]


def _mean_sd(df: pd.DataFrame, metric: str):
    return df[metric].mean(), df[metric].std(ddof=1)


def e1_curves(res: Result, path: Path) -> None:
    rm = res.run_metrics[res.run_metrics["family"] == "e1"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for ax, metric, title in zip(axes, ("auprc", "auroc"), ("AUPRC", "AUROC")):
        for g in _regimes(res):
            pts = [(p, *_mean_sd(rm[(rm["regime"] == g) & (rm["variant"] == p)], metric)) for p in E1_PERCENTS]
            x, m, sd = np.array(pts).T
            ax.plot(x, m, color=COLOR[g], marker=MARKER[g], label=LABEL[g])
            ax.fill_between(x, m - sd, m + sd, color=COLOR[g], alpha=0.12, linewidth=0)
        ax.set_xlabel("training budget (% of the partition's train split)")
        ax.set_title(f"E1 {title} (mean ± SD over runs)", color=INK, loc="left")
        ax.set_xticks(E1_PERCENTS)
    axes[0].legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def forest(res: Result, path: Path) -> None:
    h = res.hypotheses[res.hypotheses["estimate"].notna()] if "estimate" in res.hypotheses else res.hypotheses.iloc[0:0]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * max(len(h), 1) + 1.2))
    y = np.arange(len(h))[::-1]
    for yi, (_, r) in zip(y, h.iterrows()):
        ax.plot([r["model_ci_lo"], r["model_ci_hi"]], [yi + 0.12] * 2, color="#2a78d6", linewidth=3, solid_capstyle="round")
        ax.plot([r["boot_lo"], r["boot_hi"]], [yi - 0.12] * 2, color=MUTED, linewidth=1.5, solid_capstyle="round")
        ax.plot(r["estimate"], yi, marker="D" if r["confirmed"] else "o", color=INK, markersize=7,
                markerfacecolor=INK if r["confirmed"] else "white")
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['hypothesis']}  p_Holm={r['p_holm']:.3g}" for _, r in h.iterrows()])
    ax.set_xlabel("paired difference (metric units)")
    ax.set_title("H1–H7: estimate, model 95 % CI (thick), cluster-bootstrap 95 % CI (thin); ◆ confirmed", color=INK, loc="left", fontsize=9)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def e2_strata(res: Result, path: Path) -> None:
    rm = res.run_metrics[res.run_metrics["family"] == "e2"]
    regs = _regimes(res)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, st in enumerate(("overall", "flat", "polypoid")):
        for j, g in enumerate(regs):
            sub = rm[(rm["regime"] == g) & (rm["slot"] == st)]
            m, sd = _mean_sd(sub, "auroc")
            x = i + (j - (len(regs) - 1) / 2) * 0.18
            ax.errorbar(x, m, yerr=sd, color=COLOR[g], marker=MARKER[g], capsize=0, linewidth=2,
                        label=LABEL[g] if i == 0 else None)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["overall", "negatives + flat", "negatives + polypoid"])
    ax.set_ylabel("AUROC")
    ax.set_title("E2 AUROC by morphology stratum (mean ± SD over runs)", color=INK, loc="left")
    ax.legend(loc="lower left", ncol=2)
    fig.savefig(path)
    plt.close(fig)


def e4_arms(res: Result, path: Path, metric: str = "balanced_accuracy") -> None:
    rm = res.run_metrics[res.run_metrics["family"] == "e4"]
    regs = _regimes(res)
    fig, axes = plt.subplots(1, len(E4_BUDGETS), figsize=(11, 3.2), sharey=True)
    for ax, b in zip(axes, E4_BUDGETS):
        for g in regs:
            pts = [_mean_sd(rm[(rm["regime"] == g) & (rm["variant"] == b) & (rm["slot"] == a)], metric) for a in E4_ARMS]
            m, sd = np.array(pts).T
            dx = (regs.index(g) - (len(regs) - 1) / 2) * 0.12
            ax.errorbar(np.arange(4) + dx, m, yerr=sd, color=COLOR[g], marker=MARKER[g], label=LABEL[g], capsize=0, linewidth=1.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(E4_ARMS)
        ax.set_title(f"s{b}", color=INK, loc="left")
        ax.set_xlabel("arm")
    axes[0].set_ylabel(metric.replace("_", " "))
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("E4 arms (A unadapted/inherited τ, B unadapted/recalibrated, C adapted/inherited, D adapted/recalibrated)",
                 x=0.01, ha="left", fontsize=9, color=INK)
    fig.savefig(path)
    plt.close(fig)


def reliability(rel: pd.DataFrame, res: Result, path: Path) -> None:
    if rel.empty:
        return
    panels = [("sun_full", None, "all", "sun_full test"), ("e3", None, "all", "E3 polypgen_clean_test")]
    panels += [("e4", b, arm, f"E4 s{b} {'unadapted' if arm == 'A' else 'adapted'}") for b in E4_BUDGETS for arm in ("A", "C")]
    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.7 * nrow), sharex=True, sharey=True)
    for ax, (fam, variant, slot, title) in zip(axes.ravel(), panels):
        sub = rel[(rel["family"] == fam) & (rel["slot"] == slot)]
        sub = sub[sub["variant"].isna()] if variant is None else sub[sub["variant"] == variant]
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
        for g in _regimes(res):
            s = sub[sub["regime"] == g].sort_values("bin")
            ax.plot(s["mean_prob"], s["frac_pos"], color=COLOR[g], marker=MARKER[g], markersize=4, linewidth=1.5, label=LABEL[g])
        ax.set_title(title, color=INK, loc="left", fontsize=8)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    axes.ravel()[0].legend(fontsize=6, loc="upper left")
    fig.supxlabel("mean predicted probability (15 equal-mass bins)")
    fig.supylabel("observed fraction of positives")
    fig.savefig(path)
    plt.close(fig)


def loco_e3(res: Result, path: Path) -> None:
    sm = res.summary
    if "analysis" not in sm:
        return
    sub = sm[(sm["analysis"] == "E3 leave-one-centre-out") & (sm["metric"] == "auroc")]
    if sub.empty:
        return
    comps = list(dict.fromkeys(sub["comparison"]))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    centers = sorted(set(sub["excluded"]))
    for j, comp in enumerate(comps):
        base = comp.split(" - ")[1]
        s = sub[sub["comparison"] == comp].set_index("excluded").loc[centers]
        x = np.arange(len(centers)) + (j - (len(comps) - 1) / 2) * 0.2
        ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["model_ci_lo"], s["model_ci_hi"] - s["mean"]],
                    color=COLOR.get(base, MUTED), marker=MARKER.get(base, "o"), linestyle="none", label=f"SSL-colon − {LABEL.get(base, base)}")
    full = sm[sm["id"] == "H4"]
    if not full.empty:
        ax.axhline(float(full["mean"].iloc[0]), color="#2a78d6", linewidth=1, linestyle=":")
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(centers)))
    ax.set_xticklabels([f"without {c}" for c in centers])
    ax.set_ylabel("Δ AUROC (model 95 % CI)")
    ax.set_title("E3 leave-one-centre-out (dotted: H4 with all centres)", color=INK, loc="left")
    ax.legend(fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def write_figures(res: Result, rel: pd.DataFrame, out: Path) -> List[Path]:
    out = Path(out) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    jobs = {
        "fig_e1_learning_curves.png": lambda p: e1_curves(res, p),
        "fig_hypotheses_forest.png": lambda p: forest(res, p),
        "fig_e2_strata.png": lambda p: e2_strata(res, p),
        "fig_e4_arms_balanced_accuracy.png": lambda p: e4_arms(res, p),
        "fig_reliability.png": lambda p: reliability(rel, res, p),
        "fig_e3_loco.png": lambda p: loco_e3(res, p),
    }
    written = []
    for name, fn in jobs.items():
        fn(out / name)
        if (out / name).exists():
            written.append(out / name)
    return written
