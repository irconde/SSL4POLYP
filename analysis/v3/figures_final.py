"""Final manuscript figures (CMPB submission) drawn from the existing campaign v3 analysis outputs.

Reads only files already written by the analysis (nothing is recomputed for plotting):
  results_v3/analysis/tables/table_hypotheses.csv          Figure 2 (H1–H7 forest plot)
  results_v3/analysis/tables/table3_e1_performance.csv     Figure 3 (E1 learning curves)
  results_v3/analysis/tables/table_loco_e3.csv             Figure 4 (E3 leave-one-centre-out)
  results_v3/analysis/tables/table8_e4_arms.csv            Figure 5 (E4 arms)
  results_v3/analysis_exploratory_lr_common/contrasts_common_vs_v3.csv   Figure 6 (exploratory)
and, only to cross-check the plotted values, ``contrast_summary.csv``, ``run_metrics.csv`` and
``paired_differences.csv``.

Writes ``results_v3/figures_final/figure{2..6}.{pdf,png}`` (vector PDF; PNG at 300 dpi), 174 mm wide, no
title inside the image, and ``plotted_values.csv`` (every value drawn) plus ``verification.csv``.

Run from the repository root in the ``ssl4polyp-analysis`` environment:
    python -m analysis.v3.figures_final
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "results_v3/analysis"
TABLES = ANALYSIS / "tables"
EXPLORATORY = REPO / "results_v3/analysis_exploratory_lr_common"
OUT = REPO / "results_v3/figures_final"

MM = 1 / 25.4
WIDTH_IN = 174 * MM  # double column

REGIMES = ["sup_imnet", "ssl_imnet", "ssl_colon", "ssl_imnet_matched"]
LABEL = {"sup_imnet": "SUP-imnet", "ssl_imnet": "SSL-imnet", "ssl_colon": "SSL-colon",
         "ssl_imnet_matched": "SSL-imnet-matched"}
# Okabe–Ito colours, one per regime, identical in every figure.
COLOR = {"sup_imnet": "#0072B2", "ssl_imnet": "#E69F00", "ssl_colon": "#009E73", "ssl_imnet_matched": "#CC79A7"}
MARKER = {"sup_imnet": "o", "ssl_imnet": "s", "ssl_colon": "D", "ssl_imnet_matched": "^"}
BLACK, GREY = "#000000", "#7f7f7f"
ARMS = {"A": "A: unadapted, inherited τ", "B": "B: unadapted, recalibrated τ",
        "C": "C: adapted, inherited τ", "D": "D: adapted, recalibrated τ"}
ARM_STYLE = {"A": ("-", "o"), "B": ("--", "s"), "C": ("-.", "^"), "D": (":", "D")}
HYP_SHORT = {
    "H1": "H1: SSL-colon − SUP-imnet, AULC of AUPRC",
    "H2": "H2: SSL-colon − SSL-imnet-matched, AULC of AUPRC",
    "H3": "H3: SSL-colon − SUP-imnet, AUROC (flat stratum)",
    "H4": "H4: SSL-colon − SUP-imnet, AUROC (PolypGen, zero-shot)",
    "H5": "H5: sequence − isolated negatives, AUROC",
    "H6": "H6: arm B − arm A, balanced accuracy",
    "H7": "H7: arm C − arm A, balanced accuracy",
}


def _font() -> str:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "DejaVu Sans"):
        if name in names:
            return name
    return "DejaVu Sans"


FONT = _font()
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": 8,
    "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "patch.linewidth": 0.6, "axes.edgecolor": BLACK, "axes.labelcolor": BLACK, "xtick.color": BLACK,
    "ytick.color": BLACK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

ROWS: List[Dict[str, object]] = []


def record(figure: str, element: str, key: str, x, value, lo=None, hi=None, source: str = "", source_key: str = "") -> None:
    ROWS.append({"figure": figure, "element": element, "key": key, "x": x, "value": value, "lo": lo, "hi": hi,
                 "source": source, "source_key": source_key})


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def figure2(h: pd.DataFrame) -> None:
    src = "results_v3/analysis/tables/table_hypotheses.csv"
    h = h.set_index("hypothesis").loc[list(HYP_SHORT)]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 74 * MM), layout="constrained")
    y = np.arange(len(h))[::-1]
    for yi, (hid, r) in zip(y, h.iterrows()):
        ax.plot([r["model_ci_lo"], r["model_ci_hi"]], [yi, yi], color=BLACK, linewidth=3.0, solid_capstyle="butt", zorder=2)
        ax.plot([r["boot_lo"], r["boot_hi"]], [yi, yi], color=BLACK, linewidth=0.9, solid_capstyle="butt", zorder=1)
        filled = bool(r["confirmed"])
        ax.plot(r["estimate"], yi, marker="o", markersize=5.5, color=BLACK, markeredgewidth=0.9,
                markerfacecolor=BLACK if filled else "white", zorder=3)
        record("Figure 2", "estimate", hid, None, r["estimate"], source=src, source_key=f"{hid}.estimate")
        record("Figure 2", "t-test CI (partition means)", hid, None, None, r["model_ci_lo"], r["model_ci_hi"], src, f"{hid}.model_ci")
        record("Figure 2", "bootstrap CI", hid, None, None, r["boot_lo"], r["boot_hi"], src, f"{hid}.boot")
        record("Figure 2", "confirmed (filled marker)", hid, None, int(filled), source=src, source_key=f"{hid}.confirmed")
    ax.axvline(0, color=GREY, linewidth=0.6, zorder=0)
    ax.set_yticks(y, [HYP_SHORT[hid] for hid in h.index])
    ax.set_xlabel("Paired difference")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    handles = [
        Line2D([], [], color=BLACK, linewidth=3.0, label="95% CI, t-test on partition means"),
        Line2D([], [], color=BLACK, linewidth=0.9, label="95% CI, cluster bootstrap"),
        Line2D([], [], color=BLACK, marker="o", linestyle="none", markersize=5.5, label="Confirmed"),
        Line2D([], [], color=BLACK, marker="o", linestyle="none", markersize=5.5, markerfacecolor="white",
               label="Not confirmed"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, handlelength=1.8, columnspacing=1.2)
    save(fig, "figure2_hypotheses_forest")


def figure3(t3: pd.DataFrame) -> None:
    src = "results_v3/analysis/tables/table3_e1_performance.csv"
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 70 * MM), layout="constrained")
    for ax, metric, ylabel in zip(axes, ("auprc", "auroc"), ("AUPRC", "AUROC")):
        for g in REGIMES:
            s = t3[t3["regime"] == g].sort_values("percent")
            x = s["percent"].to_numpy(float)
            m = s[f"{metric}_mean"].to_numpy(float)
            sd = s[f"{metric}_sd_partition"].to_numpy(float)
            ax.fill_between(x, m - sd, m + sd, color=COLOR[g], alpha=0.15, linewidth=0)
            ax.plot(x, m, color=COLOR[g], marker=MARKER[g], markersize=4, linewidth=1.2, label=LABEL[g])
            for xi, mi, si in zip(x, m, sd):
                record("Figure 3", f"{ylabel} mean (band ± SD of partition means)", LABEL[g], int(xi), mi, mi - si, mi + si,
                       src, f"percent={int(xi)},regime={g},{metric}_mean/{metric}_sd_partition")
        ax.set_xlabel("Training budget (% of the SUN training split)")
        ax.set_ylabel(f"{ylabel} on SUN test")
        ax.set_xticks([5, 10, 25, 50, 100])
        ax.set_xlim(0, 103)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.16, 1.0, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    save(fig, "figure3_e1_learning_curves")


def figure4(loco: pd.DataFrame, h: pd.DataFrame) -> None:
    src_l = "results_v3/analysis/tables/table_loco_e3.csv"
    src_h = "results_v3/analysis/tables/table_hypotheses.csv"
    sub = loco[(loco["comparison"] == "ssl_colon - sup_imnet") & (loco["metric"] == "auroc")].set_index("excluded")
    h4 = h.set_index("hypothesis").loc["H4"]
    rows = [("All centers", h4["estimate"], h4["model_ci_lo"], h4["model_ci_hi"], src_h, "H4")]
    rows += [(f"Without {c}", sub.loc[c, "mean"], sub.loc[c, "model_ci_lo"], sub.loc[c, "model_ci_hi"], src_l, f"excluded={c}")
             for c in ("C1", "C2", "C3", "C4", "C5", "C6")]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 62 * MM), layout="constrained")
    y = np.arange(len(rows))[::-1]
    for yi, (label, est, lo, hi, src, key) in zip(y, rows):
        full = label == "All centers"
        ax.plot([lo, hi], [yi, yi], color=COLOR["ssl_colon"], linewidth=1.6, solid_capstyle="butt")
        ax.plot(est, yi, marker="D" if full else "o", markersize=5, color=COLOR["ssl_colon"], markeredgewidth=0.8,
                markerfacecolor=COLOR["ssl_colon"] if full else "white")
        record("Figure 4", "ΔAUROC SSL-colon − SUP-imnet, 95% CI (t-test on partition means)", label, None, est, lo, hi, src, key)
    ax.axvline(0, color=GREY, linewidth=0.6, zorder=0)
    ax.axhline(y[0] - 0.5, color=GREY, linewidth=0.5, linestyle=":")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Paired difference in zero-shot AUROC (SSL-colon − SUP-imnet)")
    save(fig, "figure4_e3_loco")


def figure5(t8: pd.DataFrame) -> None:
    src = "results_v3/analysis/tables/table8_e4_arms.csv"
    t8 = t8.copy()
    t8["budget"] = t8["budget"].astype(int)
    budgets = sorted(t8["budget"].unique())
    ymin = t8["balanced_accuracy_mean"].min()
    ymax = t8["balanced_accuracy_mean"].max()
    pad = 0.04 * (ymax - ymin)
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH_IN, 68 * MM), layout="constrained", sharey=True)
    for ax, g, letter in zip(axes, REGIMES, "abcd"):
        for arm in ARMS:
            s = t8[(t8["regime"] == g) & (t8["arm"] == arm)].sort_values("budget")
            ls, mk = ARM_STYLE[arm]
            ax.plot(s["budget"], s["balanced_accuracy_mean"], color=COLOR[g], linestyle=ls, marker=mk, markersize=3.5,
                    linewidth=1.0, label=ARMS[arm])
            for b, v in zip(s["budget"], s["balanced_accuracy_mean"]):
                record("Figure 5", "balanced accuracy mean", f"{LABEL[g]} arm {arm}", int(b), v, source=src,
                       source_key=f"budget={b},arm={arm},regime={g},balanced_accuracy_mean")
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.minorticks_off()
        ax.set_xlim(budgets[0] / 1.25, budgets[-1] * 1.25)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel("Support budget (frames)")
        ax.text(0.03, 0.98, f"{letter}  {LABEL[g]}", transform=ax.transAxes, va="top", fontsize=8, color=BLACK)
    axes[0].set_ylabel("Balanced accuracy")
    handles = [Line2D([], [], color=BLACK, linestyle=ARM_STYLE[a][0], marker=ARM_STYLE[a][1], markersize=3.5,
                      linewidth=1.0, label=ARMS[a]) for a in ARMS]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, handlelength=2.6, columnspacing=2.0)
    save(fig, "figure5_e4_arms")


def figure6(c: pd.DataFrame) -> None:
    src = "results_v3/analysis_exploratory_lr_common/contrasts_common_vs_v3.csv"
    c = c[c["base_id"].isin(["H1", "H3", "H4"]) & (c["hypothesis"] == c["base_id"])].set_index("base_id").loc[["H1", "H3", "H4"]]
    labels = {"H1": HYP_SHORT["H1"], "H3": HYP_SHORT["H3"], "H4": HYP_SHORT["H4"]}
    conds = [("v3", "Selected learning rate per regime", BLACK, "o", "black"),
             ("common", r"Shared learning rate ($1\times10^{-3}$)", GREY, "s", "white")]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 58 * MM), layout="constrained")
    base_y = np.arange(len(c))[::-1] * 1.0
    for yb, (hid, r) in zip(base_y, c.iterrows()):
        for j, (pre, name, color, mk, face) in enumerate(conds):
            yi = yb + (0.17 if j == 0 else -0.17)
            ax.plot([r[f"{pre}_t_ci_lo"], r[f"{pre}_t_ci_hi"]], [yi, yi], color=color, linewidth=3.0, solid_capstyle="butt")
            ax.plot([r[f"{pre}_boot_lo"], r[f"{pre}_boot_hi"]], [yi, yi], color=color, linewidth=0.9, solid_capstyle="butt")
            ax.plot(r[f"{pre}_estimate"], yi, marker=mk, markersize=5, color=color, markeredgewidth=0.9,
                    markerfacecolor=color if face != "white" else "white")
            record("Figure 6", f"{name}: estimate, 95% CI t-test / bootstrap", hid, None, r[f"{pre}_estimate"],
                   r[f"{pre}_t_ci_lo"], r[f"{pre}_t_ci_hi"], src, f"{hid}.{pre}_estimate/{pre}_t_ci")
            record("Figure 6", f"{name}: bootstrap CI", hid, None, None, r[f"{pre}_boot_lo"], r[f"{pre}_boot_hi"],
                   src, f"{hid}.{pre}_boot")
    ax.axvline(0, color=GREY, linewidth=0.6, zorder=0)
    ax.set_yticks(base_y, [labels[h] for h in c.index])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Paired difference")
    handles = [Line2D([], [], color=col, marker=mk, markersize=5, markerfacecolor=col if face != "white" else "white",
                      linewidth=3.0, label=name) for _, name, col, mk, face in conds]
    handles.append(Line2D([], [], color=BLACK, linewidth=0.9, label="Thin bar: cluster-bootstrap 95% CI"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, handlelength=1.8, columnspacing=1.2)
    save(fig, "figure6_exploratory_lr")


def verify(h, t3, loco, t8, c) -> pd.DataFrame:
    """Compare every plotted value with the manuscript tables and with independent analysis outputs."""

    checks = []

    def add(figure, item, plotted, reference, ref_source):
        diff = abs(float(plotted) - float(reference))
        checks.append({"figure": figure, "item": item, "plotted": plotted, "reference": reference,
                       "reference_source": ref_source, "abs_diff": diff, "match": diff == 0.0})

    cs = pd.read_csv(ANALYSIS / "contrast_summary.csv")
    rm = pd.read_csv(ANALYSIS / "run_metrics.csv")
    pdiff = pd.read_csv(ANALYSIS / "paired_differences.csv")
    hi = h.set_index("hypothesis")
    for hid in HYP_SHORT:
        row = cs[cs["id"] == hid].iloc[0]
        for col_plot, col_cs in (("estimate", "mean"), ("model_ci_lo", "model_ci_lo"), ("model_ci_hi", "model_ci_hi"),
                                 ("boot_lo", "boot_lo"), ("boot_hi", "boot_hi")):
            add("Figure 2", f"{hid} {col_plot}", hi.loc[hid, col_plot], row[col_cs], "contrast_summary.csv")
        add("Figure 2", f"{hid} estimate = mean of per-run differences", hi.loc[hid, "estimate"],
            pdiff[pdiff["id"] == hid]["d"].mean(), "paired_differences.csv (mean of d)")
    e1 = rm[(rm["family"] == "e1") & (rm["slot"] == "all")]
    for _, r in t3.iterrows():
        runs = e1[(e1["regime"] == r["regime"]) & (e1["variant"].astype(int) == int(r["percent"]))]
        for metric in ("auprc", "auroc"):
            add("Figure 3", f"{r['regime']} p{int(r['percent'])} {metric} mean", r[f"{metric}_mean"], runs[metric].mean(),
                "run_metrics.csv (mean of 15 runs)")
            pm = runs.groupby("k")[metric].mean()
            add("Figure 3", f"{r['regime']} p{int(r['percent'])} {metric} SD of partition means",
                r[f"{metric}_sd_partition"], pm.std(ddof=1), "run_metrics.csv (SD of 5 partition means)")
    lo = loco[(loco["comparison"] == "ssl_colon - sup_imnet") & (loco["metric"] == "auroc")]
    for _, r in lo.iterrows():
        row = cs[(cs["analysis"] == "E3 leave-one-centre-out") & (cs["comparison"] == r["comparison"])
                 & (cs["metric"] == "auroc") & (cs["excluded"] == r["excluded"])]
        for col in ("mean", "model_ci_lo", "model_ci_hi"):
            add("Figure 4", f"without {r['excluded']} {col}", r[col], row.iloc[0][col], "contrast_summary.csv")
    e4 = rm[rm["family"] == "e4"]
    for _, r in t8.iterrows():
        runs = e4[(e4["regime"] == r["regime"]) & (e4["variant"].astype(int) == int(r["budget"])) & (e4["slot"] == r["arm"])]
        add("Figure 5", f"{r['regime']} s{int(r['budget'])} arm {r['arm']} balanced accuracy mean",
            r["balanced_accuracy_mean"], runs["balanced_accuracy"].mean(), "run_metrics.csv (mean of 15 runs)")
    cc = c.set_index("base_id")
    for hid in ("H1", "H3", "H4"):
        for col_c, col_h in (("v3_estimate", "estimate"), ("v3_t_ci_lo", "model_ci_lo"), ("v3_t_ci_hi", "model_ci_hi"),
                             ("v3_boot_lo", "boot_lo"), ("v3_boot_hi", "boot_hi")):
            add("Figure 6", f"{hid} {col_c}", cc.loc[hid, col_c], hi.loc[hid, col_h], "table_hypotheses.csv")
    return pd.DataFrame(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", type=int, choices=[2, 3, 4, 5, 6],
                        help="regenerate only these figures; their rows in plotted_values.csv are replaced")
    args = parser.parse_args()
    selected = set(args.only or [2, 3, 4, 5, 6])
    h = pd.read_csv(TABLES / "table_hypotheses.csv")
    t3 = pd.read_csv(TABLES / "table3_e1_performance.csv")
    loco = pd.read_csv(TABLES / "table_loco_e3.csv")
    t8 = pd.read_csv(TABLES / "table8_e4_arms.csv")
    c = pd.read_csv(EXPLORATORY / "contrasts_common_vs_v3.csv")
    jobs = {2: lambda: figure2(h), 3: lambda: figure3(t3), 4: lambda: figure4(loco, h), 5: lambda: figure5(t8),
            6: lambda: figure6(c)}
    for number in sorted(selected):
        jobs[number]()
    plotted = pd.DataFrame(ROWS)
    if args.only and (OUT / "plotted_values.csv").exists():
        kept = pd.read_csv(OUT / "plotted_values.csv")
        kept = kept[~kept["figure"].isin([f"Figure {n}" for n in selected])]
        plotted = pd.concat([kept, plotted], ignore_index=True)
        plotted["_order"] = plotted["figure"].str.extract(r"(\d+)").astype(int)
        plotted = plotted.sort_values("_order", kind="stable").drop(columns="_order")
    plotted.to_csv(OUT / "plotted_values.csv", index=False, float_format="%.17g")
    checks = verify(h, t3, loco, t8, c)
    checks.to_csv(OUT / "verification.csv", index=False, float_format="%.17g")
    worst = checks["abs_diff"].max()
    print(f"font: {FONT}; figures: {sorted(selected)}; plotted values: {len(plotted)}; checks: {len(checks)}, exact matches: {int(checks['match'].sum())}, "
          f"max |diff| {worst:.3g}")
    tol = checks[checks["abs_diff"] > 1e-12]
    if not tol.empty:
        print(tol.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
