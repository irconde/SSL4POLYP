"""Contrast definitions: H1–H7 (SAP section 3) and the secondary comparisons (section 6)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .design import Contrast, RunId, Term, View, ViewKey, aulc_weights, complete_terms
from .sap import (
    ARM_CONTRASTS, CALIBRATION, COMPARATORS, E1_PERCENTS, E4_ARMS, E4_BUDGETS, LABEL, MATCHED, METRICS,
    NEGSOURCE_BUDGET, PER_POLYP, THRESHOLD_FREE,
)

COLON = "ssl_colon"
SUP = "sup_imnet"
SECONDARY_METRICS = METRICS + CALIBRATION


def _runs(partitions: Sequence[int], seeds: Sequence[int]) -> List[RunId]:
    return [(k, s) for k in partitions for s in seeds]


def _diff(treat: ViewKey, base: ViewKey, metric: str) -> List[Term]:
    return [(treat, metric, 1.0), (base, metric, -1.0)]


# --------------------------------------------------------------------------- hypotheses
def hypothesis_contrasts(
    views: Dict[ViewKey, View],
    regimes: Sequence[str],
    partitions: Sequence[int],
    seeds: Sequence[int],
    budgets: Sequence[int] = E4_BUDGETS,
    prefix: str = "",
) -> Tuple[Dict[str, Optional[Contrast]], Dict[str, str]]:
    """Return ``{H: contrast or None}`` and ``{H: note}`` (e.g. H2 pending the matched regime)."""

    runs = _runs(partitions, seeds)
    w = dict(zip(E1_PERCENTS, aulc_weights(E1_PERCENTS)))
    out: Dict[str, Optional[Contrast]] = {}
    notes: Dict[str, str] = {}

    def aulc_diff(base: str):
        def make(k: int, s: int) -> List[Term]:
            terms: List[Term] = []
            for p in E1_PERCENTS:
                terms.append((("e1", p, COLON, k, s, "all"), "auprc", w[p]))
                terms.append((("e1", p, base, k, s, "all"), "auprc", -w[p]))
            return terms

        return make

    def contrast(h: str, family: str, make, meta: Dict[str, Any]) -> Optional[Contrast]:
        terms = complete_terms(views, runs, make)
        if not terms:
            return None
        return Contrast(f"{prefix}{h}", family, {"hypothesis": h, **meta}, terms)

    out["H1"] = contrast("H1", "e1", aulc_diff(SUP), {"metric": "AULC(auprc)", "comparison": "ssl_colon - sup_imnet"})
    if MATCHED in regimes:
        out["H2"] = contrast("H2", "e1", aulc_diff(MATCHED),
                             {"metric": "AULC(auprc)", "comparison": "ssl_colon - ssl_imnet_matched"})
    else:
        out["H2"] = None
        notes["H2"] = "pending: no SSL-imnet-matched results found"
    out["H3"] = contrast("H3", "e2", lambda k, s: _diff(("e2", None, COLON, k, s, "flat"), ("e2", None, SUP, k, s, "flat"), "auroc"),
                         {"metric": "auroc", "comparison": "ssl_colon - sup_imnet", "stratum": "flat"})
    out["H4"] = contrast("H4", "e3", lambda k, s: _diff(("e3", None, COLON, k, s, "all"), ("e3", None, SUP, k, s, "all"), "auroc"),
                         {"metric": "auroc", "comparison": "ssl_colon - sup_imnet"})

    n = len(regimes)

    def h5(k: int, s: int) -> List[Term]:
        terms: List[Term] = []
        for g in regimes:
            terms.append((("neg", NEGSOURCE_BUDGET, g, k, s, "sequence"), "auroc", 1.0 / n))
            terms.append((("neg", NEGSOURCE_BUDGET, g, k, s, "isolated"), "auroc", -1.0 / n))
        return terms

    out["H5"] = contrast("H5", "neg", h5, {"metric": "auroc", "comparison": "sequence - isolated", "regimes": ",".join(regimes)})

    def arms(treat: str, base: str, bud: Sequence[int]):
        m = len(regimes) * len(bud)

        def make(k: int, s: int) -> List[Term]:
            terms: List[Term] = []
            for g in regimes:
                for b in bud:
                    terms.append((("e4", b, g, k, s, treat), "balanced_accuracy", 1.0 / m))
                    terms.append((("e4", b, g, k, s, base), "balanced_accuracy", -1.0 / m))
            return terms

        return make

    bud_txt = ",".join(f"s{b}" for b in budgets)
    out["H6"] = contrast("H6", "e4", arms("B", "A", budgets),
                         {"metric": "balanced_accuracy", "comparison": "arm B - arm A", "regimes": ",".join(regimes), "budgets": bud_txt})
    out["H7"] = contrast("H7", "e4", arms("C", "A", budgets),
                         {"metric": "balanced_accuracy", "comparison": "arm C - arm A", "regimes": ",".join(regimes), "budgets": bud_txt})
    if MATCHED not in regimes:
        for h in ("H5", "H6", "H7"):
            notes[h] = f"mean over {len(regimes)} regimes (SSL-imnet-matched pending)"
    for h, c in out.items():
        if c is None and h not in notes:
            notes[h] = "no complete paired runs"
    return out, notes


# --------------------------------------------------------------------------- secondary (section 6)
def secondary_contrasts(
    views: Dict[ViewKey, View],
    regimes: Sequence[str],
    partitions: Sequence[int],
    seeds: Sequence[int],
) -> List[Contrast]:
    runs = _runs(partitions, seeds)
    comparators = [g for g in COMPARATORS if g in regimes]
    out: List[Contrast] = []

    def add(cid: str, family: str, meta: Dict[str, Any], make) -> None:
        terms = complete_terms(views, runs, make)
        if terms:
            out.append(Contrast(cid, family, meta, terms))

    # 1. SSL-colon vs each comparator, per experiment / budget / stratum / arm / metric
    slots: List[Tuple[str, Optional[int], str, Tuple[str, ...]]] = [("sun_full", None, "all", SECONDARY_METRICS + PER_POLYP)]
    slots += [("e1", p, "all", SECONDARY_METRICS + PER_POLYP) for p in E1_PERCENTS]
    slots += [("e2", None, st, SECONDARY_METRICS + PER_POLYP) for st in ("overall", "flat", "polypoid")]
    slots += [("e3", None, "all", SECONDARY_METRICS)]
    slots += [("e4", b, arm, SECONDARY_METRICS) for b in E4_BUDGETS for arm in E4_ARMS]
    slots += [("neg", NEGSOURCE_BUDGET, src, ("auroc", "auprc", "specificity")) for src in ("isolated", "sequence")]
    for fam, variant, slot, metrics in slots:
        for base in comparators:
            for metric in metrics:
                cid = f"cmp|{fam}|{variant}|{slot}|{COLON}-{base}|{metric}"
                meta = {"analysis": "regime comparison", "experiment": fam, "variant": variant, "slot": slot,
                        "comparison": f"{COLON} - {base}", "metric": metric}
                add(cid, fam, meta, lambda k, s, fam=fam, variant=variant, slot=slot, base=base, metric=metric:
                    _diff((fam, variant, COLON, k, s, slot), (fam, variant, base, k, s, slot), metric))

    # AULC per metric (E1), SSL-colon vs comparators and per regime
    w = dict(zip(E1_PERCENTS, aulc_weights(E1_PERCENTS)))
    for metric in SECONDARY_METRICS + PER_POLYP:
        for base in comparators:
            def make(k, s, base=base, metric=metric):
                terms: List[Term] = []
                for p in E1_PERCENTS:
                    terms += [(("e1", p, COLON, k, s, "all"), metric, w[p]), (("e1", p, base, k, s, "all"), metric, -w[p])]
                return terms

            add(f"aulc|{COLON}-{base}|{metric}", "e1",
                {"analysis": "AULC comparison", "experiment": "e1", "comparison": f"{COLON} - {base}", "metric": f"AULC({metric})"}, make)
        for g in regimes:
            add(f"aulc_level|{g}|{metric}", "e1",
                {"analysis": "AULC level", "experiment": "e1", "regime": g, "metric": f"AULC({metric})"},
                lambda k, s, g=g, metric=metric: [(("e1", p, g, k, s, "all"), metric, w[p]) for p in E1_PERCENTS])

    # 2. E4: the four arms per budget, arm contrasts per regime and pooled over regimes
    for b in E4_BUDGETS:
        for treat, base in ARM_CONTRASTS:
            same_scores = {treat, base} in ({"A", "B"}, {"C", "D"})
            for metric in SECONDARY_METRICS:
                if same_scores and metric in THRESHOLD_FREE:
                    continue  # identical probabilities: threshold-free metrics differ by 0 by construction
                for g in regimes:
                    add(f"arm|{b}|{g}|{treat}-{base}|{metric}", "e4",
                        {"analysis": "E4 arm contrast", "experiment": "e4", "variant": b, "regime": g,
                         "comparison": f"arm {treat} - arm {base}", "metric": metric},
                        lambda k, s, b=b, g=g, treat=treat, base=base, metric=metric:
                        _diff(("e4", b, g, k, s, treat), ("e4", b, g, k, s, base), metric))
                n = len(regimes)
                add(f"arm|{b}|pooled|{treat}-{base}|{metric}", "e4",
                    {"analysis": "E4 arm contrast", "experiment": "e4", "variant": b, "regime": "mean of regimes",
                     "comparison": f"arm {treat} - arm {base}", "metric": metric},
                    lambda k, s, b=b, treat=treat, base=base, metric=metric, n=n:
                    [t for g in regimes for t in [(("e4", b, g, k, s, treat), metric, 1.0 / n), (("e4", b, g, k, s, base), metric, -1.0 / n)]])

    # 3. E2: flat − polypoid AUROC gap per regime and its difference between regimes
    for metric in ("auroc",):
        for g in regimes:
            add(f"gap|{g}|{metric}", "e2",
                {"analysis": "flat - polypoid gap", "experiment": "e2", "regime": g, "comparison": "flat - polypoid", "metric": metric},
                lambda k, s, g=g, metric=metric: _diff(("e2", None, g, k, s, "flat"), ("e2", None, g, k, s, "polypoid"), metric))
        for base in comparators:
            add(f"gapdiff|{COLON}-{base}|{metric}", "e2",
                {"analysis": "flat - polypoid gap", "experiment": "e2", "comparison": f"gap {COLON} - gap {base}", "metric": metric},
                lambda k, s, base=base, metric=metric:
                _diff(("e2", None, COLON, k, s, "flat"), ("e2", None, COLON, k, s, "polypoid"), metric)
                + [(("e2", None, base, k, s, "flat"), metric, -1.0), (("e2", None, base, k, s, "polypoid"), metric, 1.0)])

    # 4. Negative-source decomposition per regime (sequence − isolated)
    for g in regimes:
        for metric in ("auroc", "specificity", "balanced_accuracy"):
            add(f"negsrc|{g}|{metric}", "neg",
                {"analysis": "negative-source decomposition", "experiment": "neg", "regime": g,
                 "comparison": "sequence - isolated", "metric": metric},
                lambda k, s, g=g, metric=metric:
                _diff(("neg", NEGSOURCE_BUDGET, g, k, s, "sequence"), ("neg", NEGSOURCE_BUDGET, g, k, s, "isolated"), metric))
    return out


def sensitivity_contrasts(
    views: Dict[ViewKey, View], regimes: Sequence[str], partitions: Sequence[int], seeds: Sequence[int]
) -> List[Contrast]:
    """H1–H4 on partition k = 1 only; H6 and H7 without s500 (section 6, sensitivity analyses)."""

    out: List[Contrast] = []
    if 1 in partitions:
        hyp, _ = hypothesis_contrasts(views, regimes, [1], seeds, prefix="k1|")
        for h in ("H1", "H2", "H3", "H4"):
            c = hyp.get(h)
            if c is not None:
                c.meta.update(analysis="k = 1 only")
                out.append(c)
    budgets = [b for b in E4_BUDGETS if b != 500]
    hyp, _ = hypothesis_contrasts(views, regimes, partitions, seeds, budgets=budgets, prefix="no_s500|")
    for h in ("H6", "H7"):
        c = hyp.get(h)
        if c is not None:
            c.meta.update(analysis="without s500")
            out.append(c)
    return out


def describe(c: Contrast) -> str:
    m = c.meta
    parts = [str(m.get(k)) for k in ("analysis", "experiment", "variant", "slot", "regime", "comparison", "metric") if m.get(k) not in (None, "")]
    return " | ".join(parts)


def regime_label(g: str) -> str:
    return LABEL.get(g, g)
