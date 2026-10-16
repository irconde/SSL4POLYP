"""Simulated campaign v3 results with the structure of the real ones and known effects.

Structure (as in ``checkpoints_v3/campaign/classification``): regimes × partitions × seeds; sun_full, E1
(5 budgets), E2, E3 and E4 (4 budgets, arms A–D) run directories with ``RUN_COMPLETE.json``, the metrics
JSONs (only the keys the analysis reads: operating thresholds and run identity) and per-frame CSVs with the
spec 1.3 columns. Split sizes follow the real packs: SUN test/val 990 frames (15 positive cases, 5 flat and
10 polypoid, and 15 negative virtual cases of 33 frames); polypgen_clean_test 1,438 frames (1,254 positives
and 184 isolated negatives over C1–C6); 3,190 sequence negatives in 18 sequences; few-shot packs per
(partition, seed) with support 50/100/200/500 (train/val 46/4, 86/14, 160/40, 400/100) and evaluation
splits built as in ``polypgen_fewshot`` (s500: all isolated negatives go to the support and the evaluation
split keeps whole sequences only). SSL-imnet-matched can be written under a separate root.

Score model (latent scale): ``z = δ·y [+ m for sequence negatives] + cluster effect + N(0, 1)`` and
``prob = sigmoid(z + offset)``. Cluster effects are properties of the test frames, drawn once per simulated
campaign and shared by all models: SUN case (σ 0.5), PolypGen centre (σ 0.3), negative sequence (σ 0.4).
Each run's separation (δ, m), offset and E4 thresholds are solved so that the value of the hypothesis
metric on the realised test population (the mixture over its clusters, i.e. its expectation over the frame
noise) equals a target; the targets are

    target(treatment) = target(reference) + effect + v_k + w_{k,s} (+ small regime/budget noise),

with ``v_k ~ N(0, sd_partition)`` and ``w_{k,s} ~ N(0, sd_seed)``. The true μ of every hypothesis is
therefore the effect itself; ``truth.json`` also stores the realised per-run values. AUROC and balanced
accuracy of the mixtures are exact (normal CDFs); AP is integrated numerically. Stored E4 thresholds of
arms B and D are the solved ones (not refitted on the support CSVs); the analysis reads thresholds from the
JSONs, as for the real campaign. Targets outside the attainable range are clipped and logged
(``clipped_targets``); a valid simulation has none.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit, ndtr


def _cdf(x):
    return ndtr(x)


def _sf(x):
    return ndtr(-x)


def _pdf(x):
    return np.exp(-0.5 * np.square(x)) / np.sqrt(2 * np.pi)

from .loader import FRAME_COLUMNS
from .sap import E1_PERCENTS, E4_BUDGETS, HYPOTHESES, MATCHED, PARTITIONS, SEEDS

TAG = {"sup_imnet": "SUPImNet", "ssl_imnet": "SSLImNet", "ssl_colon": "SSLColon", "ssl_imnet_matched": "SSLImNetMatched"}
SIGMA_CASE, SIGMA_CENTER, SIGMA_SEQ = 0.5, 0.3, 0.4
CENTER_POS = {"C1": 250, "C2": 300, "C3": 200, "C4": 180, "C5": 170, "C6": 154}   # 1,254
CENTER_NEG = {"C1": 40, "C2": 35, "C3": 30, "C4": 30, "C5": 25, "C6": 24}         # 184
SEQ_SIZES = [120, 150, 160, 170, 180, 175, 165, 185, 190, 200, 155, 145, 210, 170, 180, 195, 160, 280]  # 3,190
SEQ_IDS = [str(i) for i in (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 16, 18, 20, 21, 22, 23)]
SUPPORT = {50: (46, 4), 100: (86, 14), 200: (160, 40), 500: (400, 100)}
FRAMES_PER_CASE = 33


@dataclass
class Scenario:
    name: str
    effects: Dict[str, float]
    sd_partition: Dict[str, float]
    sd_seed: Dict[str, float]
    include_matched: bool = True
    matched_separate_root: bool = True
    partitions: Tuple[int, ...] = PARTITIONS
    seeds: Tuple[int, ...] = SEEDS
    noise: float = 0.004          # regime/budget-specific noise added to the treatment targets
    level_sd: float = 0.01        # partition and seed noise of the reference levels (cancels in differences)

    @property
    def regimes(self) -> Tuple[str, ...]:
        base = ("sup_imnet", "ssl_imnet", "ssl_colon")
        return base + ((MATCHED,) if self.include_matched else ())


def null_scenario(sd_partition: float = 0.008, sd_seed: float = 0.008, **kw) -> Scenario:
    return Scenario("null", {h: 0.0 for h in HYPOTHESES}, {h: sd_partition for h in HYPOTHESES},
                    {h: sd_seed for h in HYPOTHESES}, **kw)


EFFECTS = {"H1": 0.04, "H2": 0.03, "H3": 0.07, "H4": 0.05, "H5": -0.08, "H6": 0.04, "H7": 0.08}


def effect_scenario(sd_partition: float = 0.008, sd_seed: float = 0.008, **kw) -> Scenario:
    return Scenario("effects", dict(EFFECTS), {h: sd_partition for h in HYPOTHESES}, {h: sd_seed for h in HYPOTHESES}, **kw)


# --------------------------------------------------------------------------- population metrics
# Conditional on the realised cluster effects of the test frames (they are fixed for a campaign, as the
# real test sets are): frame scores are N(cluster offset + δ·y [+ m for sequence negatives], 1).
_T = np.linspace(-14.0, 18.0, 6401)
_TRAP = getattr(np, "trapezoid", None) or np.trapz


class Pop:
    """Mixture population of one split: positive and negative components (offset, weight)."""

    def __init__(self, pos_off, pos_w, neg_off, neg_w, neg_seq):
        self.pos_off, self.neg_off = np.asarray(pos_off, float), np.asarray(neg_off, float)
        self.pos_w = np.asarray(pos_w, float) / np.sum(pos_w)
        self.neg_w = np.asarray(neg_w, float) / np.sum(neg_w)
        self.neg_seq = np.asarray(neg_seq, float)
        self.pi = float(np.sum(pos_w) / (np.sum(pos_w) + np.sum(neg_w)))

    @classmethod
    def from_frames(cls, df: pd.DataFrame, offset_of: Dict[str, float], key_col: "pd.Series") -> "Pop":
        keys = key_col.to_numpy()
        y = df["label"].to_numpy()
        seq = (df["negative_source"].to_numpy() == "sequence")
        pos = pd.Series(keys[y == 1]).value_counts()
        neg = pd.DataFrame({"k": keys[y == 0], "seq": seq[y == 0]}).value_counts()
        return cls([offset_of[k] for k in pos.index], pos.to_numpy(),
                   [offset_of[k] for k, _ in neg.index], neg.to_numpy(), [float(sq) for _, sq in neg.index])

    def _means(self, delta: float, m: float):
        return self.pos_off + delta, self.neg_off + m * self.neg_seq

    def auroc(self, delta: float, m: float = 0.0) -> float:
        mp, mn = self._means(delta, m)
        return float(self.pos_w @ _cdf((mp[:, None] - mn[None, :]) / np.sqrt(2)) @ self.neg_w)

    def ba(self, t: float, delta: float, m: float = 0.0) -> float:
        mp, mn = self._means(delta, m)
        return float(0.5 * (self.pos_w @ _sf(t - mp) + self.neg_w @ _cdf(t - mn)))

    def ba_grid(self, ts: np.ndarray, delta: float, m: float = 0.0) -> np.ndarray:
        mp, mn = self._means(delta, m)
        return 0.5 * (_sf(ts[:, None] - mp[None, :]) @ self.pos_w + _cdf(ts[:, None] - mn[None, :]) @ self.neg_w)

    def ap(self, delta: float, m: float = 0.0) -> float:
        mp, mn = self._means(delta, m)
        tpr = _sf(_T[:, None] - mp[None, :]) @ self.pos_w
        fpr = _sf(_T[:, None] - mn[None, :]) @ self.neg_w
        dens = _pdf(_T[:, None] - mp[None, :]) @ self.pos_w
        prec = self.pi * tpr / np.maximum(self.pi * tpr + (1 - self.pi) * fpr, 1e-300)
        return float(_TRAP(prec * dens, _T))


class Clip:
    """Targets outside the attainable range (they are clipped; a valid simulation has none)."""

    log: List[str] = []


def _clip(target: float, lo: float, hi: float, what: str) -> float:
    if not lo <= target <= hi:
        Clip.log.append(f"{what}: target {target:.4f} outside [{lo:.4f}, {hi:.4f}]")
        return min(max(target, lo), hi)
    return target


def solve_delta(f, target: float, what: str, lo: float = -2.0, hi: float = 12.0) -> float:
    target = _clip(target, f(lo) + 1e-9, f(hi) - 1e-9, what)
    return brentq(lambda d: f(d) - target, lo, hi, xtol=1e-11)


def solve_ba_right(pop: Pop, target: float, delta: float, m: float, what: str) -> float:
    """Threshold on the high-threshold branch where the population balanced accuracy equals ``target``."""

    grid = np.linspace(-6.0, delta + 8.0, 1401)
    vals = pop.ba_grid(grid, delta, m)
    t_max = float(grid[int(vals.argmax())])
    t_hi = t_max + 25.0
    target = _clip(target, pop.ba(t_hi, delta, m) + 1e-6, float(vals.max()) - 1e-6, what)
    return brentq(lambda t: pop.ba(t, delta, m) - target, t_max, t_hi, xtol=1e-11)

def f1_opt_tau(probs: np.ndarray, labels: np.ndarray) -> float:
    """F1-optimal τ over the unique scores (prediction = prob ≥ τ); ties: higher recall, then lower τ."""

    order = np.argsort(-probs, kind="mergesort")
    p, y = probs[order], labels[order]
    uniq, first = np.unique(-p, return_index=True)
    last = np.r_[first[1:], p.size] - 1
    tp = np.cumsum(y)[last]
    fp = (np.arange(p.size) + 1)[last] - tp
    fn = y.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    rec = tp[best]
    best = best[rec == rec.max()]
    return float(-uniq[best].max())  # among ties the lowest τ (= highest -uniq)


# --------------------------------------------------------------------------- frame tables
def _frames(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[c for c in FRAME_COLUMNS if c not in ("prob", "pred")])


def sun_pools(rng: np.random.Generator):
    pos = [{"case_id": f"SUN_{i:04d}", "morphology": "flat" if i % 10 < 3 else "polypoid"} for i in range(1, 101)]
    neg = []
    for i in range(100):
        seq = f"NEG_SUN_case{i % 13 + 1}"
        neg.append({"case_id": f"{seq}_v{i:03d}", "sequence": seq})
    return pos, neg


def sun_split(pos_cases: List[dict], neg_cases: List[dict]) -> pd.DataFrame:
    rows = []
    for c in pos_cases:
        for j in range(FRAMES_PER_CASE):
            rows.append({"frame_id": f"{c['case_id']}/images/{c['case_id']}.{j * 7 + 1:06d}.jpg", "label": 1, "case_id": c["case_id"],
                         "sequence_id": c["case_id"], "center_id": "", "origin": "sun", "morphology": c["morphology"], "negative_source": "NA"})
    for c in neg_cases:
        for j in range(FRAMES_PER_CASE):
            rows.append({"frame_id": f"{c['sequence']}/images/{c['case_id']}.{j:06d}.jpg", "label": 0, "case_id": c["case_id"],
                         "sequence_id": c["sequence"], "center_id": "", "origin": "sun", "morphology": "", "negative_source": "NA"})
    return _frames(rows)


def partition_splits(k: int, pos: List[dict], neg: List[dict]) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(9000 + k)
    flat = [c for c in pos if c["morphology"] == "flat"]
    poly = [c for c in pos if c["morphology"] == "polypoid"]
    fi, pi_, ni = rng.permutation(len(flat)), rng.permutation(len(poly)), rng.permutation(len(neg))
    test_pos = [flat[i] for i in fi[:5]] + [poly[i] for i in pi_[:10]]
    rest = [flat[i] for i in fi[5:]] + [poly[i] for i in pi_[10:]]
    val_pos = [rest[i] for i in rng.permutation(len(rest))[:15]]
    return {"test": sun_split(test_pos, [neg[i] for i in ni[:15]]), "val": sun_split(val_pos, [neg[i] for i in ni[15:30]])}


def polypgen_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(424242)
    rows = []
    for c, n in CENTER_POS.items():
        rows += [{"label": 1, "center_id": c, "negative_source": "NA"} for _ in range(n)]
    for c, n in CENTER_NEG.items():
        rows += [{"label": 0, "center_id": c, "negative_source": "isolated"} for _ in range(n)]
    rows = [rows[i] for i in rng.permutation(len(rows))]
    clean = _frames([{"frame_id": f"POLYPGEN_NOCASE.{i + 1:06d}", "case_id": "", "sequence_id": "", "origin": "polypgen_clean",
                      "morphology": "", **r} for i, r in enumerate(rows)])
    seq_rows = []
    n = 0
    for sid, size in zip(SEQ_IDS, SEQ_SIZES):
        for _ in range(size):
            n += 1
            seq_rows.append({"frame_id": f"POLYPGEN_SEQ{sid}.{n:06d}", "label": 0, "case_id": "", "sequence_id": sid, "center_id": "",
                             "origin": "polypgen_clean", "morphology": "", "negative_source": "sequence"})
    return clean, _frames(seq_rows)


def fewshot_pack(clean: pd.DataFrame, seqs: pd.DataFrame, k: int, s: int, b: int) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(1000 * k + s + 7 * b)
    n_train, n_val = SUPPORT[b]
    half = b // 2
    pos_idx = rng.permutation(np.flatnonzero(clean["label"].to_numpy() == 1))
    iso_idx = rng.permutation(np.flatnonzero(clean["label"].to_numpy() == 0))
    sup_pos = clean.iloc[pos_idx[:half]]
    if b < 500:
        sup_neg = clean.iloc[iso_idx[:half]]
        eval_df = pd.concat([clean.iloc[pos_idx[half:]], clean.iloc[iso_idx[half:]], seqs])
    else:
        order = rng.permutation(SEQ_IDS)
        target = int(rng.integers(302, 1032))
        eval_seq, total, i = [], 0, 0
        while total < target and i < len(order):
            eval_seq.append(order[i])
            total += SEQ_SIZES[SEQ_IDS.index(order[i])]
            i += 1
        support_seq = seqs[seqs["sequence_id"] == order[i]].iloc[: half - len(iso_idx)]
        sup_neg = pd.concat([clean.iloc[iso_idx], support_seq])
        eval_df = pd.concat([clean.iloc[pos_idx[half:]], seqs[seqs["sequence_id"].isin(eval_seq)]])
    v_half = n_val // 2
    val = pd.concat([sup_pos.iloc[:v_half], sup_neg.iloc[:v_half]])
    train = pd.concat([sup_pos.iloc[v_half:], sup_neg.iloc[v_half:]])
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "eval": eval_df.reset_index(drop=True)}


# --------------------------------------------------------------------------- writing
def _write_csv(df: pd.DataFrame, probs: np.ndarray, tau: float, path: Path) -> None:
    out = df.copy()
    out.insert(1, "prob", probs)
    out.insert(3, "pred", (probs >= tau).astype(int))
    path.parent.mkdir(parents=True, exist_ok=True)
    out[list(FRAME_COLUMNS)].to_csv(path, index=False, float_format="%.9g")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))


def _complete(run_dir: Path, name: str) -> None:
    _write_json(run_dir / "RUN_COMPLETE.json", {"run": name, "simulated": True})


def _identity(exp: str, regime: str, k: int, s: int, stem: str) -> dict:
    return {"run": {"experiment": exp, "partition": k, "seed": s, "model": TAG[regime], "stem": stem},
            "provenance": {"model": regime, "train_seed": s}, "seed": s, "simulated": True}


# --------------------------------------------------------------------------- generator
class Generator:
    def __init__(self, scenario: Scenario, sim_seed: int):
        self.sc = scenario
        self.rng = np.random.default_rng(sim_seed)
        self.truth: Dict[str, Dict] = {h: {"effect": scenario.effects[h], "per_run": {}} for h in HYPOTHESES}
        self.terms: Dict[str, Dict[Tuple[int, int], List[float]]] = {h: {} for h in HYPOTHESES}

    def re(self, h: str) -> Dict[Tuple[int, int], float]:
        """Hypothesis-specific random effects v_k + w_{k,s} on the difference."""

        v = {k: self.rng.normal(0, self.sc.sd_partition[h]) if self.sc.sd_partition[h] > 0 else 0.0 for k in self.sc.partitions}
        return {(k, s): self.sc.effects[h] + v[k] + (self.rng.normal(0, self.sc.sd_seed[h]) if self.sc.sd_seed[h] > 0 else 0.0)
                for k in self.sc.partitions for s in self.sc.seeds}

    def lvl(self) -> Dict[Tuple[int, int], float]:
        u = {k: self.rng.normal(0, self.sc.level_sd) for k in self.sc.partitions}
        return {(k, s): u[k] + self.rng.normal(0, self.sc.level_sd) for k in self.sc.partitions for s in self.sc.seeds}

    def eta(self) -> float:
        return float(self.rng.normal(0, self.sc.noise))


def simulate_campaign(scenario: Scenario, out: Path, sim_seed: int = 0, minimal: bool = False) -> Dict[str, object]:
    """Write a simulated campaign under ``out`` and return the truth (also written to ``out/truth.json``).

    Main root: ``out/campaign/classification``; matched regime: ``out/campaign_matched/classification`` when
    ``matched_separate_root``. ``minimal`` skips the CSVs the analysis does not read (validation and support).
    """

    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    gen = Generator(scenario, sim_seed)
    rng = gen.rng
    sc = scenario
    regimes = sc.regimes
    Clip.log = []

    def root_for(regime: str) -> Path:
        if regime == MATCHED and sc.matched_separate_root:
            return out / "campaign_matched" / "classification"
        return out / "campaign" / "classification"

    pos_pool, neg_pool = sun_pools(rng)
    splits = {k: partition_splits(k, pos_pool, neg_pool) for k in sc.partitions}
    case_eff = {c["case_id"]: rng.normal(0, SIGMA_CASE) for c in pos_pool + neg_pool}
    clean, seqs = polypgen_tables()
    center_eff = {c: rng.normal(0, SIGMA_CENTER) for c in CENTER_POS}
    seq_eff = {sid: rng.normal(0, SIGMA_SEQ) for sid in SEQ_IDS}
    pg_all = pd.concat([clean, seqs]).reset_index(drop=True)
    pg_cluster = np.array([center_eff[c] if c else seq_eff[q] for c, q in zip(pg_all["center_id"], pg_all["sequence_id"])])
    pg_index = {f: i for i, f in enumerate(pg_all["frame_id"])}
    packs = {(k, s, b): fewshot_pack(clean, seqs, k, s, b) for k in sc.partitions for s in sc.seeds for b in E4_BUDGETS}
    runs = [(k, s) for k in sc.partitions for s in sc.seeds]

    def sun_pop(k: int, positives: str) -> Pop:
        df = splits[k]["test"]
        keep = (df["label"] == 0) | ((df["morphology"] == positives) if positives != "all" else True)
        df = df[keep]
        return Pop.from_frames(df, case_eff, df["case_id"])

    pg_offset = {**{c: e for c, e in center_eff.items()}, **{f"seq:{q}": e for q, e in seq_eff.items()}}

    def pg_pop(df: pd.DataFrame) -> Pop:
        key = df["center_id"].where(df["center_id"] != "", "seq:" + df["sequence_id"])
        return Pop.from_frames(df, pg_offset, key)

    def sun_scores(df: pd.DataFrame, delta_flat: float, delta_poly: float, offset: float) -> np.ndarray:
        y = df["label"].to_numpy()
        morph = df["morphology"].to_numpy()
        delta = np.where(y == 1, np.where(morph == "flat", delta_flat, delta_poly), 0.0)
        z = delta + np.array([case_eff[c] for c in df["case_id"]]) + rng.normal(0, 1, len(df))
        return expit(z + offset)

    # ---- sun_full and E2 (H3): flat/polypoid AUROC targets
    re3, lvl_f, lvl_p = gen.re("H3"), gen.lvl(), gen.lvl()
    sun_tau: Dict[Tuple[str, int, int], float] = {}
    sun_offset: Dict[Tuple[str, int, int], float] = {}
    pops = {k: {m: sun_pop(k, m) for m in ("flat", "polypoid", "all")} for k in sc.partitions}
    for (k, s) in runs:
        colon_flat, colon_poly = 0.88 + lvl_f[(k, s)], 0.95 + lvl_p[(k, s)]
        flat_t = {"ssl_colon": colon_flat, "sup_imnet": colon_flat - re3[(k, s)],
                  "ssl_imnet": colon_flat - 0.02 + gen.eta(), MATCHED: colon_flat - 0.03 + gen.eta()}
        poly_t = {"ssl_colon": colon_poly, "sup_imnet": colon_poly - 0.01 + gen.eta(),
                  "ssl_imnet": colon_poly - 0.005 + gen.eta(), MATCHED: colon_poly - 0.01 + gen.eta()}
        gen.truth["H3"]["per_run"][f"{k},{s}"] = flat_t["ssl_colon"] - flat_t["sup_imnet"]
        for g in regimes:
            df_f = solve_delta(pops[k]["flat"].auroc, flat_t[g], f"sun flat AUROC {g} k{k} s{s}")
            df_p = solve_delta(pops[k]["polypoid"].auroc, poly_t[g], f"sun polypoid AUROC {g} k{k} s{s}")
            offset = rng.normal(-1.0, 0.3)
            p_val = sun_scores(splits[k]["val"], df_f, df_p, offset)
            tau = f1_opt_tau(p_val, splits[k]["val"]["label"].to_numpy())
            p_test = sun_scores(splits[k]["test"], df_f, df_p, offset)
            sun_tau[(g, k, s)] = tau
            sun_offset[(g, k, s)] = offset
            stem = f"{TAG[g]}_SUNFull_s{s}"
            run_dir = root_for(g) / f"k{k}" / "sun_full" / f"{g}_seed{s}"
            d = run_dir / "sun_baselines"
            _write_csv(splits[k]["test"], p_test, tau, d / f"{stem}_test_outputs.csv")
            if not minimal:
                _write_csv(splits[k]["val"], p_val, tau, d / f"{stem}_val_outputs.csv")
            ident = _identity("sun_full", g, k, s, stem)
            _write_json(d / f"{stem}_last.metrics.json", {**ident, "test_primary": {"tau": tau},
                                                          "thresholds": {"primary": {"policy": "f1_opt_on_val", "tau": tau}}})
            _write_json(d / f"{stem}.metrics.json", {**ident})
            _complete(run_dir, stem)
            # E2: same model and τ on the same test split
            stem2 = f"{TAG[g]}_SUNMorph_s{s}"
            run2 = root_for(g) / f"k{k}" / "e2" / f"{g}_seed{s}"
            _write_csv(splits[k]["test"], p_test, tau, run2 / "sun_morphology" / f"{stem2}_test_outputs.csv")
            _write_json(run2 / "sun_morphology" / f"{stem2}.metrics.json",
                        {**_identity("e2", g, k, s, stem2), "eval_only": True, "test_primary": {"tau": tau}})
            _complete(run2, stem2)

    # ---- E1 (H1, H2): AP targets per budget; AULC of the difference = effect + v + w + noise
    w = np.array([2.5, 10, 20, 37.5, 25]) / 95.0  # trapezoid weights on 5,10,25,50,100
    base_ap = {5: 0.84, 10: 0.87, 25: 0.90, 50: 0.92, 100: 0.93}
    re1, re2 = gen.re("H1"), gen.re("H2")
    lvl1 = {p: gen.lvl() for p in E1_PERCENTS}
    for (k, s) in runs:
        d1, d2 = [], []
        for p in E1_PERCENTS:
            colon = base_ap[p] + lvl1[p][(k, s)]
            t = {"ssl_colon": colon, "sup_imnet": colon - re1[(k, s)] - gen.eta(),
                 "ssl_imnet": colon - 0.01 + gen.eta(), MATCHED: colon - re2[(k, s)] - gen.eta()}
            d1.append(t["ssl_colon"] - t["sup_imnet"])
            d2.append(t["ssl_colon"] - t[MATCHED])
            for g in regimes:
                delta = solve_delta(pops[k]["all"].ap, t[g], f"E1 AP {g} p{p} k{k} s{s}", lo=-1.0, hi=10.0)
                offset = rng.normal(-1.0, 0.3)
                p_val = sun_scores(splits[k]["val"], delta, delta, offset)
                tau = f1_opt_tau(p_val, splits[k]["val"]["label"].to_numpy())
                p_test = sun_scores(splits[k]["test"], delta, delta, offset)
                stem = f"{TAG[g]}_SUNP{p}_s{s}"
                run_dir = root_for(g) / f"k{k}" / "e1" / f"p{p}" / f"{g}_seed{s}"
                d = run_dir / "sun_subsets" / f"p{p}" / f"seed{s}"
                _write_csv(splits[k]["test"], p_test, tau, d / f"{stem}_test_outputs.csv")
                if not minimal:
                    _write_csv(splits[k]["val"], p_val, tau, d / f"{stem}_val_outputs.csv")
                _write_json(d / f"{stem}_last.metrics.json", {**_identity("e1", g, k, s, stem), "test_primary": {"tau": tau},
                                                              "thresholds": {"primary": {"policy": "f1_opt_on_val", "tau": tau}}})
                _write_json(d / f"{stem}.metrics.json", _identity("e1", g, k, s, stem))
                _complete(run_dir, stem)
        gen.truth["H1"]["per_run"][f"{k},{s}"] = float(np.dot(w, d1))
        gen.truth["H2"]["per_run"][f"{k},{s}"] = float(np.dot(w, d2))

    # ---- E3 (H4) and E4 (H5, H6, H7)
    re4, re5, re6, re7 = gen.re("H4"), gen.re("H5"), gen.re("H6"), gen.re("H7")
    lvl4, lvlA = gen.lvl(), gen.lvl()
    pop_clean = pg_pop(clean)
    clean_ids = set(clean["frame_id"])
    pop_eval = {key: pg_pop(pk["eval"]) for key, pk in packs.items()}
    for (k, s) in runs:
        colon = 0.92 + lvl4[(k, s)]
        e3_t = {"ssl_colon": colon, "sup_imnet": colon - re4[(k, s)], "ssl_imnet": colon - 0.02 + gen.eta(), MATCHED: colon - 0.03 + gen.eta()}
        gen.truth["H4"]["per_run"][f"{k},{s}"] = e3_t["ssl_colon"] - e3_t["sup_imnet"]
        p50 = packs[(k, s, 50)]["eval"]
        common = (p50["label"] == 1) & p50["frame_id"].isin(clean_ids)
        pop50_iso = pg_pop(p50[common | (p50["negative_source"] == "isolated")])
        pop50_seq = pg_pop(p50[common | (p50["negative_source"] == "sequence")])
        h5_terms, h6_terms, h7_terms = [], [], []
        for g in regimes:
            delta_u = solve_delta(pop_clean.auroc, e3_t[g], f"E3 AUROC {g} k{k} s{s}")
            a_iso = pop50_iso.auroc(delta_u)
            target_seq = a_iso + re5[(k, s)] + gen.eta()
            m_seq = -solve_delta(lambda mm: pop50_seq.auroc(delta_u, -mm), target_seq, f"H5 sequence AUROC {g} k{k} s{s}", lo=-10, hi=10)
            h5_terms.append(pop50_seq.auroc(delta_u, m_seq) - a_iso)
            tau_sun = sun_tau[(g, k, s)]
            ba_a50 = 0.57 + lvlA[(k, s)] + (0.0 if g == "ssl_colon" else gen.eta())
            t_a = solve_ba_right(pop_eval[(k, s, 50)], ba_a50, delta_u, m_seq, f"E4 arm A s50 {g} k{k} s{s}")
            off_u = logit(tau_sun) - t_a
            y_all = pg_all["label"].to_numpy()
            z_u = np.where(y_all == 1, delta_u, np.where(pg_all["negative_source"].to_numpy() == "sequence", m_seq, 0.0)) + pg_cluster + rng.normal(0, 1, len(pg_all))
            prob_u = expit(z_u + off_u)
            # E3
            stem = f"{TAG[g]}_PolypGenClean_s{s}"
            run3 = root_for(g) / f"k{k}" / "e3" / f"{g}_seed{s}"
            idx_clean = np.array([pg_index[f] for f in clean["frame_id"]])
            _write_csv(clean, prob_u[idx_clean], tau_sun, run3 / "polypgen_clean_test" / f"{stem}_test_outputs.csv")
            _write_json(run3 / "polypgen_clean_test" / f"{stem}.metrics.json",
                        {**_identity("e3", g, k, s, stem), "eval_only": True, "test_primary": {"tau": tau_sun}})
            _complete(run3, stem)
            # E4
            delta_a = delta_u + 0.6
            for b in E4_BUDGETS:
                pack = packs[(k, s, b)]
                pe = pop_eval[(k, s, b)]
                ba_a = pe.ba(t_a, delta_u, m_seq)
                target_b = ba_a + re6[(k, s)] + gen.eta()
                target_c = ba_a + re7[(k, s)] + gen.eta()
                tag = f"{g} s{b} k{k} s{s}"
                t_b = solve_ba_right(pe, target_b, delta_u, m_seq, f"E4 arm B {tag}")
                t_c = solve_ba_right(pe, target_c, delta_a, m_seq, f"E4 arm C {tag}")
                off_a = logit(tau_sun) - t_c
                t_d = solve_ba_right(pe, target_c + 0.02, delta_a, m_seq, f"E4 arm D {tag}")
                h6_terms.append(pe.ba(t_b, delta_u, m_seq) - ba_a)
                h7_terms.append(pe.ba(t_c, delta_a, m_seq) - ba_a)
                tau_b, tau_d = float(expit(t_b + off_u)), float(expit(t_d + off_a))

                def adapted(df: pd.DataFrame) -> np.ndarray:
                    ii = np.array([pg_index[f] for f in df["frame_id"]])
                    y = df["label"].to_numpy()
                    mu = np.where(y == 1, delta_a, np.where(df["negative_source"].to_numpy() == "sequence", m_seq, 0.0))
                    return expit(mu + pg_cluster[ii] + rng.normal(0, 1, len(df)) + off_a)

                stem = f"{TAG[g]}_PolypGenFew{b}_s{s}"
                run4 = root_for(g) / f"k{k}" / "e4" / f"s{b}" / f"{g}_seed{s}"
                d = run4 / "polypgen_fewshot" / f"s{b}" / f"seed{s}"
                ev_idx = np.array([pg_index[f] for f in pack["eval"]["frame_id"]])
                _write_csv(pack["eval"], prob_u[ev_idx], tau_sun, d / f"{stem}_zeroshot_outputs.csv")
                _write_csv(pack["eval"], adapted(pack["eval"]), tau_sun, d / f"{stem}_test_outputs.csv")
                if not minimal:
                    for part in ("train", "val"):
                        ii = np.array([pg_index[f] for f in pack[part]["frame_id"]])
                        _write_csv(pack[part], prob_u[ii], tau_sun, d / f"{stem}_zeroshot_{part}_outputs.csv")
                        _write_csv(pack[part], adapted(pack[part]), tau_sun, d / f"{stem}_{part}_outputs.csv")
                arms = {"A": {"description": "unadapted, inherited tau", "metrics": {"tau": tau_sun}},
                        "B": {"description": "unadapted, tau recalibrated F1-optimal on support (train+val)", "metrics": {"tau": tau_b}},
                        "C": {"description": "adapted, inherited tau", "metrics": {"tau": tau_sun}},
                        "D": {"description": "adapted, tau recalibrated F1-optimal on the support validation split only",
                              "metrics": {"tau": tau_d}, "kept_inherited_tau": 0}}
                ident = _identity("e4", g, k, s, stem)
                _write_json(d / f"{stem}_last.metrics.json", {**ident, "test_primary": {"tau": tau_sun}, "e4_arms": arms})
                _write_json(d / f"{stem}.metrics.json", {**ident, "test_primary": {"tau": tau_sun}})
                _complete(run4, stem)
        gen.truth["H5"]["per_run"][f"{k},{s}"] = float(np.mean(h5_terms))
        gen.truth["H6"]["per_run"][f"{k},{s}"] = float(np.mean(h6_terms))
        gen.truth["H7"]["per_run"][f"{k},{s}"] = float(np.mean(h7_terms))

    truth = {
        "scenario": asdict(sc), "sim_seed": sim_seed, "clipped_targets": len(Clip.log), "clip_log": Clip.log[:50],
        "roots": [str(out / "campaign" / "classification")] + ([str(out / "campaign_matched" / "classification")]
                                                               if sc.include_matched and sc.matched_separate_root else []),
        "hypotheses": {h: {"effect": v["effect"], "realised_mean": float(np.mean(list(v["per_run"].values()))) if v["per_run"] else None,
                           "per_run": v["per_run"]} for h, v in gen.truth.items()},
    }
    if not sc.include_matched:
        truth["hypotheses"]["H2"]["realised_mean"] = None
    _write_json(out / "truth.json", truth)
    return truth
