# Unadapted transfer by negative source

**Post hoc sensitivity analysis, after the registered analysis and not part of the SAP.** It answers the objection that
the isolated negatives of PolypGen invalidate the cross-centre transfer conclusions. It recomputes the unadapted AUROC
and the regime contrasts against sequence negatives only, isolated negatives only, and all negatives. Analysis of
existing data only: nothing was trained and no registered result was modified.

    PYTHONPATH=. python scripts/v3/zeroshot_by_negative_source.py --out results_v3/verification/zeroshot_by_negative_source

`ssl4polyp-analysis`, run 2026-10-31, 112 s (`run_log.txt`), registered campaign `checkpoints_v3/campaign/classification`.

## Data

- **Outputs:** the stored per-frame outputs of the unadapted models used by H5: the `*_zeroshot_outputs.csv` of the
  E4 50-frame runs, i.e. the `sun_full` model of each partition, seed and regime, on the 50-frame evaluation split.
- **Rows:** as in H5 (`design.negsource_rows`), the positives also present in `polypgen_clean_test` plus the negatives of
  one population:
  - **(a) sequence:** sequence negatives only (registered view `("neg", 50, regime, k, s, "sequence")`);
  - **(b) isolated:** isolated negatives only (registered view `(..., "isolated")`);
  - **(c) all:** all negatives (the union of the two).

Composition per pack (`composition_by_pack.csv`; per pack and regime in `composition_by_pack_regime.csv`):

| Pack | Positives | Sequence negatives | Isolated negatives |
|---|---|---|---|
| each of the 15 (k1–k5 × seeds 13, 29, 47) | 1,229 | 3,190 | 159 |

- **Positives:** all 1,229 positives of the split are in `polypgen_clean_test`.
- **Packs:** the 15 splits hold different frames but the same counts.
- **Regimes:** within a pack, the four regimes evaluate the same frames in every population (checked).
- **Isolated negatives:** they come from 6 centres (C4 69, C3 53, C2 26, C1 5, C6 4, C5 2). The sequence negatives
  come from 18 sequences.

Each population therefore has, per pack and regime: (a) 4,419 frames, (b) 1,388 frames, (c) 4,578 frames.

## Inference

The registered code (`analysis/v3`), unchanged:

- **Estimate:** paired, the mean of the 15 per-run values (5 partitions × 3 seeds).
- **t-test CI:** on the 5 partition means (`mixed`, variant `ttest`).
- **Bootstrap CI:** cluster bootstrap of the E4 units (B = 2000, seed 12345). The units are 6 centres for positives,
  and 24 negative units: 18 sequences plus the 6 centres of the isolated negatives.
  - The (c) views hold no unit outside the (a) and (b) views, so the bootstrap space and draws are those of the
    registered `neg` family. The script checks this.
  - **Valid replicates:** 2000 in (a) and (c), **1997 in (b)**. In 3 replicates no isolated-negative unit is drawn, so
    (b) is undefined; as in the registered analysis, those replicates are discarded (column `boot_B`).

Contrasts: SSL-colon − SUP-imnet (the comparison of H4), SSL-colon − SSL-imnet, SSL-colon − SSL-imnet-matched and
SSL-imnet − SUP-imnet. AUROC levels per regime are also given.

## Check

The SSL-colon contrasts in (a) and (b) are registered secondary contrasts (`cmp|neg|50|<source>|ssl_colon-<base>|auroc`
in `results_v3/analysis/contrast_summary.csv`). They are reproduced: 36/36 quantities equal, max |diff| 4.6e-11
(`registered_check.csv`). The quantities are estimate, t-test CI, bootstrap CI and B. The script stops if any differs.

## Results

Each cell gives the estimate, then the t-test CI and the bootstrap CI.

| Contrast | (a) Sequence negatives | (b) Isolated negatives | (c) All negatives |
|---|---|---|---|
| SSL-colon − SUP-imnet (H4) | −0.1074 [−0.1337, −0.0812; −0.1864, −0.0427] | −0.0699 [−0.0954, −0.0444; −0.3542, 0.1108] | −0.1056 [−0.1308, −0.0805; −0.1829, −0.0412] |
| SSL-colon − SSL-imnet | −0.0504 [−0.0870, −0.0137; −0.1187, 0.0057] | −0.0770 [−0.0954, −0.0586; −0.2972, 0.0522] | −0.0516 [−0.0869, −0.0164; −0.1177, 0.0030] |
| SSL-colon − SSL-imnet-matched | 0.0180 [0.0021, 0.0339; −0.0240, 0.0688] | −0.0002 [−0.0214, 0.0210; −0.1936, 0.1003] | 0.0171 [0.0021, 0.0321; −0.0265, 0.0689] |
| SSL-imnet − SUP-imnet | −0.0571 [−0.0747, −0.0394; −0.0824, −0.0340] | 0.0071 [−0.0101, 0.0244; −0.0694, 0.0614] | −0.0540 [−0.0706, −0.0374; −0.0786, −0.0301] |

**AUROC per regime:**

| Regime | (a) Sequence | (b) Isolated | (c) All |
|---|---|---|---|
| SUP-imnet | 0.8828 [0.8526, 0.9130; 0.7912, 0.9572] | 0.6886 [0.6723, 0.7050; 0.5039, 0.8781] | 0.8736 [0.8442, 0.9030; 0.7846, 0.9451] |
| SSL-imnet | 0.8258 [0.7890, 0.8626; 0.7225, 0.9103] | 0.6958 [0.6776, 0.7139; 0.5501, 0.8301] | 0.8196 [0.7848, 0.8544; 0.7202, 0.9012] |
| SSL-colon | 0.7754 [0.7287, 0.8221; 0.6181, 0.9035] | 0.6187 [0.5920, 0.6454; 0.4037, 0.7519] | 0.7680 [0.7232, 0.8128; 0.6154, 0.8909] |
| SSL-imnet-matched | 0.7574 [0.7105, 0.8043; 0.5907, 0.8926] | 0.6190 [0.6085, 0.6294; 0.5258, 0.7169] | 0.7509 [0.7061, 0.7956; 0.5919, 0.8766] |

## Reading

- **Regime ordering:** with sequence negatives only, it is the same as with all negatives: SUP-imnet > SSL-imnet >
  SSL-colon ≈ SSL-imnet-matched.
- **SSL-colon − SUP-imnet:** negative in all three populations. It is −0.107 without isolated negatives and −0.106 with
  all. The registered H4, on `polypgen_clean_test`, is −0.072.
- **(a) vs (c):** the two populations give almost the same contrasts. Isolated negatives are 159 of 3,349 negatives
  (4.7 %), so they barely move the contrasts that use all negatives.
- **Isolated negatives are harder for every regime.** AUROC is 0.62–0.70 against isolated negatives and 0.76–0.88
  against sequence negatives (H5).
- **Wide bootstrap intervals in (b):** the isolated negatives form only 6 centre units, 122 of the 159 frames in two
  centres. Against isolated negatives, SSL-imnet − SUP-imnet is close to 0.
- Sensitivity analysis: no Holm, no verdicts. Registered results are unchanged.

## Files

| File | Content |
|---|---|
| `contrasts_long.csv`, `zeroshot_by_negative_source.md` | AUROC levels and contrasts per population: estimate, partition and seed SD, t-test CI, unadjusted p, bootstrap CI, valid replicates `boot_B` |
| `run_auroc.csv` | AUROC per run (regime, partition, seed) and population |
| `composition_by_pack.csv`, `composition_by_pack_regime.csv` | positives, sequence and isolated negatives per pack (and regime) and population |
| `registered_check.csv`, `summary.json`, `run_log.txt` | reproduction check and run information |
