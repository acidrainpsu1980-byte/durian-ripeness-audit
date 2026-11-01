# Analysis plan, round 2: protocol-aware adaptation of the knock CNN across harvest batches

Status: **written and committed before any round-2 model was fitted or scored.** The git tag `prereg-round2` of this file precedes
every result file of this round (`results/round2/`). Deviations will be listed in `results/round2/DEVIATIONS.md` and in the paper.

## 0. What is known before this plan

Round 1 (`docs/ANALYSIS_PLAN_anchor_physics.md`, tag `prereg-anchor-physics`; code and deviations fixed at tag
`pre-evaluation-anchor-physics`; results in `results/v21/`) was evaluated once. On the 106 LOBO endpoint sessions (macro-F1 over
Ripe/Overripe), it found:

- **Primary contrasts.** All three pre-registered primary contrasts were negative and not beneficial:
  - PA1, CNN M1 vs M0: −2.8 pp;
  - PA2, mean rule M1 vs M0: −2.4 pp;
  - PC1, physics M1 vs CNN M0: −4.2 pp;
  - Holm p = 1.0 for all three.
- **Exploratory results.**
  - Knock CNN M0: 0.464.
  - Capture-day rule CDR-RO (anchors Unripe; endpoints by the larger of P(Ripe) and P(Overripe) under M0): 0.656.
  - AdaBN on the held-out batch's intake knocks (M5): 0.652.
  - AdaBN on all its knocks (M6): 0.524.
  - Unanchored per-batch standardization (M2): 0.579.
  - The peak + centroid + MFCC LR: 0.599 (0.628 with the capture-day rule).
- **Intake state differs by batch.** Median intake dominant frequency is 726 / 495 / 519 Hz (Immature / Mature / Overmature).
  The M0 CNN calls 51% of Overmature intake sessions ripened.

The candidates below were designed **after** seeing these results on the same sessions. Their individual outer scores are therefore
not independent of the design and are reported as exploratory. The primary estimate (§3) selects among the candidates without
using the held-out batch, which limits optimism from choosing a candidate. It cannot remove the influence of round 1 on which
candidates exist.

## 1. Data, anchors and endpoints

These are as in round 1:
- 268 four-sensor sessions;
- the label-free intake rule: the earliest session of each fruit, if ≤ 4 days after its recorded harvest;
- 162 anchors and 106 endpoint sessions;
- 432 verified recordings;
- the common label.

Endpoint status is known by protocol: a session that is not an intake session. It may be used at test time, but labels may not.
End-of-storage weight, weight loss, storage time and any endpoint variable are never used.

## 2. Candidates (fixed; no further candidates will be added)

The knock CNN has the architecture, schedule and seed of the paper (20 epochs, seed 42, CPU 3 threads). AdaBN works as in round 1:
BatchNorm running statistics are re-estimated as a cumulative average, with no augmentation and dropout off. Gating means that
anchors are predicted Unripe and endpoints are predicted as the more probable of Ripe and Overripe.

- **C1 = M5.** Three-class CNN trained on the source batches, with AdaBN on the target batch's anchor knocks. Prediction: argmax
  over the three classes.
- **C2 = M5 + gating.** C1's probabilities, gated.
- **C3 = endpoint-only CNN.** A two-class Ripe-vs-Overripe CNN trained on the source knocks labeled Ripe or Overripe (common
  label), with class-balanced weights. AdaBN uses the target batch's unlabeled **endpoint** knocks. Gated.
- **C4 = C2 + acoustic LR.** The mean of C2's renormalized Ripe/Overripe probabilities and those of the peak + centroid + MFCC
  LR (round-1 `phys_pcm`, M0), gated.
- **C5 = gated fusion.** The thermal + sound + RGB mean, with the C1 CNN probabilities and the M0 thermal SVM-RBF and RGB probe
  probabilities, gated.

Reference rules (not candidates):
- **CDR-RO**: the M0 CNN, gated (round 1);
- **M5**: C1.

## 3. Primary estimate: nested leave-one-batch-out selection

For each held-out batch b, with source batches s and t:

1. **Inner fits.** Every candidate is fitted on s alone and scored on the endpoint sessions of t, and vice versa. For each inner
   test batch, AdaBN statistics and anchors come from that inner test batch, under the same label-free rules.
2. **Inner score.** Macro-F1 over Ripe/Overripe, pooled over the two inner test batches.
3. **Selection.** The candidate with the highest inner score is selected. Ties go to the lower candidate number.
4. **Outer prediction.** The selected candidate's outer model (trained on s and t) predicts the endpoint sessions of b.

The procedure's predictions on the three held-out batches are pooled into the primary score. Reported for each held-out batch: the
selected candidate and the inner scores of all candidates.

Inner fits:
- knock CNN, 3-class: 6 single-batch fits;
- knock CNN, 2-class: 6 single-batch fits;
- the thermal, RGB and peak + centroid + MFCC models: refitted on one batch with their round-1 families and grids.

Outer fits:
- 3-class CNN: the round-1 M0 networks;
- 2-class CNN: 3 new LOBO fits;
- classical models: the round-1 M0 LOBO fits.

## 4. Primary contrasts, decision rule and multiplicity

All contrasts use the pooled 106 endpoint sessions; Holm adjustment across the two.

- **R2-1.** Nested-selection procedure vs CDR-RO.
- **R2-2.** Nested-selection procedure vs M5.

Each contrast reports:
- the difference in macro-F1 over Ripe/Overripe;
- a fruit-cluster bootstrap 95% CI (2,000 resamples, seed 42);
- a fruit-level permutation test (20,000, seed 42);
- the clustered McNemar test for accuracy.

The procedure is called **beneficial** only if, against CDR-RO, all three hold:
- the gain is at least 3 pp;
- the bootstrap CI excludes zero;
- the Holm-adjusted p < 0.05.

Otherwise it is reported as not beneficial, whatever its direction. The within-batch check of round 1 is not part of this round:
gating uses capture-day knowledge, which is equally available within a batch.

## 5. Secondary (exploratory)

The following are reported descriptively, unadjusted:
- per-candidate outer scores (pooled and per batch), with contrasts against CDR-RO and M5;
- false-Unripe rate and class recall;
- all-268-session scores;
- the inner score table.

## 6. Checks

- **Controls.**
  - C1 outer reproduces round-1 M5 exactly.
  - The gated round-1 M0 reproduces round-1 CDR-RO.
  - The 2-class CNN mode leaves the default 3-class behaviour of `cnn_common.py` unchanged.
- **Leakage audit.** No label of a target batch enters AdaBN statistics, gating or selection. Inner selection never reads the
  outer batch.
- All outputs go to `results/round2/`, produced by `src/anchor_physics/round2.py`.
