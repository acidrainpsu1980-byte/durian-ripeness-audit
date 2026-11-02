# Round 4 results: fruit-specific acoustic baseline

- **Plan:** `docs/ANALYSIS_PLAN_round4.md`, tag `prereg-round4` (dc42c77).
- **Code and predictions frozen:** tag `pre-evaluation-round4` (cea8099).
- **Evaluation:** run once, after the tag: `round4.py evaluate`.
- **Population:** 104 endpoint sessions (103 fruits) with a verified intake recording of the same fruit (Immature 34, Mature 41, Overmature 29).
- **Gating:** every method is gated to Ripe/Overripe (false-Unripe rate 0 for all).
- **Metric:** macro-F1 over Ripe and Overripe (R/O).
- **Sources:** `scores.csv`, `primary_contrasts.csv`, `secondary_contrasts.csv`, `baseline_only_diagnostic.csv` and `audit_and_cost.json`; every number below is copied from these files.

## Decision (pre-registered rule)

| Contrast | Δ macro-F1 (pp) | 95% bootstrap CI | p (perm.) | Holm p | Rule met |
|---|---|---|---|---|---|
| P1: K4 paired vs K1 current | −2.0 | [−12.9, 9.5] | 0.776 | 0.776 | no |
| P2: K4 paired vs CDR-RO | −11.3 | [−22.2, −1.1] | 0.030 | 0.060 | no |

- **K4 is not beneficial.** Under P1 it performed no better than the matched current-knock CNN.
- **K4 versus CDR-RO.** It was worse than the round-1 CDR-RO reference. The CI excludes zero, but the Holm p of 0.060 is above 0.05.
- **"Fruit-specific" condition not met.** K4 − K6 (shuffled baseline) = −3.8 pp [−13.0, 5.8]. A baseline from a different fruit of the same batch and fold did as well as the fruit's own baseline.

## LOBO scores (pooled and per batch)

| Model | Pooled F1 | Acc. | Ripe rec. | Overripe rec. | Immature F1 | Mature F1 | Overmature F1 | Overmature Ripe rec. |
|---|---|---|---|---|---|---|---|---|
| K1 current | 0.567 | 0.606 | 0.286 | 0.979 | 0.463 | 0.715 | 0.370 | 0.000 |
| K2 baseline only | 0.541 | 0.558 | 0.339 | 0.812 | 0.244 | 0.634 | 0.695 | 0.500 |
| K3 concat | 0.538 | 0.548 | 0.375 | 0.750 | 0.273 | 0.685 | 0.370 | 0.000 |
| K4 paired (proposed) | 0.548 | 0.548 | 0.482 | 0.625 | 0.463 | 0.568 | 0.341 | 0.000 |
| K5 change only | 0.642 | 0.644 | 0.679 | 0.604 | 0.696 | 0.641 | 0.436 | 0.167 |
| K6 shuffled baseline | 0.585 | 0.615 | 0.321 | 0.958 | 0.429 | 0.776 | 0.370 | 0.000 |
| L1 current features | 0.587 | 0.587 | 0.554 | 0.625 | 0.555 | 0.549 | 0.370 | 0.000 |
| L2 baseline features | 0.610 | 0.615 | 0.679 | 0.542 | 0.396 | 0.610 | 0.721 | 0.750 |
| L3 concat features | 0.560 | 0.567 | 0.643 | 0.479 | 0.414 | 0.586 | 0.416 | 0.083 |
| L4 simple differences | 0.504 | 0.548 | 0.786 | 0.271 | 0.471 | 0.441 | 0.473 | 0.417 |
| L5 current + differences | 0.573 | 0.577 | 0.625 | 0.521 | 0.442 | 0.568 | 0.370 | 0.000 |
| T1 elapsed days (reference) | 0.460 | 0.471 | 0.571 | 0.354 | 0.244 | 0.441 | 0.487 | 0.917 |
| CDR-RO (round-1 M0, gated) | 0.661 | 0.673 | 0.446 | 0.938 | 0.558 | 0.829 | 0.370 | 0.000 |
| M1-RO (anchored std., gated) | 0.626 | 0.644 | 0.393 | 0.938 | 0.377 | 0.878 | 0.370 | 0.000 |

## Fruit-grouped 5-fold CV on the primary partition (p0)

| Model | K1 | K2 | K3 | K4 | K5 | K6 | L1 | L2 | L3 | L4 | L5 | T1 | CDR-RO | M1-RO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pooled macro-F1 | 0.764 | 0.726 | 0.776 | 0.786 | 0.796 | 0.806 | 0.682 | 0.639 | 0.757 | 0.662 | 0.730 | 0.476 | 0.753 | 0.755 |

- **Exploratory p0 contrasts:**
  - K4 − K1 +2.2 pp [−2.6, 6.9];
  - K4 − CDR-RO +3.3 pp [−1.8, 8.7];
  - K4 − K6 −2.0 pp [−7.0, 2.9].
- **L1–L5 means over ten partitions:** 0.702, 0.667, 0.722, 0.626 and 0.693.

## Exploratory contrasts (LOBO)

- **K5 change only vs K1:** +7.4 pp [−5.1, 19.5] (p 0.277). This was the highest-scoring K arm, but the CI includes zero and K5 was not a primary arm.
- **K3 vs K1:** −2.9 [−13.4, 8.2].
- **K4 vs K3:** +0.9 [−5.6, 7.4].
- **K4 vs T1:** +8.8 [−2.6, 19.9].
- **T1 vs CDR-RO:** −20.1 [−33.8, −6.2].
- **L5 vs L1:** −1.4 [−7.0, 4.2]. The differences added no information to the current features.
- **L4 vs L1:** −8.2 [−19.1, 3.0].
- **K4 vs L5:** −2.5 [−11.8, 6.5].
- **M1-RO vs CDR-RO:** −3.5 [−11.0, 4.1].

## Baseline-only diagnostic (latent relation)

| Split | K2 (CNN on intake knocks) | L2 (features on intake recording) |
|---|---|---|
| LOBO | 0.541 [0.445, 0.635] | 0.610 [0.514, 0.703] |
| p0 | 0.726 [0.632, 0.809] | 0.639 [0.544, 0.728] |

- **What the diagnostic shows.** The intake recording, taken at harvest before any ripening, predicts the endpoint class above chance (lower CI bound > 0.5 for three of four entries).
- **Within the primary partition.** K2 (0.726) scored close to K1 (0.764).
- **Interpretation.** The endpoint class therefore relates to fruit or batch properties already present at intake, which is a latent relation as defined in the plan. Any K3/K4 gain over K1 would carry this caveat. In LOBO, K2 and L2 reach Overmature Ripe recall of 0.50 and 0.75, against 0 for K1, K3, K4, K6, CDR-RO and M1-RO.

## Measurement noise and cost

- **Intake knock-to-knock variability.** The median within-fruit SD of log f_dom was 0.342.
- **Intake-to-endpoint change.** The median |log(f_dom,t / f_dom,0)| was 0.572, i.e. about 1.7× the knock-to-knock SD.
- **Cost of the baseline.** It needed one extra recording session per fruit at intake (median 1 recording, 31 knocks; 103 fruits).
- **Leakage audit (`audit_and_cost.json`).** No storage time or weight variable enters any round-4 model or feature, and baselines come from the label-free intake rule.

## Reading

1. **No benefit from a fruit-specific knock baseline.** On this dataset, under batch-disjoint evaluation, it did not improve day-of-capture R/O classification. The proposed paired model (K4) was not beneficial against either primary reference.
2. **The shuffled-baseline control removes the "fruit-specific" interpretation.** The control matched or exceeded K4 in both LOBO and p0.
3. **Within batches, the paired arms add little.** All CNN arms score 0.73–0.81 within batches. K3/K4 add at most 2.2 pp over K1 there, less than the shuffled-baseline K6 adds (+4.2 pp).
4. **The Overmature batch is unchanged.** It is still the failure point: Overmature Ripe recall is 0 for K1, K3, K4, K6, L1, L5, CDR-RO and M1-RO, and ≤ 0.42 for K5, L3 and L4.
5. **The highest Overmature Ripe recall comes from inputs that carry no endpoint acoustics.** These are the intake-only arms K2/L2 (0.50/0.75) and elapsed time T1 (0.92), which points to batch-level design information rather than an acoustic ripening signal.
6. **Change-only (K5) is the only arm with a positive but unconfirmed signal.** It is exploratory (+7.4 pp vs K1, CI includes zero) and would need confirmation on an independent harvest.
