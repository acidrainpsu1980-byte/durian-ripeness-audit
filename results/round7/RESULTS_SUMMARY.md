# Round 7 — Rank-then-Anchor (RtA): results

Plan `docs/ANALYSIS_PLAN_round7.md` (tag `prereg-round7`, 35ea496); frozen predictions and deviations at `pre-evaluation-round7` (f899f99), both pushed and verified on the remote before the next step. `evaluate7` ran once on 10 November 2026 (10:58 UTC). LOBO endpoint sessions: 104 (103 fruits), macro-F1 over Ripe and Overripe, gated. Round 7 is the seventh analysis of the same sessions and is exploratory.

## Primary contrast (decision rule unchanged from rounds 2–6)

| Contrast | Pooled macro-F1 | Δ (pp) [95% CI] | p (perm.; Holm = identity) | Beneficial |
|---|---|---|---|---|
| R7 vs CDR-RO | 0.670 vs 0.661 | +1.0 [−10.7, 11.7] | 0.865 | **no** |

## Scores (LOBO; macro-F1 pooled / Immature / Mature / Overmature)

| Arm | Pooled | Immature | Mature | Overmature | Overmature Ripe recall |
|---|---|---|---|---|---|
| **R7** (centred, ranking head) | 0.670 | 0.591 | 0.631 | 0.626 | 0.33 |
| R7-nc (no centring) | 0.644 | 0.598 | 0.631 | 0.455 | 0.08 |
| R7-ce (centred, round-5 head) | 0.750 | 0.698 | 0.731 | 0.709 | 0.42 |
| R7-A (BEATs block only) | 0.569 | 0.490 | 0.392 | 0.455 | 0.08 |
| R7-noRGB | 0.583 | 0.504 | 0.527 | 0.455 | 0.08 |
| CDR-RO (reference) | 0.661 | 0.558 | 0.829 | 0.370 | 0.00 |
| K1 | 0.567 | 0.463 | 0.715 | 0.370 | 0.00 |
| F3 | 0.692 | 0.639 | 0.726 | 0.529 | 0.17 |
| R6 (anchored F3) | 0.759 | 0.698 | 0.756 | 0.709 | 0.42 |

All four references reproduce their shipped values exactly. p0 cost check (fruit-grouped, descriptive): R7 0.777, R7-ce 0.854, F3 0.835.

## Pre-specified wording conditions (reported whether met or not)

- *Threshold transfers without a post-hoc shift* (threshold loss ≤ 5 pp in each batch): **not met** — R7 loses 9.9 / 10.0 / 13.3 pp to the oracle threshold (Immature / Mature / Overmature); it predicts Overripe for 15 % / 41 % / 83 % of endpoint sessions against true fractions of 32 % / 49 % / 59 %.
- *Within-batch ranking training does not cost ranking* (mean per-batch AUC within 0.03 of F3): **not met** — 0.732 vs 0.811 (R7 AUC 0.68 / 0.69 / 0.83; F3 0.70 / 0.84 / 0.90).
- *Intake centring is the active component* (R7 − R7-nc ≥ 3 pp, CI > 0): **not met** — +2.6 [−2.1, 7.2].
- *The ranking head is the active component* (R7 − R7-ce ≥ 3 pp, CI > 0): **not met** — −8.0 [−16.0, −0.3].
- Hence, as pre-specified: **the components could not be separated at n = 104.**

## Exploratory contrasts (unadjusted)

R7 − F3 −2.2 [−9.8, 4.9]; R7 − R6 −8.9 [−16.8, −1.2]; R7 − K1 +10.3 [−1.3, 22.0]; R7-ce − CDR-RO +8.9 [−2.0, 19.4]; R7-nc − CDR-RO −1.7 [−12.0, 8.6]; R7-A − CDR-RO −9.2 [−20.3, 2.4]; R7-noRGB − CDR-RO −7.7 [−18.6, 3.7]; R7-A − R7 −10.1 [−19.2, −1.2]; R7-noRGB − R7 −8.7 [−16.4, −0.7]. p0: R7 − F3 −5.8 [−13.4, 1.0]; R7-ce − F3 +1.9 [−4.8, 8.8].

## Batch decodability after centring

Harvest batch was decoded from the uncentred intake blocks at balanced accuracy 1.00 and from the centred blocks at 0.00. The second value is below chance (0.33) and is an artefact of exact per-batch centring under cross-validation (removing a fold's units moves the remaining units of that batch to the opposite side of zero); this probe therefore cannot measure residual batch information after Step 1. As the plan states, neither value is evidence about ripeness predictions.

## Interpretation (plan Section 7, third row)

R7 was not beneficial and none of the wording conditions was met. The within-batch ranking head ranked worse than the pooled cross-entropy head on the same centred blocks and did not remove the threshold problem. Intake centring with the round-5 head (R7-ce) reproduced R6's per-batch scores in the Immature and Overmature batches (exploratory). Together with rounds 1–6, the limit remains the data: three batches confounded with maturity, camera and harvest. The new-harvest protocol and its primary contrast are unchanged.
