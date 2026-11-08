# Round 6 results: intake-anchored logit shift of the round-5 embedding fusion

**Provenance.**
- Plan: `docs/ANALYSIS_PLAN_round6.md`, tag `prereg-round6` (06bd2eb, pushed). The hypothesis was formed after the round-5 post-hoc diagnostic; the plan discloses exactly what had been seen.
- Pre-evaluation state: `pre-evaluation-round6` (96ee6de), local commit made before evaluation; not pushed at the time of evaluation (no explicit yes).
- Evaluation: run once (`round5.py evaluate6`). Population: the 104 gated endpoint sessions; metric macro-F1 over Ripe/Overripe under LOBO.
- Sources: `scores.csv`, `primary_contrast.csv`, `secondary_contrasts.csv`, `audit.json`; every number below is copied from these.

## Decision (pre-registered rule)

| Contrast | Δ macro-F1 (pp) | 95% bootstrap CI | p (perm.) | Holm p | Beneficial |
|---|---|---|---|---|---|
| R6 (anchored F3) vs CDR-RO | +9.9 | [−1.1, 20.2] | 0.077 | 0.077 | **no** |

R6 scored 0.759 against 0.661. The gain exceeds 3 pp, but the CI includes zero and p > 0.05, so R6 is reported as not beneficial.

**Wording condition "the shift correction recovers the Overmature batch": not met.** R6's Overmature macro-F1 is 0.709 (≥ 0.70), but pooled R6 − F3 is +6.7 pp [−0.1, 13.9]; the CI includes zero.

## LOBO scores

| Arm | Pooled | Immature | Mature | Overmature | Ripe rec. | Overripe rec. | Overmature Ripe rec. |
|---|---|---|---|---|---|---|---|
| **R6 anchored F3 (primary)** | **0.759** | 0.698 | 0.756 | 0.709 | 0.732 | 0.792 | 0.417 |
| F3 (round 5, unshifted) | 0.692 | 0.639 | 0.726 | 0.529 | 0.661 | 0.729 | 0.167 |
| F3-med (median-centering; seen post hoc) | 0.721 | 0.636 | 0.756 | 0.757 | 0.714 | 0.729 | 0.833 |
| A5-anch | 0.737 | 0.598 | 0.826 | 0.648 | 0.786 | 0.688 | 0.417 |
| A5 (unshifted) | 0.739 | 0.609 | 0.878 | 0.648 | 0.625 | 0.875 | 0.417 |
| F1-anch | 0.727 | 0.404 | 0.804 | 0.595 | 0.786 | 0.667 | 0.250 |
| F1 (unshifted) | 0.731 | 0.671 | 0.829 | 0.370 | 0.661 | 0.812 | 0.000 |
| A1-anch | 0.660 | 0.547 | 0.715 | 0.370 | 0.518 | 0.833 | 0.000 |
| A1 (unshifted) | 0.649 | 0.428 | 0.715 | 0.455 | 0.714 | 0.583 | 0.083 |
| CDR-RO (reference) | 0.661 | 0.558 | 0.829 | 0.370 | 0.446 | 0.938 | 0.000 |
| K1 (round-4 task CNN) | 0.567 | 0.463 | 0.715 | 0.370 | 0.286 | 0.979 | 0.000 |

Anchoring shifts (logit added to the held-out batch, R6): Immature −1.01, Mature +1.09, Overmature −0.74 (from 59/54/49 intake sessions).

## Secondary contrasts (LOBO, unadjusted)

| Contrast | Δ (pp) | 95% CI | p |
|---|---|---|---|
| R6 vs F3 | +6.7 | [−0.1, 13.9] | 0.088 |
| R6 vs K1 | +19.2 | [8.8, 30.0] | 0.001 |
| R6 vs F3-med | +3.9 | [−4.8, 13.0] | 0.390 |
| R6 vs F1-anch | +3.2 | [−5.5, 11.4] | 0.376 |
| A1-anch vs A1 | +1.0 | [−7.6, 9.3] | 0.677 |
| A5-anch vs A5 | −0.2 | [−7.6, 7.4] | 0.817 |
| F1-anch vs F1 | −0.4 | [−7.6, 6.3] | 0.785 |
| F3-med vs CDR-RO | +6.0 | [−6.2, 17.7] | 0.349 |
| A5-med vs CDR-RO | +7.9 | [−3.1, 18.4] | 0.166 |

## Reading

1. **The pre-registered arm did what the mechanism predicted, but not with enough precision to pass the rule.** The anchored shift lifted F3 in every batch (Immature +5.9, Mature +3.0, Overmature +18.0 pp) and raised Overmature Ripe recall from 0.17 to 0.42. Pooled, R6 is the highest score of all six rounds. The CI's lower bound (−1.1 pp) and p = 0.077 are exactly the "consistent with the mechanism, not confirmed" outcome the plan anticipated at n = 104.
2. **The gain is specific to the fusion arm.** The same shift changed A1, A5 and F1 by −0.4 to +1.0 pp. The fusion's batch offset is revealed by its intake sessions; the single-encoder offsets are not, or are not the limiting factor.
3. **Against the task-specific CNN the difference is resolved.** R6 − K1 = +19.2 pp [8.8, 30.0]. This is a secondary contrast and does not change the decision, but it shows that the foundation-embedding fusion plus label-free anchoring is a different regime from the round-1 to round-4 CNNs.
4. **Statistical limit.** With 103 fruit clusters, a true +10 pp effect has about 50% power under this rule (`docs/protocol/power_analysis_methods.md`). A confirmatory test needs the new-harvest design (about 200 endpoint sessions for +10 pp, 400 for +7 pp).
