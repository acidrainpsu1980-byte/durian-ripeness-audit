# Results summary: round 2 (docs/ANALYSIS_PLAN_round2.md; code fixed at tag pre-evaluation-round2)

106 LOBO endpoint sessions, macro-F1 over Ripe/Overripe. Evaluated once.

## Nested selection

| Held-out batch | Selected | Inner C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|
| Immature | C2 | 0.565 | 0.620 | 0.582 | 0.593 | 0.420 |
| Mature | C4 | 0.546 | 0.552 | 0.543 | 0.615 | 0.599 |
| Overmature | C4 | 0.599 | 0.599 | 0.578 | 0.612 | 0.570 |

## Primary contrasts (Holm over two)

| Contrast | Reference | Δ macro-F1 (pp) [95% CI] | p perm | p Holm | Δ accuracy (pp) | Beneficial |
|---|---|---|---|---|---|---|
| R2-1 | CDR_RO | +2.3 [−6.3, +11.7] | 0.686 | 0.936 | +0.9 | no |
| R2-2 | M5 | +2.8 [−4.7, +10.2] | 0.468 | 0.936 | +2.8 | — |

## Endpoint scores

| Rule | Immature | Mature | Overmature | Pooled | Ripe recall | Overripe recall |
|---|---|---|---|---|---|---|
| procedure | 0.597 | 0.802 | 0.354 | 0.679 | 0.632 | 0.735 |
| CDR_RO | 0.558 | 0.829 | 0.367 | 0.656 | 0.439 | 0.939 |
| M5 | 0.597 | 0.715 | 0.367 | 0.652 | 0.491 | 0.837 |
| C1 | 0.597 | 0.715 | 0.367 | 0.652 | 0.491 | 0.837 |
| C2 | 0.597 | 0.752 | 0.367 | 0.666 | 0.526 | 0.837 |
| C3 | 0.601 | 0.749 | 0.311 | 0.581 | 0.632 | 0.531 |
| C4 | 0.598 | 0.802 | 0.354 | 0.688 | 0.667 | 0.714 |
| C5 | 0.528 | 0.518 | 0.367 | 0.523 | 0.298 | 0.837 |

## Candidates vs references (exploratory, unadjusted)

| Candidate | Reference | Δ macro-F1 (pp) [95% CI] | p perm |
|---|---|---|---|
| C1 | CDR_RO | −0.5 [−8.4, +7.8] | 0.873 |
| C2 | CDR_RO | +1.0 [−6.7, +8.9] | 0.812 |
| C2 | M5 | +1.5 [+0.0, +3.8] | 0.497 |
| C3 | CDR_RO | −7.5 [−17.8, +2.5] | 0.156 |
| C3 | M5 | −7.0 [−17.5, +3.3] | 0.192 |
| C4 | CDR_RO | +3.2 [−6.3, +12.9] | 0.569 |
| C4 | M5 | +3.7 [−5.2, +12.0] | 0.416 |
| C5 | CDR_RO | −13.4 [−23.9, −2.5] | 0.022 |
| C5 | M5 | −12.9 [−22.3, −3.1] | 0.010 |

Leakage audit: {'adabn_reference_uses_labels': False, 'gating_uses_labels': False, 'selection_reads_outer_batch_scores': False} (all False = passed).
