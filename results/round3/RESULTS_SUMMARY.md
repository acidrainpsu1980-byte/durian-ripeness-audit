# Results summary: round 3, final (docs/ANALYSIS_PLAN_round3.md; code fixed at tag pre-evaluation-round3)

106 LOBO endpoint sessions, macro-F1 over Ripe/Overripe. Evaluated once.

## Nested selection

| Held-out batch | Selected | Inner C2 | Inner C4 | Inner D1 | Inner D2 | Inner D3 | Inner D4 |
|---|---|---|---|---|---|---|---|
| Immature | C2 | 0.620 | 0.593 | 0.541 | 0.595 | 0.345 | 0.595 |
| Mature | D1 | 0.552 | 0.615 | 0.677 | 0.581 | 0.558 | 0.519 |
| Overmature | C4 | 0.599 | 0.612 | 0.365 | 0.544 | 0.573 | 0.544 |

## Primary contrasts (Holm over two)

| Contrast | Reference | Δ macro-F1 (pp) [95% CI] | p perm | p Holm | Beneficial |
|---|---|---|---|---|---|
| R3-1 | CDR_RO | −10.0 [−20.5, +0.9] | 0.096 | 0.096 | no |
| R3-2 | round2_procedure | −12.3 [−20.2, −5.5] | <0.001 | 0.001 | — |

## Endpoint scores

| Rule | Immature | Mature | Overmature | Pooled | Ripe recall | Overripe recall | Overmature Ripe recall |
|---|---|---|---|---|---|---|---|
| procedure | 0.597 | 0.339 | 0.354 | 0.556 | 0.667 | 0.449 | 0.000 |
| CDR_RO | 0.558 | 0.829 | 0.367 | 0.656 | 0.439 | 0.939 | 0.000 |
| round2_procedure | 0.597 | 0.802 | 0.354 | 0.679 | 0.632 | 0.735 | 0.000 |
| C2 | 0.597 | 0.752 | 0.367 | 0.666 | 0.526 | 0.837 | 0.000 |
| C4 | 0.598 | 0.802 | 0.354 | 0.688 | 0.667 | 0.714 | 0.000 |
| D1 | 0.404 | 0.339 | 0.367 | 0.558 | 0.772 | 0.367 | 0.000 |
| D2 | 0.609 | 0.853 | 0.581 | 0.696 | 0.719 | 0.673 | 0.692 |
| D3 | 0.404 | 0.328 | 0.367 | 0.567 | 0.404 | 0.776 | 0.000 |
| D4 | 0.489 | 0.804 | 0.452 | 0.602 | 0.632 | 0.571 | 0.538 |

## Candidates vs references (exploratory, unadjusted)

| Candidate | Reference | Δ macro-F1 (pp) [95% CI] | p perm |
|---|---|---|---|
| C2 | CDR_RO | +1.0 [−6.7, +8.9] | 0.812 |
| C2 | round2_procedure | −1.3 [−8.5, +5.8] | 0.612 |
| C4 | CDR_RO | +3.2 [−6.3, +12.9] | 0.569 |
| C4 | round2_procedure | +0.9 [−3.9, +5.7] | 1.000 |
| D1 | CDR_RO | −9.8 [−21.3, +2.4] | 0.141 |
| D1 | round2_procedure | −12.1 [−22.3, −2.4] | 0.022 |
| D2 | CDR_RO | +4.0 [−6.3, +15.1] | 0.500 |
| D2 | round2_procedure | +1.7 [−7.7, +11.2] | 0.836 |
| D3 | CDR_RO | −9.0 [−21.8, +3.4] | 0.177 |
| D3 | round2_procedure | −11.2 [−22.2, −0.3] | 0.052 |
| D4 | CDR_RO | −5.5 [−15.7, +4.0] | 0.295 |
| D4 | round2_procedure | −7.8 [−18.1, +2.5] | 0.126 |

## Estimated Overripe priors (EM) and source shares

| Held-out | Split | π_O source | D1 prior O | D3 prior O |
|---|---|---|---|---|
| Immature | outer | 0.528 | 0.000 | 0.000 |
| Mature | outer | 0.446 | 0.034 | 1.000 |
| Overmature | outer | 0.413 | 1.000 | 1.000 |
| Immature | inner Mature>Overmature | 0.488 | 1.000 | 1.000 |
| Immature | inner Overmature>Mature | 0.581 | 0.057 | 1.000 |
| Mature | inner Immature>Overmature | 0.324 | 1.000 | 1.000 |
| Mature | inner Overmature>Immature | 0.581 | 0.266 | 0.580 |
| Overmature | inner Immature>Mature | 0.324 | 0.818 | 1.000 |
| Overmature | inner Mature>Immature | 0.488 | 0.722 | 0.346 |

Leakage audit: {'rules_use_labels': False, 'selection_reads_outer_batch': False} (all False = passed). EM priors often converge to 0 or 1 (degenerate), so D1/D3 assign nearly all endpoints of a batch to one class.
