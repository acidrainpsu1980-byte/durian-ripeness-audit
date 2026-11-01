# Results summary: pre-registered anchor/physics analysis (results/v21)

Generated from primary_contrasts.csv, secondary_contrasts.csv and scores.csv. Primary score: LOBO, 106 endpoint sessions, macro-F1 over Ripe/Overripe.

## Primary contrasts (Holm over three; decision rule of plan §5)

| Contrast | Candidate | Reference | Δ macro-F1 (pp) [95% CI] | p perm | p Holm | Δ false-Unripe (pp) | Within-batch drop (pp, p0) | Beneficial |
|---|---|---|---|---|---|---|---|---|
| PA1 | sound_cnn/M1 | sound_cnn/M0 | −2.8 [−10.7, +4.9] | 0.513 | 1.000 | −11.3 | +0.8 | no |
| PA2 | tsr_mean/M1 | tsr_mean/M0 | −2.4 [−11.1, +5.6] | 0.580 | 1.000 | +3.8 | +0.0 | no |
| PC1 | physics/M1 | sound_cnn/M0 | −4.2 [−16.0, +8.6] | 0.587 | 1.000 | −13.2 | +18.2 | no |

## Endpoint scores (pooled)

| Model / variant | macro-F1 R/O | false-Unripe | Ripe recall | Overripe recall |
|---|---|---|---|---|
| sound_cnn / M0 | 0.464 | 0.396 | 0.105 | 0.796 |
| sound_cnn / M1 | 0.436 | 0.283 | 0.088 | 0.857 |
| sound_cnn / M2 | 0.579 | 0.160 | 0.333 | 0.796 |
| sound_cnn / M5 | 0.652 | 0.019 | 0.491 | 0.837 |
| sound_cnn / M6 | 0.524 | 0.160 | 0.316 | 0.694 |
| sound_cnn / CDR_RO | 0.656 | 0.000 | 0.439 | 0.939 |
| tsr_mean / M0 | 0.416 | 0.425 | 0.105 | 0.653 |
| tsr_mean / M1 | 0.391 | 0.462 | 0.053 | 0.673 |
| tsr_mean / CDR_RO | 0.493 | 0.000 | 0.246 | 0.857 |
| physics / M0 | 0.546 | 0.142 | 0.298 | 0.796 |
| physics / M1 | 0.422 | 0.264 | 0.526 | 0.224 |
| phys_pcm / M0 | 0.599 | 0.113 | 0.579 | 0.551 |
| phys_weight / M0 | 0.392 | 0.123 | 0.333 | 0.408 |
| hybrid / M2 | 0.587 | 0.160 | 0.333 | 0.816 |

## Selected secondary contrasts (exploratory, unadjusted)

| Candidate | Reference | Δ macro-F1 (pp) [95% CI] | p perm |
|---|---|---|---|
| sound_cnn/M5 | sound_cnn/M0 | +18.8 [+10.5, +27.1] | <0.001 |
| sound_cnn/CDR_RO | sound_cnn/M0 | +19.3 [+11.5, +27.7] | <0.001 |
| sound_cnn/M5 | sound_cnn/M6 | +12.8 [+5.1, +20.8] | 0.002 |
| sound_cnn/M2 | sound_cnn/M0 | +11.5 [+1.8, +21.6] | 0.038 |
| sound_cnn/M1 | sound_cnn/M2 | −14.3 [−24.9, −3.9] | 0.016 |
| tsr_mean/CDR_RO | tsr_mean/M0 | +7.7 [+0.4, +15.6] | 0.063 |
| physics/M0 | sound_cnn/M0 | +8.2 [−3.0, +19.6] | 0.197 |
| phys_pcm/M0 | sound_cnn/M0 | +13.5 [+2.0, +24.0] | 0.038 |
| physics/M1 | phys_pcm/M1 | −6.6 [−13.4, +0.1] | 0.070 |
| physics/M1 | physics/M0 | −12.4 [−25.6, +0.9] | 0.102 |

Leakage audit: all assertions passed (anchor rule re-derived label-free: True; label use in CNN_NORM/AdaBN/standardization blocks: none).

Descriptive (anchors, verified recordings): median f_dom of intake recordings 726 / 495 / 519 Hz (Immature / Mature / Overmature); the M0 CNN predicts 51% of Overmature intake sessions as ripened (37% Ripe, 14% Overripe), 0% Immature, 4% Mature.
