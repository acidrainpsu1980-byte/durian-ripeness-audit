# Round 5 results: pretrained audio and vision foundation encoders

**Provenance.**
- Plan: `docs/ANALYSIS_PLAN_round5.md`, tag `prereg-round5` (6810a38, pushed).
- Pre-evaluation state: all predictions committed locally as `pre-evaluation-round5` (6ab67cb) before evaluation. Not yet pushed: the user did not give an explicit yes.
- Evaluation: run once (`round5.py evaluate`).
- Population: the 104 gated endpoint sessions of round 4. Metric: macro-F1 over Ripe/Overripe.
- Sources: `scores.csv`, `primary_contrasts.csv`, `audio_family_contrasts.csv`, `secondary_contrasts.csv`, `baseline_only_diagnostic.csv`, `batch_decodability.csv` and `audit.json`; every number below is copied from these.

## Decision (pre-registered rule; Holm across Q1 and Q2)

| Contrast | Δ macro-F1 (pp) | 95% bootstrap CI | p (perm.) | Holm p | Beneficial |
|---|---|---|---|---|---|
| Q1: A1 BEATs frozen vs CDR-RO | −1.1 | [−12.3, 9.9] | 0.872 | 0.872 | no |
| Q2: F1 four-modality foundation fusion vs CDR-RO | +7.0 | [−2.6, 16.8] | 0.168 | 0.337 | no |

Neither hypothesis meets the rule.
- **Q1.** The frozen BEATs probe was no better than the round-1 reference.
- **Q2.** The four-modality fusion F1 scored 0.731 against 0.661 for CDR-RO, but its CI includes zero. In the Overmature batch it gave F1 0.370 and Ripe recall 0.

**Audio family (Holm across four, vs CDR-RO):**

| Arm | Δ (pp) | 95% CI | Holm p |
|---|---|---|---|
| A1 BEATs | −1.1 | [−12.3, 9.9] | 0.872 |
| A2 CNN14 | −5.4 | [−16.5, 5.8] | 0.721 |
| A3 CLAP | −9.1 | [−19.9, 0.8] | 0.334 |
| A4 AST | −12.0 | [−22.9, −1.7] | 0.154 |

**Wording conditions (unadjusted).**

| Phrase | Condition | Result | Met |
|---|---|---|---|
| "transfers better than the task-specific CNN" | A1 − K1 | +8.2 [−2.9, 19.8] | no |
| "within-fruit change carries the signal" | A1c − A1s, and A1c − A1 | +10.5 [−2.1, 22.7]; +0.7 [−9.2, 10.2] | no |
| "fine-tuning helps" | A1ft − A1 | +8.1 [0.4, 15.9], p 0.029 | yes |

**What "fine-tuning helps" does and does not mean.** The condition is met under LOBO only.
- A1ft vs CDR-RO is +7.0 [−2.6, 16.8], so the fine-tuned model was still not shown to beat the reference.
- On the primary partition, A1ft − A1 is 0.0.

## LOBO scores (selected; all arms in `scores.csv`)

| Arm | Pooled F1 | Ripe rec. | Overripe rec. | Immature | Mature | Overmature | Overmature Ripe rec. |
|---|---|---|---|---|---|---|---|
| CDR-RO (reference) | 0.661 | 0.446 | 0.938 | 0.558 | 0.829 | 0.370 | 0.000 |
| K1 task CNN (round 4) | 0.567 | 0.286 | 0.979 | 0.463 | 0.715 | 0.370 | 0.000 |
| A1 BEATs frozen | 0.649 | 0.714 | 0.583 | 0.428 | 0.715 | 0.455 | 0.083 |
| A1ft BEATs LP-FT | 0.730 | 0.714 | 0.750 | 0.530 | 0.854 | 0.455 | 0.083 |
| A1c BEATs change-only | 0.656 | 0.750 | 0.562 | 0.717 | 0.529 | 0.455 | 0.083 |
| A1s shuffled baseline | 0.552 | 0.625 | 0.479 | 0.477 | 0.487 | 0.487 | 0.333 |
| A1b baseline only | 0.560 | 0.643 | 0.479 | 0.549 | 0.493 | 0.561 | 0.417 |
| A2 CNN14 | 0.606 | 0.429 | 0.833 | 0.589 | 0.600 | 0.482 | 0.167 |
| A3 CLAP | 0.569 | 0.321 | 0.917 | 0.500 | 0.620 | 0.505 | 0.167 |
| A4 AST | 0.541 | 0.304 | 0.875 | 0.528 | 0.547 | 0.435 | 0.083 |
| A5 wav2vec2 (descriptive) | 0.739 | 0.625 | 0.875 | 0.609 | 0.878 | 0.648 | 0.417 |
| A6 HuBERT (descriptive) | 0.702 | 0.625 | 0.792 | 0.549 | 0.829 | 0.597 | 0.333 |
| V-thermal DINO | 0.489 | 0.232 | 0.875 | 0.429 | 0.590 | 0.341 | 0.000 |
| V-ms DINO | 0.640 | 0.500 | 0.812 | 0.471 | 0.780 | 0.597 | 0.333 |
| V-rgb DINO | 0.625 | 0.571 | 0.688 | 0.404 | 0.559 | 0.505 | 0.167 |
| F1 fusion (DINO) | 0.731 | 0.661 | 0.812 | 0.671 | 0.829 | 0.370 | 0.000 |
| F2 fusion (SigLIP) | 0.608 | 0.446 | 0.812 | 0.589 | 0.626 | 0.370 | 0.000 |
| F3 embedding fusion | 0.692 | 0.661 | 0.729 | 0.639 | 0.726 | 0.529 | 0.167 |
| F4 F1 without RGB | 0.606 | 0.429 | 0.833 | 0.441 | 0.780 | 0.370 | 0.000 |

**Fruit-grouped 5-fold CV on the primary partition (pooled F1):**

| Arm | A1 | A1ft | A1c | A1p | F1 | F3 | F4 | CDR-RO | K1 |
|---|---|---|---|---|---|---|---|---|---|
| Pooled F1 | 0.758 | 0.758 | 0.835 | 0.826 | 0.825 | 0.835 | 0.815 | 0.753 | 0.764 |

Within the primary partition, A1c − A1s was +12.4 [3.7, 21.6].

## Diagnostics

- **Batch identity is encoded before ripening.** On intake recordings or images only (all Unripe at harvest), every frozen embedding predicts the harvest batch with balanced accuracy 0.81–1.00. The range runs from CNN14 0.812 to SigLIP-RGB 0.997; the other audio encoders score 0.969–0.988.
  - **What this means.** Batch-level differences such as harvest, capture days and recording conditions dominate these representations. This is the round-4 latent-relation finding restated in embedding space.
- **Baseline-only (A1b).** LOBO 0.560 [0.457, 0.656]; p0 0.717 [0.626, 0.802].
- **RGB camera is confounded with batch.** Among the 104 endpoint sessions, all 34 Immature sessions were photographed with the EOS R50 and all Mature (41) and Overmature (29) sessions with the M50; no session mixes cameras.
  - **Consequence for the per-camera rows.** The R50 rows are therefore the Immature batch.
  - **Consequence for F1.** Removing RGB from F1 (F4) cost −12.5 pp [−20.6, −4.9] in LOBO. Any RGB contribution cannot be separated from camera/batch identity.
- **Leakage audit (`audit.json`).** No storage time or weight variable enters any embedding, feature or fine-tuning step, and the fine-tuning asserts fruit-disjoint train and test sets.

## Reading

1. **Larger pretrained encoders did not solve the batch shift on this dataset.** The pre-specified BEATs probe matched the reference, and the pre-specified four-modality fusion was higher (+7.0 pp) but not significantly so. The Overmature batch is still the failure point: Ripe recall is 0 for F1, CDR-RO and K1, and ≤ 0.08 for A1/A1ft.
2. **Exploratory signals that a new harvest would need to confirm:**
   - fine-tuning BEATs (+8.1 pp over frozen, LOBO only);
   - wav2vec2 (0.739, the highest LOBO score, and Overmature F1 0.648). This is a descriptive arm and the best of many, so selection bias applies.
   - the change-versus-shuffled contrast, which is significant only within batches.
3. **The data are the limit, not the model class.** Every encoder separates the harvest batches almost perfectly at intake, and camera is confounded with batch. Rounds 1–5 together therefore point to the dataset design (three batches confounded with maturity stage, capture days and, for RGB, camera) as the limit.
