# Deviations from docs/ANALYSIS_PLAN_anchor_physics.md (tag `prereg-anchor-physics`, commit 261c000)

Every item below was decided and written before `src/anchor_physics/evaluate.py` was first run, unless marked otherwise.

## Additions (not in the plan)

1. **CDR_RO baseline (exploratory).** Anchors labeled Unripe; each endpoint session assigned the more probable of Ripe and Overripe
   under the M0 model. This is the trivial use of capture-day knowledge for the endpoint metric (it removes false-Unripe errors by
   construction), added so that any gain of anchoring can be compared with what knowing the capture day alone provides. It is reported
   with the secondary contrasts and does not enter the primary contrasts or the decision rule.
2. **M4 for the knock CNN** is applied per knock (Saerens EM on the held-out batch's knock probabilities; source prior = class
   frequencies of the common-label training knocks), mirroring the per-capture M4 of the classical models. The plan did not state the level.
3. **M1 knock-CNN networks were retrained a second time with weight export** (`CNN_SAVE_WEIGHTS`), because the first M1 run did not
   save weights and the anchor-sensitivity analysis re-applies the M1 networks with subset anchor statistics. The rerun must reproduce
   the first run's predictions (control below).
4. Accuracy permutation p-values and bootstrap CIs for the false-Unripe rate and accuracy are reported in addition to the planned tests.

## Implementation choices the plan left open

5. **Anchor-sensitivity draws:** n = max(2, ceil(f × N)) anchor fruits of the held-out batch (N = 59/54/49 anchor fruits for Immature/
   Mature/Overmature; 6/12/24, 6/11/22 and 5/10/20 fruits for 10/20/40 %), numpy `default_rng(42)`, fruit-level draws shared by all modalities
   (`anchor_draws.csv`). Only the held-out batch's statistics change with the draw; the source batches always use all their anchors, so
   each LOBO model is fitted once and re-applied. Fraction 1.0 reproduces M1 (control).
6. **Within-batch M0 for thermal, RGB and physics** is refitted with the same routine as M1 (`within_batch.py`: same families, grids
   and inner fruit-grouped 3-fold CV as the LOBO refits), so that M1 and M0 differ only by the standardization. For the primary
   partition this reproduces the shipped thermal predictions exactly (max |diff| 0) and the shipped RGB predictions to within 0.0016 in
   probability (1 of 268 sessions changes its predicted class). The shipped RGB within-batch predictions come from the candidate-model code
   (`src/nested_selection/cands.py`), whose embedding preparation and estimator settings are not identical to the LOBO refit routine;
   the cause of the difference was not isolated, and it does not depend on the thread count. The knock-CNN M0 within-batch predictions are the shipped primary-partition CPU fits (`intermediate_v14/cnn_cpu_p0`).
7. **Within-batch comparator of the primary contrasts** is the primary partition for all three (each involves a knock CNN):
   PA1 CNN M1 vs CNN M0; PA2 mean rule M1 vs M0; PC1 physics M1 vs CNN M0.

## Information seen before evaluation

8. The knock-CNN training script prints the knock-level accuracy on the held-out batch after each LOBO fit (an existing diagnostic of
   `cnn_common.py`). These knock accuracies of M0, M1 and M2 were therefore visible in the training logs before evaluation. They are not
   the pre-registered score (session level, endpoint sessions, macro-F1 over Ripe/Overripe), and no analysis choice remained open.

## Data observations

9. **f_first** (lowest peak with prominence ≥ 10 dB in 100–2000 Hz) has a median of 153 Hz (SD 16 Hz) across knocks, i.e. it mostly
   sits near the lower band edge (the recordings' high-pass region or low-frequency noise) rather than at a fruit resonance. It is kept
   in the physics model as pre-registered.

## Controls

- Physics extraction: the full-band dominant frequency reproduces the paper's `dom_freq` for all 15,709 knocks.
- Classical LOBO M0 (thermal, RGB) reproduces `results/lobo_table.csv` exactly (accuracy and macro-F1 per held-out batch).
- Anchor sensitivity, fraction 1.0 = M1: max |diff| 0 for thermal, RGB and physics.
- Threads: all classical stages are run with `OMP_NUM_THREADS=4`, as in `run_all.sh` (a first run of the within-batch, baseline and
  anchor-sensitivity stages with 1 thread differed by up to 0.0011 in RGB probabilities and was replaced before evaluation).
- Knock CNN M0 (CPU, 3 threads) vs the shipped LOBO CNN (`intermediate_common/sound/cnn_lobo_recprobs.json`): max |diff| 0 over the
  432 verified recordings (`cnn_controls.json`).
- Reloaded M0 networks vs M0 predictions: max |diff| 0.
- M1 rerun with weight export (`cnn/M1w`, weights `cnn/M1_weights`) vs first M1 run (`cnn/M1`): max |diff| 0.
