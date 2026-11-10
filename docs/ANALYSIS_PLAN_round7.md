# Analysis plan, round 7: Rank-then-Anchor (RtA) — intake-centred embeddings with a within-batch ranking head under batch-disjoint evaluation

Status: **committed and tagged `prereg-round7` on 10 November 2026, before any model below was fitted or scored.**
- The tag `prereg-round7` must precede every result file of this round (`results/round7/`). **The tag is pushed and verified on the remote (`git ls-remote --tags origin prereg-round7`) before `fit7` is started**; the round-6 pre-evaluation tag was lost because it was not pushed.
- Deviations will be listed in `results/round7/DEVIATIONS.md` and in the paper.
- No new embedding or encoder is introduced. The round uses the round-5 frozen embeddings (md5 in `docs/ANALYSIS_PLAN_round5.md`, Section 8). The head and the centring step are new.

## 0. Disclosure

**Where this round comes from.** Round 7 was designed on 10 November 2026 after all six rounds had been evaluated and after the manuscript reporting them had been drafted. **No new diagnostic motivates it.** It was motivated by the wish to test a model that builds the round-5 post-hoc diagnosis and the round-6 result into the training objective, instead of correcting a finished model afterwards. Everything below has been seen:

- **Pooled LOBO macro-F1 (104 gated endpoint sessions, Ripe/Overripe).** CDR-RO 0.661; K1 0.567; A1 0.649; A1ft 0.730; A5 0.739; F1 0.731; F3 0.692; **R6 (anchored F3) 0.759**; F3-med 0.721.
- **Round-6 decision.** R6 vs CDR-RO +9.9 pp [−1.1, 20.2], p 0.077: not beneficial. R6 vs F3 +6.7 pp [−0.1, 13.9]. R6 vs K1 +19.2 pp [8.8, 30.0].
- **Per batch (Immature / Mature / Overmature).** CDR-RO 0.558 / 0.829 / 0.370; F3 0.639 / 0.726 / 0.529; R6 0.698 / 0.756 / 0.709. R6 shifts −1.01 / +1.09 / −0.74 logits.
- **Post-hoc ranking diagnostic (round 5).** Overmature within-batch AUC: CDR-RO 0.45, K1 0.39, F3 0.90, F1 0.82, A1 0.80. Immature: every arm 0.49–0.75. Oracle-threshold Overmature macro-F1 of F3: 0.85.
- **Label-free score rules on the endpoint sessions' own score distribution** (median-centering, train-prior quantile, prior-shift EM) have been evaluated post hoc; EM diverged to one class. **No rule of that family is used in this round.**
- **Batch decodability.** Every frozen embedding decodes harvest batch from intake units at balanced accuracy 0.81–1.00. The Immature batch is the only batch captured with the EOS R50 camera.
- **Precision.** With 104 sessions in 103 fruit clusters the paired bootstrap CI has a half-width of about ±10 pp; a true +10 pp effect has about 50 % power under the decision rule.

**Consequences for this plan.**
- The choice of the F3 blocks, of intake information and of a ranking objective was made after seeing that F3 ranks the Overmature batch and that an intake-anchored shift raised it. Pre-registration protects against choosing among heads, centring variants, thresholds or subsets after seeing the *round-7* results; it does not make the design independent of the data.
- Round 7 is the seventh analysis of the same 104 sessions. Its result, whatever it is, is exploratory and is **the best-or-not of a sequence**; no interval of this round accounts for that.
- The claims this round can support are restricted in advance (Section 5): a macro-F1 superiority claim only under the unchanged rule, and two descriptive statements about ranking and offset that are reported with their numbers whatever they show.

**Final-round statement, revised again.** The round-3 plan called round 3 final; rounds 4, 5 and 6 revised that statement, and round 7 revises it once more. Unlike rounds 4–6, the reason is not a new diagnostic but a model-design question. **The paper will report all seven rounds**, in order, with every rule and every outcome, and will state that rounds 1–7 were all evaluated on the same sessions. The new-harvest protocol (`docs/protocol/PROTOCOL_new_harvest.md`) and its primary contrast (anchored fusion vs CDR-RO) are **not changed** by this round, whatever its outcome; if RtA is to be tested on a new harvest it is added there as a secondary arm in a dated amendment.

**Hypothesis.** The round-5 embedding fusion fails in a held-out batch because (i) each batch carries an additive offset in embedding space that is visible in its label-free intake sessions, and (ii) a head trained with pooled cross-entropy is rewarded for separating batches whose Ripe/Overripe ratios differ. If so, a head that sees only intake-centred embeddings and is trained only on Ripe–Overripe pairs **within the same batch** cannot use an additive batch offset, and its decision threshold, expressed relative to the batch's own intake sessions, transfers to a held-out batch without a post-hoc shift.

## 1. Population and protocol information (unchanged from rounds 4–6)

- **Population.** The 104 endpoint sessions of round 4 (Immature 34, Mature 41, Overmature 29; Ripe/Overripe per batch 23/11, 21/20, 12/17). Every one has all four sensors and a verified intake recording of the same fruit.
- **Gating.** Intake sessions are predicted Unripe; endpoint sessions are classified Ripe or Overripe. The false-Unripe rate is 0 by construction.
- **Intake (anchor) sessions.** The label-free rule of round 1 (`results/v21/anchor_sessions.csv`, `anchor == True`): 162 sessions (Immature 59, Mature 54, Overmature 49); 159 labelled Unripe and 3 Ripe (Overmature), kept as realistic anchor contamination. No anchor label enters any computation.
- **Excluded inputs.** No method uses storage time, the time between visits, end-of-storage weight, weight loss, or any endpoint label or endpoint score of the held-out batch.
- **Training data per split.** Exactly the round-5 training rows: the non-intake sessions labelled Ripe or Overripe (common label) of the training fruits that have a baseline. Intake sessions are never training rows; they enter only the centring means.
- **Splits.** Primary: leave-one-batch-out (LOBO). Secondary: the p0 fruit-grouped 5-fold split, as a within-batch cost check only (Section 4).

## 2. Inputs (fixed)

- Frozen embeddings: the round-5 files, on the GPU host. Session-level embeddings of the four F3 blocks: A1 BEATs (W150; mean over knocks within recording, then over recordings), V-thermal DINO, V-ms DINO (8-band concatenation), V-rgb DINO (mean over images), exactly as `probe_blocks` builds them for F3.
- References: CDR-RO and K1 from the round-1 and round-4 session probabilities; F3 and R6 from `results/round6/probe_sessions_with_intake.parquet`.

## 3. Model (fixed; no further candidates will be added)

**Step 1 — intake centring (label-free).** For every batch g (training or held-out) and every block m, let μ(g, m) be the mean session embedding of block m over the intake sessions of batch g. Every session embedding x(m) of batch g is replaced by x(m) − μ(g, m). The held-out batch contributes only its intake sessions to its own μ.

**Step 2 — block scaling (as F3).** Each centred block is standardised with the mean and SD of the training rows of the split and scaled by 1/√dim; the four blocks are concatenated to a vector u. No other dimensionality reduction.

**Step 3 — within-batch ranking head.** A linear scorer f(u) = w·u **without intercept**. For every training batch g, all ordered pairs (i, j) with i Overripe and j Ripe **in the same batch g** form difference vectors d = u_i − u_j. w minimises the L2-regularised pairwise logistic loss Σ_g (1/P_g) Σ_pairs log(1 + exp(−w·d)), where P_g is the number of pairs of batch g, so that each training batch has equal weight. Implementation: scikit-learn `LogisticRegression(fit_intercept=False, penalty="l2")` on {(d, +1), (−d, −1)} with sample weights 1/P_g. C ∈ {0.001, 0.01, 0.1} (the round-5 grid), chosen by inner fruit-grouped 3-fold CV on the training fruits only; inner criterion: the mean over training batches of the within-batch AUC of the inner out-of-fold scores. Ties are broken towards the smaller C. Pairs across batches are never formed.

**Step 4 — threshold in intake-centred coordinates.** From the inner out-of-fold scores of the selected C, for every training batch g the midpoint t_g = ½ (mean f of its Ripe sessions + mean f of its Overripe sessions). The threshold is t = mean of t_g over the two training batches. An endpoint session of the held-out batch is predicted Overripe if f(u) > t, otherwise Ripe. **No statistic of the held-out batch's endpoint sessions and no post-hoc shift is used**; the only held-out information is μ of Step 1.

**Arms of this round.**
- **R7 = Steps 1–4 on the four F3 blocks. Primary arm.**
- R7-nc: Steps 2–4 without Step 1 (no intake centring). Ablation of centring.
- R7-ce: Steps 1–2, then the round-5 `embedding_logreg` head (pooled class-balanced cross-entropy, intercept, threshold 0.5). Ablation of the ranking head.
- R7-A: Steps 1–4 on the A1 BEATs block only. Audio-only variant (one phone recording per fruit and occasion).
- R7-noRGB: Steps 1–4 without the V-rgb block. Camera-confound control.
- **References (gated, same 104 sessions):** CDR-RO 0.661, K1 0.567, F3 0.692, R6 0.759, recomputed and asserted equal to the shipped values.

## 4. Evaluation

**Primary score.** LOBO pooled macro-F1 over Ripe and Overripe on the 104 sessions.

**Secondary scores (reported, unadjusted).**
- Per-batch macro-F1, Ripe recall and Overripe recall for every arm and reference, with Overmature Ripe recall reported explicitly.
- **Ranking.** Within-batch AUC (Overripe vs Ripe) of the held-out batch for R7, its ablations, F3, CDR-RO and K1, with the fruit-cluster bootstrap 95 % CI, and the unweighted mean of the three per-batch AUCs.
- **Offset.** For every arm, per held-out batch: the fraction of endpoint sessions predicted Overripe next to the true fraction (0.32 / 0.49 / 0.59), and the macro-F1 at the oracle threshold. The difference between the fixed-threshold and oracle macro-F1 is the *threshold loss* of the arm in that batch.
- **Batch decodability after centring.** The round-5 probe (logistic regression on intake sessions, fruit-grouped 5-fold CV, balanced accuracy for harvest batch) on the concatenated blocks before and after Step 1. Step 1 equalises the batch means by construction, so a drop is expected and is **not** evidence about ripeness predictions; it is reported to show how much batch information beyond the mean remains.
- **p0 cost check.** R7 and R7-ce under the p0 fruit-grouped 5-fold split (centring means from all intake sessions of each batch; label-free), next to the round-5 p0 score of F3. Descriptive; not part of the decision.

**Uncertainty.** As in rounds 1–6: fruit-cluster bootstrap (2,000 resamples, seed 42); fruit-level permutation (20,000, seed 42); clustered McNemar for accuracy. Code: `evaluate.paired`, `evaluate.holm`, `evaluate.endpoint_scores`, unchanged.

## 5. Primary contrast, decision rule and multiplicity

**One primary contrast: R7 vs CDR-RO.** Holm adjustment across one contrast is the identity, and the Holm column is still written.

R7 is called **beneficial** only if its contrast shows all three of:
- a pooled LOBO gain of at least 3 pp;
- a bootstrap 95 % CI excluding zero;
- a (Holm-adjusted) permutation p < 0.05.

Otherwise it is reported as not beneficial, whatever its direction. The rule is the rule of rounds 2–6, unchanged.

**Conditions for specific wording** (unadjusted, reported with their numbers whether met or not):
- **"the threshold transfers without a post-hoc shift":** the threshold loss of R7 is at most 5 pp in **each** of the three held-out batches.
- **"within-batch ranking training does not cost ranking":** the mean of the three per-batch AUCs of R7 is not more than 0.03 below that of F3.
- **"intake centring is the active component"** / **"the ranking head is the active component":** R7 − R7-nc, respectively R7 − R7-ce, is at least 3 pp pooled with a bootstrap CI excluding zero. If neither holds, the paper states that the components could not be separated at n = 104.

**No wording about superiority over R6 will be used.** R7 vs R6 is reported as exploratory; R6 was itself selected post hoc.

**Exploratory (all unadjusted, labelled as such):** R7 vs F3, R7 vs R6, R7 vs K1; R7-A and R7-noRGB vs CDR-RO and vs R7; all p0 contrasts.

## 6. Checks

**Controls.**
- **Identity.** `fit7` rebuilds the F3 session blocks with the round-5 code path and asserts that the uncentred, round-5-headed F3 endpoint probabilities equal `results/round6/probe_sessions_with_intake.parquet` (max |diff| < 1e−9). If the assertion fails, nothing is evaluated.
- **References.** `evaluate7` asserts CDR-RO 0.661, K1 0.567, F3 0.692 and R6 0.759 on the 104 sessions.
- **Centring.** After Step 1 the intake mean of every block and batch is zero (max |mean| < 1e−9, asserted); the number of intake sessions behind every μ is written (expected 59 / 54 / 49).
- **Pairs.** No difference vector joins two batches (asserted); the pair counts per split and batch are written.

**Leakage audit.** The audit asserts that the round-7 code reads no storage, visit-interval or weight variable; that intake sessions are never training rows; that no label of the held-out batch enters any mean, pair, threshold or selection; and that the held-out batch contributes only its intake-session embeddings to μ.

**Information seen before evaluation.** `fit7` prints only arm names, row and pair counts, the selected C, run times and the identity-check maximum difference. Scores and thresholds are not inspected before `evaluate7` runs once.

**Structural smoke test.** Before the tag, `fit7` → `evaluate7` is run once locally on random embeddings of the real shapes, including a negative test in which a cross-batch pair is injected and the pair assertion must fail. Its scores are meaningless and are not inspected.

**Tags.** `prereg-round7` (this plan and the code) and `pre-evaluation-round7` (frozen predictions and the deviation list) are both **pushed and verified on the remote** before the next step starts.

**Outputs.** `results/round7/`: `rta_sessions.parquet` (scores f, thresholds, predictions, with intake sessions), `fit7_identity_check.json`, `scores.csv`, `primary_contrast.csv`, `secondary_contrasts.csv`, `ranking_offset_diagnostics.csv`, `batch_decodability_centred.csv`, `audit.json`, then `RESULTS_SUMMARY.md` and `DEVIATIONS.md`. Code: `src/foundation/round7.py` (`fit7`, `evaluate7`); the round-5 and round-6 functions are unchanged in behaviour. `fit7` runs on the GPU host (CPU), where the embeddings are; `evaluate7` runs in the repository.

## 7. What a result would mean, and what to expect

| Result | Interpretation |
|---|---|
| R7 beneficial | Within these three batches, a head trained on within-batch pairs of intake-centred embeddings beats the protocol-gated classifier. Exploratory: the seventh analysis of the same sessions, designed after the six others; a new harvest is needed to confirm. |
| R7 not beneficial, wording conditions on threshold and ranking met | The model removes the threshold problem by construction and keeps the ranking, but the pooled gain is not resolved at n = 104. Reported as a design that is consistent with the diagnosis, not as a superior classifier. |
| R7 not beneficial, wording conditions not met | An additive intake offset does not describe the batch shift of these embeddings (or the Immature batch, which no arm ranks, dominates). Together with rounds 1–6, the limit is the data: three batches confounded with maturity, camera and harvest. |

**Expected outcome.** R6, a post-hoc correction of the same blocks, reached +9.9 pp [−1.1, 20.2]. RtA uses the same information (the four blocks and the intake sessions) and has about 63–75 training sessions per split for a head on several thousand dimensions, so a gain of **+3 to +10 pp is the plausible range and would not meet the rule**; a result below R6 is as likely as one above it. In the Immature batch no arm of rounds 1–6 ranked the sessions (AUC 0.49–0.75), and nothing in this design is expected to change that. The most likely report is therefore the second row of the table: **"consistent with the diagnosis, not confirmed"**. If that is the outcome, the paper presents RtA as the seventh round with the same weight as the others, not as a proposed classifier, and the confirmatory test remains the new-harvest design (about 200 endpoint sessions for +10 pp, 400 for +7 pp, with the number of independent harvests fixed first).

## 8. Implementation notes (fixed before the tag)

These notes record choices the text above leaves open; they were fixed in `src/foundation/round7.py` before the tag and before any fit.

- **Section 0 check.** Every number in Section 0 was checked against `results/round5/scores.csv`, `results/round5/posthoc_auc_ci.csv`, `results/round5/posthoc_shift_diagnostic.csv`, `results/round5/batch_decodability.csv`, `results/round6/scores.csv`, `results/round6/primary_contrast.csv`, `results/round6/secondary_contrasts.csv`, `results/round6/audit.json` and `docs/protocol/power_analysis.csv`. For precision: the CI half-width at 104 sessions is 9.6 pp, and the power for a true +10 pp is 0.44 (simulation) to 0.53 (analytic model).
- **Penalty.** scikit-learn 1.9.1 deprecates the `penalty` argument; the head uses `LogisticRegression(fit_intercept=False, C=C, l1_ratio=0, max_iter=5000)`, which is the same pure-L2 objective.
- **Inner CV of the ranking head.** Inner folds are `GroupKFold(3)` over the training fruits on the Step-2 vectors of the split (the block standardisation is fitted once on the split's training rows, as for F3). In every inner fold, pairs are formed only among the inner-training rows and only within a batch.
- **p0 threshold.** Each p0 training split contains all three batches, so t is the mean of t_g over the three training batches.
- **Scores used by the diagnostics.** R7, R7-nc, R7-A, R7-noRGB: f(u). R7-ce: log p(Overripe)/p(Ripe) with threshold 0 (equivalent to the 0.5 boundary). References: the R/O logits of CDR-RO, K1 and F3; R6: the anchored F3 logit, whose within-batch AUC equals that of F3.
- **Oracle threshold.** The maximum macro-F1 over thresholds at the batch's unique scores (as `src/foundation/posthoc_shift.py`).
- **AUC interval.** Fruit-cluster bootstrap within the held-out batch, 2,000 resamples, seed 42; resamples with one class are redrawn.
- **Intake rows.** As in round 6, every fitted model also scores all intake sessions; these rows are outputs only and are never training rows.
- **Batch decodability** (Section 4) is computed in `fit7`, where the embeddings are, and written without being printed.
- **Smoke test.** The raw embedding files and their indexes are on the GPU host, so the local smoke test uses random session-level blocks of the real shapes (768 / 768 / 6,144 / 768) for the 268 four-sensor sessions; the round-5 block-building path is exercised by the identity control of `fit7` on the host.

