# Analysis plan, round 6: intake-anchored logit shift of the round-5 embedding fusion under batch-disjoint evaluation

Status: **written and committed before the anchored variant below was computed or scored.**
- The tag `prereg-round6` precedes every result file of this round (`results/round6/`).
- Deviations will be listed in `results/round6/DEVIATIONS.md` and in the paper.
- No new embedding, encoder, head or training set is introduced. The round uses the round-5 frozen embeddings (md5 in `docs/ANALYSIS_PLAN_round5.md`, Section 8) and the round-5 heads, refitted with identical code, seeds and training rows (Section 3).

## 0. Disclosure

**Where this round comes from.** Round 6 was designed after the round-5 evaluation and after a post-hoc diagnostic that was **not** pre-registered: `results/round5/POSTHOC_shift_diagnostic.md` (computed 2026-11-03 from `results/round5/probe_sessions.parquet`, the A1ft files and the round-1/4 references). Everything in that file has been seen, in particular:

- **Within-batch AUC (Overripe vs Ripe) and oracle-threshold macro-F1.** In the Overmature batch the task-specific CNNs have no ranking signal (CDR-RO AUC 0.45, K1 0.39) while the foundation arms do: A1 BEATs 0.80, A1ft 0.75, A5 wav2vec2 0.79, F1 0.82, **F3 0.90**, V-ms DINO 0.78; V-thermal DINO 0.44 and V-rgb DINO 0.59 do not. Their mean p(Overripe) in that batch is 0.67–0.87, so the 0.5 threshold assigns almost everything to Overripe (Overmature argmax macro-F1: F3 0.53, F1 0.37, A1 0.45). With an **oracle** (label-using) threshold the Overmature macro-F1 would be F3 0.85, F1 0.79, A1 0.73, A5 0.76, A1ft 0.74. In the Immature batch every arm ranks weakly (AUC 0.49–0.75; oracle 0.57–0.75).
- **Label-free score-shift rules applied within the held-out batch (pooled LOBO macro-F1, Δ vs CDR-RO 0.661).** Median-centering of the within-batch scores: A1 0.72 (+6.0 pp [−5.0, 17.0]), A1ft 0.68 (+2.2), A5 0.74 (+8.0 [−2.7, 18.3]), F1 0.70 (+4.1), **F3 0.72 (+6.0 [−6.1, 17.4])**. Train-prior quantile: A1 0.69, A1ft 0.65, A5 0.73, F1 0.67, F3 0.71. Prior-shift EM: 0.48–0.66 (diverges to one class). Per batch, median-centering moves F3's Overmature macro-F1 from 0.53 to 0.79 (Ripe recall 0.17 → 0.83) and F1's from 0.37 to 0.72; in the Immature batch its effect is −16 pp (F1), −8 pp (A1ft), +14 pp (A1), +3 pp (A5) and 0 (F3). No rule brings a pooled CI above zero at n = 104.

**Consequences for this plan.**
- Score-level **median-centering is already evaluated** and is therefore reported in this round as exploratory only. It can never be called confirmatory on these sessions.
- The **intake-anchored logit shift** of Section 3 has **not** been computed for any round-5 arm. It is a different rule (it uses the label-free intake sessions of the held-out batch, as round-1 M1/M3 did, instead of the endpoint sessions' own score distribution), and it is the single pre-registered contrast of this round. It was chosen because it uses only protocol information already used in round 1 and because it does not look at the endpoint sessions at all.
- The choice of **F3 as the primary arm** was made after seeing F3's Overmature AUC (0.90) and oracle macro-F1 (0.85). Pre-registration therefore protects against choosing among arms, shift rules, thresholds or subsets after seeing the *anchored* results; it does not make the design independent of the data. A new harvest is required for confirmation.
- **Known before this plan** (104 LOBO endpoint sessions, gated, macro-F1 over Ripe/Overripe): CDR-RO 0.661; K1 0.567; A1 0.649; A1ft 0.730; A5 0.739; F1 0.731; F3 0.692 (round-5 `scores.csv`); Q1 (A1 vs CDR-RO) −1.1 pp, Q2 (F1 vs CDR-RO) +7.0 pp [−2.6, 16.8], Holm p 0.337, neither beneficial. Every frozen embedding decodes harvest batch from intake units at balanced accuracy 0.81–1.00 (`batch_decodability.csv`). The Immature batch is the only batch captured with the EOS R50 camera, so camera and batch are confounded there.

**Final-round statement, revised again.** The round-3 plan called round 3 final; rounds 4 and 5 revised that statement, and round 6 revises it once more. The reason is the same as before: an unplanned diagnostic suggested a specific mechanism (decision-threshold shift, not loss of ranking signal) that can be tested with information already in the protocol. **The paper will report all six rounds**, in order, with every rule and every outcome, and will state that rounds 1–6 were all evaluated on the same 104 sessions. No further round on these sessions is planned; any further hypothesis is to be tested on a new harvest, following a prospective protocol to be written as `docs/PROTOCOL_new_harvest.md`.

**Hypothesis.** The round-5 embedding fusion F3 ranks Overmature-batch sessions correctly but with an Overripe-shifted score offset that is a property of the batch, not of the fruit. If so, the offset is visible in the batch's label-free intake sessions (all fruit Unripe at harvest by protocol), and removing the intake-to-intake logit difference between the held-out and the training batches recovers the Overmature classification without any endpoint information from the held-out batch.

## 1. Population and protocol information (unchanged from rounds 4–5)

- **Population.** The 104 endpoint sessions of round 4 (Immature 34, Mature 41, Overmature 29; labels Ripe/Overripe per batch 23/11, 21/20, 12/17). Every one has all four sensors and a verified intake recording of the same fruit.
- **Gating.** Intake sessions are predicted Unripe; endpoint sessions are classified Ripe or Overripe. The false-Unripe rate is 0 by construction.
- **Intake (anchor) sessions.** The label-free rule of round 1, fixed in `docs/ANALYSIS_PLAN_anchor_physics.md` Section 1 and materialised in `results/v21/anchor_sessions.csv` (`anchor == True`): the earliest capture session of each fruit, within 4 days of the recorded harvest date. There are 162 anchor sessions (Immature 59, Mature 54, Overmature 49), all with a verified intake recording; 159 are labelled Unripe and 3 Ripe (Overmature batch). The 3 are kept as realistic anchor contamination, as in round 1. Labels are never used to define or select anchors, and no anchor label enters any computation of this round.
- **Excluded inputs.** No method uses storage time, the time between visits, end-of-storage weight, weight loss, or any endpoint label or endpoint score of the held-out batch.
- **Training data per split.** Exactly the round-5 training rows: the non-intake sessions labelled Ripe or Overripe (common label) of the training fruits that have a baseline. Intake sessions are never training rows.
- **Splits.** Primary: leave-one-batch-out (LOBO). The p0 fruit-grouped 5-fold split is **not** evaluated for the anchored arms (Section 4).

## 2. Inputs (fixed)

- Frozen embeddings: the round-5 files (plan round 5, Section 8), on the GPU host.
- Round-5 session probabilities: `results/round5/probe_sessions.parquet` (endpoint sessions only; used as the identity control, Section 6).
- References: CDR-RO (round-1 M0 knock CNN) and K1 (round-4 current-knock CNN), from the round-1 and round-4 session probabilities.

## 3. Models (fixed; no further candidates will be added)

**Base arms (round 5, unchanged heads and features).** F3 (session-level embedding fusion of A1 BEATs and the three DINO image arms, each block standardised on the training rows and scaled by 1/√dim, same `embedding_logreg` head and C grid), A1, A5, and F1 (equal-weight probability mean of A1, V-thermal-DINO, V-ms-DINO, V-rgb-DINO). They are refitted with the identical code path (`probe` / `probe_blocks` with `intake_rows=True`), identical training rows, estimator, grid, inner CV and seeds; the only change is that the fitted model is **also applied to the intake sessions** of all three batches. The refitted endpoint probabilities must be identical to round 5 (Section 6).

**Session score.** For an arm with session probabilities (p_Ripe, p_Overripe) (mean over the session's units, as in round 5), the R/O logit is z = log p_Overripe − log p_Ripe (probabilities clipped to [1e−12, 1 − 1e−12]).

**Anchored logit shift (round-1 M3 rule, applied to session scores).** For held-out batch b, with the model trained on the two other batches:

  s_b = mean{ z(intake sessions of the two training batches) } − mean{ z(intake sessions of batch b) }

and every endpoint session of batch b is scored by z + s_b; the gated prediction is Overripe if z + s_b > 0, otherwise Ripe. All three quantities come from the same model of split b. The training-batch intake sessions are scored by that model out of its training rows (they are never training rows), so s_b is the intake-to-intake offset the model itself produces between the batches it was trained on and the batch it is applied to. Under the round-1 definition the shift was computed on the mean logit vector of anchor captures; here it is the scalar R/O logit of anchor sessions, because every arm of this round is a gated two-class R/O model.

**Arms of this round.**
- **R6 = F3 with the anchored shift. Primary arm.**
- A1-anch, A5-anch, F1-anch: the same shift rule applied to A1, A5 and F1 (for F1, z is the logit of the F1 probability mean, and the intake means are those of the F1 probability mean; the intake sessions used are those for which all four component arms have a prediction). Exploratory.
- F3-med, A1-med, A5-med, F1-med: score-level median-centering within the held-out batch (Overripe if z exceeds the median z of the batch's endpoint sessions). **Already seen post hoc**; exploratory, reported for completeness next to the anchored rule.
- **A1ft is not shifted.** `beats_ft.py` predicted only the endpoint knocks of each held-out split (0 finite intake knocks in every LOBO file), so A1ft has no intake logits of its own. Using the frozen-A1 intake offset as a proxy was considered and rejected: the shift corrects the offset of a specific model's scores, and the fine-tuned model's logit scale and offset differ from the frozen probe's (A1ft − A1 was +8.1 pp in round 5), so a proxy shift would not be the M3 rule. A1ft would need a GPU re-run on intake knocks, which would be a new fit; it is not done in this round.

**References (gated, same 104 sessions).** CDR-RO 0.661 and K1 0.567, recomputed from the round-1 and round-4 session probabilities and asserted equal to the round-4 values. Unshifted F3, A1, A5 and F1 are recomputed from the refitted probabilities and asserted equal to the round-5 scores.

## 4. Evaluation

**Primary score.** LOBO pooled macro-F1 over Ripe and Overripe on the 104 sessions.

**Secondary scores (reported, unadjusted).**
- Per-batch macro-F1, Ripe recall and Overripe recall for every arm of Section 3 and the references, with Overmature Ripe recall reported explicitly.
- The per-batch shifts s_b and the number of intake sessions behind each mean (label-free; written to `audit.json`).

**p0 (fruit-grouped 5-fold) is not evaluated for the anchored arms.** Under p0 every test fold contains fruits of all three batches and every training fold contains intake sessions of the same three batches, so a per-batch intake shift computed within a p0 fold compares intake sessions of the same batch across folds: it measures fold-to-fold sampling noise, not a batch offset, and there is no batch shift to correct by design. The unshifted p0 scores of F3, A1, A5 and F1 are already in `results/round5/scores.csv` and are not recomputed. `fit6` still writes the p0 rows of the refitted arms (with intake sessions) so that the identity control covers both splits.

**Uncertainty.** As in rounds 1–5: fruit-cluster bootstrap (2,000 resamples, seed 42); fruit-level permutation (20,000, seed 42); clustered McNemar for accuracy. Code: `evaluate.paired`, `evaluate.holm`, `evaluate.endpoint_scores`, unchanged.

## 5. Primary contrast, decision rule and multiplicity

**One primary contrast: R6 (anchored F3) vs CDR-RO.** Holm adjustment across one contrast is the identity, and the Holm column is still written.

R6 is called **beneficial** only if its contrast shows all three of:
- a pooled LOBO gain of at least 3 pp;
- a bootstrap 95% CI excluding zero;
- a (Holm-adjusted) permutation p < 0.05.

Otherwise it is reported as not beneficial, whatever its direction.

**Condition for specific wording** (unadjusted, reported): **"the shift correction recovers the Overmature batch"** may be written only if R6's Overmature macro-F1 ≥ 0.70 **and** pooled R6 − F3 ≥ 3 pp with a bootstrap CI excluding zero. If only one holds, the result is described as a partial or batch-specific change.

**Exploratory (all unadjusted, labelled as such in the files and the paper):**
- A1-anch, A5-anch and F1-anch vs CDR-RO, and each vs its unshifted arm (A1-anch vs A1, A5-anch vs A5, F1-anch vs F1);
- R6 vs F1-anch, R6 vs K1;
- the median-centering arms vs CDR-RO and R6 vs F3-med (already seen post hoc; reported so the reader can compare the pre-registered rule with the one that generated the hypothesis).

## 6. Checks

**Controls.**
- **Identity.** `fit6` refits only F3, A1, A5, V-thermal-DINO, V-ms-DINO and V-rgb-DINO, with the round-5 code path, and asserts that the endpoint-session probabilities of every refitted arm and split are identical (max |diff| < 1e−9) to `results/round5/probe_sessions.parquet`. The refit is required only to obtain the intake-session predictions; if the assertion fails, nothing is evaluated.
- **Unshifted scores.** `evaluate6` asserts that unshifted F3, A1, A5 and F1 reproduce the round-5 LOBO pooled macro-F1 values, and that CDR-RO and K1 reproduce the round-4 values (0.661, 0.567).
- **Intake coverage.** Every held-out batch has intake sessions on both sides of the shift (asserted, counts written).

**Leakage audit.** The audit asserts that the round-6 code reads no storage, visit-interval or weight variable, that intake sessions are never training rows (by construction of `probe`), and that the held-out batch contributes only its label-free intake sessions to the shift.

**Information seen before evaluation.** `fit6` prints only arm names, row counts, run times and the identity-check maximum difference. The intake-session predictions and the shifts are not inspected before `evaluate6` runs once.

**Structural smoke test.** Before the tag, `fit` → `fit6` → `evaluate6` was run once locally on random embeddings of the real shapes (built from the round-5 index parquets), including a negative test in which a perturbed reference parquet makes `fit6` fail its identity assertion. Its scores are meaningless and were not inspected.

**Outputs.** `results/round6/`: `probe_sessions_with_intake.parquet`, `fit6_identity_check.json`, `scores.csv`, `primary_contrast.csv`, `secondary_contrasts.csv`, `audit.json`, then `RESULTS_SUMMARY.md` and `DEVIATIONS.md`. Code: `src/foundation/round5.py` (`fit6`, `evaluate6`; the round-5 functions are unchanged in behaviour). `fit6` runs on the GPU host (CPU), where the embeddings are; `evaluate6` runs in the repository.

## 7. What a result would mean, and what to expect

| Result | Interpretation |
|---|---|
| R6 beneficial | Within these three batches, the fusion's failure in the held-out batch is a score offset that the batch's own intake sessions reveal; label-free intake anchoring (protocol information) corrects it. Exploratory: the primary arm was chosen after a post-hoc diagnostic on the same sessions, so a new harvest is needed to confirm. |
| R6 not beneficial, wording condition met | The Overmature batch is recovered but the pooled gain is not resolved at n = 104; reported as batch-specific, consistent with the mechanism, not confirmed. |
| Neither | The intake offset does not track the endpoint offset (or the Immature batch loses what the Overmature batch gains). Together with rounds 1–5, the limit is the data: three batches confounded with maturity, camera and harvest. |

**Expected outcome.** With 104 sessions in 103 fruit clusters, the fruit-cluster bootstrap CI of a paired macro-F1 difference has a half-width of about ±10 pp (the round-5 primary and audio-family contrasts had CI widths of 19–22 pp; all LOBO secondary contrasts 15–25 pp). The post-hoc median-centering of F3 gave +6.0 pp. A gain of +5–8 pp for R6 is therefore the plausible range and would **not** meet the rule; it would be reported as **"consistent with the mechanism, not confirmed"**, and the paper would state that a confirmatory test needs about four times the number of fruits or a new harvest in which the intake anchoring is applied once, prospectively.
