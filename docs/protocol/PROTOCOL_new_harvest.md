# Protocol: confirmatory new-harvest study of durian ripeness sensing

Status: draft for pre-registration (to be tagged `prereg-harvest2` before the first fruit is harvested).
Companion documents: `power_analysis_methods.md`, `power_analysis.csv`, `power_curve.png`, `harvest_variance_scenarios.csv`.
Every requirement below is followed by the finding from rounds 1–5 (Zenodo 10.5281/zenodo.18603796 data, repository
`durian-ripeness-audit`, tags `prereg-anchor-physics`, `prereg-round2/3/4/5`) that motivates it.

## 1. Purpose and primary question

1.1 The 2025 dataset supported five pre-registered rounds; none met the rule (gain ≥ 3 pp macro-F1 AND fruit-cluster bootstrap 95 % CI
excluding 0 AND Holm p < 0.05). The best pooled LOBO contrast was the four-modality foundation fusion F1 vs CDR-RO, +7.0 pp
[−2.6, 16.8], Holm p 0.337 (round 5, Q2). The CI half-width at n = 104 endpoint sessions is ≈ 10 pp, so an effect of the plausible size
(+5 to +10 pp) cannot be confirmed on the existing data (power 0.17–0.53; `power_analysis.csv`, row n = 104).

1.2 The new study has one primary question: does the arm selected by round 6 (frozen before this protocol is tagged; default F3-anchored
session-embedding fusion) exceed the knock-CNN reference CDR-RO in macro-F1 over Ripe/Overripe on endpoint sessions of fruit from
harvests the model has never seen?

1.3 Secondary (exploratory) questions are listed in §8; they are not part of the decision rule.

## 2. Design: batch crossed with maturity stage

2.1 **Two or more harvest sources (orchards or harvest dates) × three maturity stages at harvest (Immature, Mature, Overmature) = ≥ 6
batches.** Each source supplies all three stages; each stage is supplied by all sources.
Rationale: in 2025 each maturity stage was a single harvest, so stage ≡ batch ≡ harvest date. Every frozen embedding decoded harvest batch
from intake units at balanced accuracy 0.81–1.00 (round 5, `batch_decodability.csv`), and the F1-vs-CDR-RO gain was +11 pp in Immature and
0 in Mature and Overmature: batch effects and stage effects cannot be separated. Crossing removes the confound and makes
leave-one-source-out a test of transfer to a new harvest rather than to a new stage.

2.2 **Evaluation split: leave-one-source-out (LOSO).** Every fold trains on all stages of the other source(s) and tests on all stages of the
held-out source. LOBO (leave-one-stage-batch-out) is retained only as a secondary split.
Rationale: LOBO in 2025 forced each model to extrapolate to a stage it had never seen; in the Overmature batch the task CNNs had AUC
0.39–0.45 and every foundation arm's mean p(Overripe) was 0.67–0.87, so the 0.5 threshold assigned nearly all sessions to Overripe
(round-5 post-hoc shift diagnostic). The deployment question is a new harvest, not an unseen stage.

2.3 **One phone pair and one camera for all batches.** The same iPhone and Android handset (ids recorded, §9) record every knock session;
one RGB camera body and lens, fixed settings, for every batch. Thermal and multispectral devices likewise fixed.
Rationale: in 2025 the Immature batch was the only batch photographed with the Canon EOS R50 (34/34 sessions), so camera ≡ batch; the
RGB arm could not be evaluated across batches, and F4 (F1 without RGB) had to serve as a camera-confound control (round 5).

2.4 **Capture days interleaved across batches.** Fruit from different sources and stages are measured on the same days in a schedule drawn
once before harvest (block randomization by day); within a day, recording order is randomized and written to the session table before
recording starts.
Rationale: in 2025 capture date was nested in batch, so any day-level effect (room acoustics, ambient temperature, operator) is absorbed
into the batch effect that the embeddings decode at 0.81–1.00 balanced accuracy.

2.5 **Same operator set and same room for all batches**, operator id recorded per recording.

## 3. Ground truth

3.1 **Classes fixed before collection.** Ripe and Overripe are defined by thresholds on firmness (penetrometer, N) and dry matter (%, oven
method) or TSS (°Brix), written in this protocol and tagged before harvest. The numeric thresholds are to be derived from the 2025
`durian_characteristics_cleaned.csv` distributions and the literature table (`TableS18`) and inserted here before tagging; Unripe is any
session before the fruit first meets the Ripe criterion.
Rationale: in 2025 the label of record (`Actual_Ripening_Status`) differed from the planned ripeness for 15 fruits and 2 fruits were
diseased; the Ripe/Overripe boundary was decided after the fact, and Ripe and Overripe were nearly inseparable by mean DM, TSS and firmness.
A label without a pre-specified criterion cannot be audited.

3.2 **Destructive reference sub-sample at every session.** On each capture day a pre-scheduled random subset of the fruit present
(target 15 %, drawn from the randomization list) is opened after sensing and measured: firmness (3 arils, penetrometer), DM, TSS.
These fruit leave the longitudinal cohort; their last session is their endpoint.
Rationale: without a per-session verifiable criterion the ground truth of the non-destructive sessions is unanchored (3.1); the sub-sample
calibrates the sensory panel (3.3) to the physical thresholds on the same days and in the same batches.

3.3 **Blinded two-rater sensory panel for non-destructive sessions.** Two trained raters independently classify each fruit at each session
(Unripe / Ripe / Overripe) using a pre-specified checklist (aroma, stem abscission, suture opening; tapping by ear is not allowed because the
knock recording is the primary sensor). Raters do not see sensor outputs, model predictions, the fruit's source/stage, or each other's
rating. Disagreements are resolved by the destructive criterion when available, otherwise by a third rater; the disagreement flag is kept.
Rationale: the 2025 `label_disagreement` field exists but ratings were not blind to batch or sensing.

3.4 **Labels are assigned by a person who has not seen any sensor output or model prediction**, and are committed to the repository
(hash recorded) before any model is run on the new data.

## 4. Longitudinal schedule

4.1 **Intake session within 1 day of harvest for every fruit**: knock recordings with both phones (≥ 5 knocks per phone, verified as
belonging to the fruit), RGB, thermal and multispectral images.
Rationale: the change-only knock arm (A1c, embedding difference from the fruit's own intake) was +12.4 pp [3.7, 21.6] over the shuffled
baseline within batches and +10.5 pp [−2.1, 22.7] under LOBO (round 5). Making the within-fruit-change hypothesis confirmatory requires a
verified intake recording for every fruit; in 2025 the population had to be gated to fruits with an intake recording (104 of 106 endpoints).

4.2 **Follow-up sessions every 2 days** (same sensors, same order rule) until the fruit meets the Overripe criterion or is destructively
sampled, whichever comes first. The first session at which a fruit meets the Ripe criterion and the first at which it meets Overripe are
its endpoint sessions.
Rationale: in 2025 almost every fruit has exactly one endpoint session, which fixed the evaluation population at ≈ 1 session per fruit and
made the fruit-cluster bootstrap behave as 1/√n_fruits (`power_analysis_methods.md` §2); two endpoints per fruit roughly doubles the
usable sessions for the same number of fruit, but they are clustered and do not double the effective n.

4.3 **Per-session environment record**: ambient temperature and relative humidity in the recording room, storage temperature and
humidity (logger, 10-min interval), storage container, days since harvest.
Rationale: none of these were recorded per session in 2025, so they cannot be excluded as the signal that makes batch decodable.

4.4 **Knock protocol fixed**: same knocker, same contact rule (equatorial, between spines), same phone positions and distances; written
and photographed before the first session.

## 5. Sample size

5.1 Numbers are from `power_analysis.csv` (fruit-cluster bootstrap half-width ≈ 97.6/√n_fruits pp; 80 % power that the 95 % CI excludes
0; one primary contrast). Endpoint sessions needed (≈ fruits, assuming ≈ 1 endpoint session per fruit as in 2025):

| True gain over CDR-RO | Endpoint sessions | With 6-batch design (range, see 5.2) |
|---|---|---|
| +10 pp | 197 | 98–197 |
| **+7 pp (planning target)** | **401** | 201–401 |
| +5 pp | 786 | 393–786 |

If two primary contrasts are pre-registered (Holm, α = 0.025 for the stronger), the +7 pp requirement rises to 486 sessions.

5.2 The range for the 6-batch design assumes that crossing source with stage halves the between-batch variance (standard error smaller by
up to √2); the present bootstrap cannot estimate that component from 3 batches, so **plan to the upper end**.

5.3 **Enrolment: 446 fruits (401 ÷ 0.9 for 10 % attrition — disease, damage, fruit consumed by the destructive sub-sample before an
endpoint), ≈ 75 fruits per source × stage cell with 2 sources, ≈ 50 per cell with 3 sources.** Per-stage proportions in 2025 were
33 / 39 / 28 %; equal cells are preferred in the new design.

5.4 Minimum acceptable: with ≈ 220 fruits the study is powered for +10 pp only, and the plan must say so.

5.5 **Harvests, not only fruits (added in v2.2.0).** The numbers above are a fruit-level illustration: they resample fruits of the three
2025 batches and do not represent variation of the true gain between harvests. If the gain varies between harvests with SD τ, the
standard error of the mean gain over *H* harvests cannot fall below τ/√*H* however many fruits are added; for +7 pp, 80 % power is attainable
with a finite number of fruits only from 5 harvests when τ = 5 pp (≈ 803 endpoint sessions with 8) and from 17 when τ = 10 pp
(`harvest_variance_scenarios.csv`, `src/protocol/harvest_scenarios.py`; same convention as `power_analysis.csv`, so τ = 0 gives 401 and 197).
The number of independent harvests (sources × dates, ideally across seasons) is therefore fixed first, and the harvest is the unit of
inference for the primary claim.

## 6. Pre-registration and verification workflow

6.1 **Analysis plan** `docs/ANALYSIS_PLAN_harvest2.md` specifies, before harvest:
- population: endpoint sessions (first Ripe, first Overripe) of fruit with a verified intake session and all four sensors;
- metric: macro-F1 over Ripe/Overripe (an endpoint predicted Unripe is an error), plus the false-Unripe rate;
- primary contrast: the round-6 selected arm (default F3-anchored) vs CDR-RO, trained and selected exactly as in round 6 and frozen at tag
  time (weight files and code hashes listed in the plan); one primary contrast unless power is budgeted for two (5.1);
- split: leave-one-source-out, pooled over held-out sources; LOBO and within-source fruit-grouped 5-fold CV as secondary;
- uncertainty: fruit-cluster bootstrap (2,000, seed 42), fruit-level permutation (20,000, seed 42), clustered McNemar, as implemented in
  `src/anchor_physics/evaluate.py`;
- rule: beneficial iff Δ ≥ 3 pp AND 95 % CI excludes 0 AND Holm p < 0.05;
- wording conditions for every sentence the manuscript may claim.
Rationale: this is the rule applied unchanged in rounds 1–5; changing it on the confirmatory data would void the confirmation.

6.2 **Tag sequence** (as in rounds 1–5): `prereg-harvest2` (plan + frozen models, pushed) → data collection → labels committed →
`pre-evaluation-harvest2` (all predictions written and committed before any score is computed, pushed) → `evaluate` run exactly once →
`results/harvest2/RESULTS_SUMMARY.md` with every number copied from the result files → `DEVIATIONS.md` listing every departure from the
plan, dated, with its reason.
Rationale: round 5's `pre-evaluation-round5` tag was committed locally but not pushed; the new protocol requires both tags to be pushed
before evaluation so that the sequence is externally verifiable.

6.3 No model is trained or tuned on the new data before the labels are committed and the pre-evaluation tag is pushed. Post-hoc analyses
(for example score-level median-centering within the test source, already tried on the 2025 data: F3 pooled 0.69 → 0.72, +6.0 pp
[−6.1, 17.4] vs CDR-RO) are reported in a separate `POSTHOC_*.md` and never enter the decision.

## 7. Data management

7.1 Session table columns (fixed before collection): `session_id, fruit_id, source, stage_at_harvest, harvest_date, capture_date,
days_since_harvest, capture_day_block, recording_order, operator_id, phone_ids, camera_id, thermal_id, ms_id, room_T, room_RH, store_T,
store_RH, destructive (0/1), firmness_N, DM_pct, TSS_brix, rater1, rater2, rater3, label_criterion (destructive|panel), label,
is_endpoint, endpoint_type (first_ripe|first_overripe)`.

7.2 Each recording file is named `<fruit_id>_<capture_date>_<device>_<k>` and listed in a manifest with its md5 at the time of capture;
the manifest is committed daily.
Rationale: in 2025 the sound-features table carried stale `session_id`/`date` fields and some recordings were assigned to the wrong
session (`data/sound_exclude_wrong_session.txt`, `sound_recording_manifest.csv`); a daily-committed manifest prevents this.

## 8. Secondary (exploratory) analyses, declared in advance

8.1 Change-only knock arm (A1c-type) vs CDR-RO and vs its shuffled-intake baseline, LOSO.
8.2 Cross-device audio: train on one phone, test on the other.
8.3 Batch decodability of each embedding from intake sessions, by source and by stage separately (expected if §2 succeeded: stage
decodable, source not).
8.4 Per-source and per-stage AUC of p(Overripe) and oracle-threshold F1 (diagnostic of the calibration shift seen in 2025 Overmature).
8.5 DM/TSS regression from the sensors on the destructive sub-sample.

## 9. Data release

9.1 Zenodo record with: raw recordings and images per modality, `checksums.md5` for every file, the session table (§7.1), a device table
(phone model / OS / app version, camera body / lens / firmware, thermal and multispectral device ids and calibration dates), the
randomization schedule, the destructive-measurement sheet, and the pre-registration tag hashes.

9.2 `README.md` defines Unripe / Ripe / Overripe exactly as in §3.1, states which label field is the label of record, lists excluded fruit
with reasons, and names the evaluation population (endpoint sessions with verified intake and four sensors), so that the 2025 ambiguity
between planned and actual ripeness status cannot recur.

## 10. Timeline and roles (to be filled before tagging)

Harvest windows per source; sensing team (2 operators); rater panel (2 + 1 reserve); destructive-lab person; data manager responsible for
daily manifest commits; analyst who holds the frozen models and does not touch the raw data until the pre-evaluation tag is pushed.
