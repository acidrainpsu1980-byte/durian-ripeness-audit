# Analysis plan, round 4: a fruit-specific acoustic baseline (paired intake–endpoint knock model)

Status: **written and committed before any round-4 model was fitted or scored.** The tag `prereg-round4` precedes every result file of
this round (`results/round4/`). Deviations will be listed in `results/round4/DEVIATIONS.md` and in the paper.

## 0. Disclosure: revision of the "final round" statement

The round-3 plan (tag `prereg-round3`) declared round 3 the final round of method development on this dataset. Round 4 revises that
statement. The paper had not been submitted anywhere when this plan was written, and the authors decided to test one further,
qualitatively different hypothesis. The paper will report:
- the original statement;
- this revision and its date;
- all four rounds in full.

Round 4 was designed after rounds 1–3 had been evaluated on the same sessions. Its results are therefore exploratory and need a new
harvest to confirm.

**What is known before this plan.** All scores are on the 106 leave-one-batch-out (LOBO) endpoint sessions, macro-F1 over
Ripe/Overripe:
- the unadapted knock CNN scored 0.464;
- the capture-day rule (CDR-RO) scored 0.656;
- AdaBN on intake knocks (M5) scored 0.652;
- the round-2 nested procedure scored 0.679 and the round-3 procedure 0.556;
- in the Overmature batch, Ripe recall was 0 for every rule except the quantile rules D2 and D4 and the two-class CNN C3;
- intake recordings differ between batches (median dominant frequency 726 / 495 / 519 Hz).

**Hypothesis.** Rounds 1–3 standardized each batch to the mean of its intake fruits. Round 4 references each fruit to its own intake
recording. The question changes from "what state does today's knock indicate" to "how far has this fruit's knock moved from its own
intake knock, and does that change indicate ripeness". A fruit-specific baseline is not guaranteed to remove batch effects; this round
tests whether it does.

## 1. Population and protocol information

- **Endpoint sessions.** These are the round-1 endpoint sessions: four-sensor sessions that are not intake sessions under the
  label-free intake rule (earliest session of the fruit, ≤ 4 days after the recorded harvest).
- **Population.** The 104 of the 106 endpoint sessions whose fruit has at least one verified intake recording: Immature 34,
  Mature 41, Overmature 29. The 2 Overmature endpoint sessions without an intake recording are excluded from every method and
  reported.
- **Baseline of a fruit.** All verified knocks of the fruit's intake recording(s), from both phones where available. Of the 104 endpoint
  sessions, 62 have an iPhone-only baseline and 42 an iPhone + Android baseline; the median is 31 baseline knocks per session.
- **Equal protocol information.** Every method receives the same protocol information. Intake sessions are predicted Unripe.
  Endpoint sessions are classified as Ripe or Overripe. No method uses storage time, the time between visits, end-of-storage
  weight, weight loss or any endpoint label of the test fruits.
- **Training data.** For each training fold, the training data are the non-intake sessions of the 164 analysis fruits that are
  labeled Ripe or Overripe (common label) and have a baseline, together with those fruits' intake knocks.

## 2. Models (fixed; no further candidates will be added)

**Architecture-matched knock CNNs (K1–K4).** The shared encoder E is the knock-CNN trunk of the paper: 4 convolution blocks with
global mean pooling, giving a 96-dimensional embedding with dropout 0.3. A linear two-class head h is applied to:

- **K1 current**: h(z_t);
- **K2 baseline**: h(z_0);
- **K3 concatenation**: h([z_t, z_0]);
- **K4 paired, proposed**: h([z_t, z_t − z_0]);
- **K5 change only**: h(z_t − z_0);
- **K6 shuffled baseline (control)**: K4 with each fruit's baseline replaced, in training and at test time, by the intake knocks of
  another fruit of the same harvest batch and the same primary fruit fold. The donor is fixed by a random derangement within each
  (batch, fold) group (numpy `default_rng(42)`, among fruits with a baseline), so donors never cross a train/test boundary in either
  design.

These arms were added after a prior-art check (`docs/novelty/novelty_memo_baseline.md`), before this plan was tagged:
- **K3 and K4.** With a linear head, [z_t, z_0] and [z_t, z_t − z_0] carry the same information. They are expected to differ
  only through optimization and dropout, and are reported as near-equivalent.
- **K5.** Tests the change alone.
- **K6.** Tests whether any gain is fruit-specific or only batch-level.

Training:
- each endpoint knock is paired with one intake knock of the same fruit, drawn at random in every mini-batch;
- both pass through the same encoder in one forward pass, with shared BatchNorm statistics;
- the endpoint label is the target;
- settings as in the paper: 20 epochs, batch 128, AdamW lr 2e-3, one-cycle max 3e-3, label smoothing 0.05, the paper's
  augmentation applied to both inputs, global input normalization over the training knocks, class-balanced weights, seed 42,
  CPU 3 threads.

At test time:
- z_0 is the mean embedding (eval mode) over all baseline knocks of the fruit;
- each endpoint knock gives a probability;
- recording probability = mean over its knocks; session probability = mean over its verified recordings.

**Feature-level comparators (L1–L5).** These are logistic regressions on recording features, with in-fold standardization, class
balancing, and C chosen from {0.01, 0.1, 1, 10} by inner fruit-grouped 3-fold CV, as in round 1.
- **Feature set F.** The round-1 physics features (f_dom, f_first, centroid, decay20, 1/Q, stiffness) plus mean MFCC 1–13.
- **Baseline F_0.** The fruit's mean of F over its verified intake recordings.

The five comparators are:

- **L1 current**: F_t + device;
- **L2 baseline**: F_0;
- **L3 concatenation**: [F_t, F_0] + device;
- **L4 differences**: log(f_dom,t / f_dom,0), log(f_first,t / f_first,0), centroid_t − centroid_0, decay20_t − decay20_0 and
  1/Q_t − 1/Q_0, + device. The stiffness ratio is omitted because it equals the squared frequency ratio with the same harvest
  weight.
- **L5 current + differences**: [F_t, L4 differences] + device. This is the feature-level analogue of K4.

Session probability = mean over the session's recordings.

**Elapsed-time comparator (T1, reference only).** A logistic regression, set up like L1–L5, on the days between the fruit's intake
session and the endpoint session.
- **Why it is needed.** The change vector partly encodes elapsed time, so T1 shows how much of any gain time alone provides.
- **Restriction.** Elapsed time is protocol information and is never an input to K1–K6 or L1–L5.

**References (gated, restricted to the same 104 sessions).**
- **CDR-RO**: the round-1 M0 knock CNN, with endpoints assigned the more probable of Ripe and Overripe.
- **M1-RO**: the round-1 anchored-standardization knock CNN, gated the same way.

## 3. Evaluation

**Primary score.** LOBO (train on two harvest batches, test on the third), pooled over the 104 endpoint sessions: macro-F1 over
Ripe and Overripe. With gating, the false-Unripe rate is 0 for every method.

**Secondary scores.**
- per-batch macro-F1, Ripe recall and Overripe recall;
- fruit-grouped 5-fold CV on the primary partition (`fold_fruit`) for K1–K4, L1–L5 and the references; for L1–L5 also the mean
  over the ten partitions;
- the cost of the baseline: extra recordings and knocks per fruit.

**Baseline-only diagnostic.** K2 and L2 are reported with bootstrap CIs. If the endpoint class can be predicted from the intake
recording alone, a latent relation is at work, such as size, harvest maturity or the planned endpoint design. In that case, any gain
of K3/K4 over K1 is interpreted with that caveat.

**Uncertainty.** As in rounds 1–3:
- fruit-cluster bootstrap (2,000 resamples, seed 42);
- fruit-level permutation (20,000, seed 42);
- clustered McNemar test for accuracy.

## 4. Primary contrasts, decision rule and multiplicity

Holm adjustment across the two contrasts:
- **P1**: K4 paired vs K1 current (architecture-matched; isolates the fruit-specific baseline);
- **P2**: K4 paired vs CDR-RO (the protocol-information baseline of rounds 1–3).

K4 is called **beneficial** only if both P1 and P2 show all three of:
- a pooled LOBO gain of at least 3 pp;
- a bootstrap 95% CI excluding zero;
- a Holm-adjusted p < 0.05.

Otherwise it is reported as not beneficial, whatever its direction.

**Condition for the word "fruit-specific."** The paper will describe a gain as fruit-specific only if K4 also exceeds the shuffled
baseline K6 by at least 3 pp with a bootstrap CI excluding zero (unadjusted; reported).

All other comparisons are exploratory:
- K3 vs K1, K4 vs K3 and K5 vs K1;
- K4 vs T1;
- L1–L5 against each other and against K1/K4;
- M1-RO.

**Descriptive.** Intake knock-to-knock variability (within-fruit SD of log f_dom over intake knocks) compared with the
intake-to-endpoint change (|log f_dom,t / f_dom,0| per endpoint recording).

## 5. Checks

- **Controls.**
  - With the baseline arm removed, the K1 training code uses the paper's CNN trunk, schedule and seed.
  - CDR-RO and M1-RO restricted to the 104 sessions are recomputed from the round-1 session probabilities.
  - The pairing sampler draws only intake knocks of the same fruit.
- **Leakage audit.** The audit asserts that:
  - no endpoint label of a test fruit, storage time, visit interval, end-of-storage weight or weight loss enters any feature,
    pairing or normalization;
  - intake knocks used as a baseline come only from intake sessions defined by the label-free rule.
- **Outputs.** All outputs go to `results/round4/`, produced by `src/anchor_physics/paired_cnn.py` and
  `src/anchor_physics/round4.py`.
