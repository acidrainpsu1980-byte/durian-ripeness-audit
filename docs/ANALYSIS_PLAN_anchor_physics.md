# Analysis plan: intake-anchored batch calibration and physics-guided acoustic features

Status: **written and committed before any anchored or physics-guided model was fitted or scored.** The git commit and tag
(`prereg-anchor-physics`) of this file precede every result file of this analysis (`results/v21/`). Deviations from this plan will be
listed in `results/v21/DEVIATIONS.md` and in the paper.

## 0. What is known before this plan, and what this plan cannot protect against

- The leave-one-harvest-batch-out (LOBO) results of the paper (Table 8) are known: every sensor and fusion falls to or below the
  majority-class rate on at least one held-out batch, with the predictions displaced along the ripeness axis (Immature batch: nearly
  everything predicted Unripe; Overmature batch: about half of the Unripe sessions predicted ripened, no Ripe recall).
- The methods below were designed after seeing that pattern. Pre-registration therefore protects against choosing among variants,
  metrics or subsets after seeing their results; it does not make the design independent of the data.
- Prior art (novelty memos, `docs/novelty/`): adaptation with target data of one known class exists (fault diagnosis with
  healthy-only target data, e-nose drift correction with one calibration class, negative-control normalization); resonance
  frequency, damping and the stiffness coefficient are established acoustic firmness descriptors, including for durian. The
  contribution tested here is the use of **protocol-supplied anchors** (harvest-day intake scans, Unripe by process) in postharvest
  grading, and whether physical acoustic parameters **transfer across harvest batches** better than a log-mel CNN.

## 1. Data, sessions and anchors

- Analysis set: the 268 four-sensor sessions (common label), the 315 sessions of the 164 analysis fruits for training captures, the
  432 verified sound recordings, exactly as in the paper.
- **Intake (anchor) session, label-free rule fixed here:** the earliest capture session of each fruit, provided it lies within 4 days
  of the fruit's recorded harvest date (`days_from_recorded_harvest <= 4`). Labels are never used to define anchors. Among the 268
  sessions this rule marks 162 anchors (159 Unripe, 3 Ripe Overmature-batch sessions of fruits without a harvest-day four-sensor
  session); the 3 are kept as realistic anchor contamination and reported.
- **Endpoint sessions:** all other sessions of the 268: 106 (57 Ripe, 49 Overripe; Immature 34, Mature 41, Overmature 31).
- Anchor statistics of a batch use the modality's captures of that batch's anchor sessions (all 315 sessions' captures).

## 2. Models (fixed; no further candidates will be added)

Base models are the development pipelines of the paper, trained on the common label with their own inner searches:
thermal SVM-RBF (67 features), RGB EfficientNet-B0 frozen embedding + LR probe, sound knock CNN (64 x 31 log-mel per knock; 20 epochs,
seed 42, CPU 3 threads), sound classical LR on the 265 recording features (secondary), and the thermal + sound + RGB mean rule.

**Physics-guided acoustic model (C).** Per knock, from the same 150 ms windows: dominant frequency (largest spectral peak, 100–2000
Hz), first prominent spectral peak (lowest peak with prominence >= 10 dB in 100–2000 Hz), spectral centroid, 20 dB decay time, and
damping as the half-power (−3 dB) bandwidth of the dominant peak divided by its frequency (1/Q). Per recording: the median over knocks
and the knock count. Mass-normalized stiffness: S = f_dom^2 * m^(2/3) with m = harvest weight (kg). **End-of-storage weight, weight
loss, storage days and any endpoint variable are never used.** Primary physics model: multinomial logistic regression (standardized
in-fold, class-balanced, C chosen by inner fruit-grouped 3-fold CV from {0.01, 0.1, 1, 10}) on
{f_dom, f_first, centroid, decay20, 1/Q, S, device}. Session probability = mean over the session's recordings.
Hybrid (C+CNN): mean of the physics-LR and knock-CNN session probabilities (no trained weights).
Physics baselines (secondary): harvest weight alone; f_dom alone; the peak-frequency + spectral-centroid + MFCC set (mean MFCC 1–13)
in the same LR.

## 3. Adaptation variants (A)

Held-out batch b; source = the two other batches.

- **M0 none**: the model trained on the source batches and applied unchanged (must reproduce the paper's LOBO predictions exactly for
  the shipped models; this is the control).
- **M1 anchored standardization (primary)**: every feature (embedding dimension for RGB; mel band for the CNN input) is standardized
  per batch with the mean and SD of that batch's anchor captures, for each source batch during training and for the held-out batch at
  test time. Classical/probe models are refitted on the standardized source data; the CNN is retrained with per-batch anchor mel
  normalization (per mel band) replacing its global normalization.
- **M2 unanchored standardization (baseline)**: as M1 but with the mean and SD of all captures of each batch (for the held-out batch:
  all of its captures, unlabeled).
- **M3 anchored logit shift (secondary)**: M0 model; target logits are shifted by the difference between the mean logit vector of the
  source anchor captures and that of the target anchor captures.
- **M4 prior-shift EM (baseline, secondary)**: Saerens et al. EM re-estimation of class priors on all target sessions, applied to M0.
- **CNN only, M5 AdaBN anchored (secondary)** / **M6 AdaBN unanchored (baseline)**: the M0 CNN with BatchNorm statistics re-estimated
  on the held-out batch's anchor knocks / all knocks.
- **Capture-day rule (trivial baseline)**: anchor sessions are labeled Unripe, endpoint sessions by the M0 model.

The physics model is evaluated with M0 and M1 (anchor standardization of the physics features, i.e. parameters relative to the new
batch's intake knocks) and M2.

## 4. Evaluation

- **Primary score:** LOBO, endpoint sessions only, pooled over the three held-out batches (106 sessions): macro-F1 over Ripe and
  Overripe (`labels=[Ripe, Overripe]`; an endpoint predicted Unripe counts as an error for its class), together with the false-Unripe
  rate. Per-batch values are reported alongside.
- **Secondary scores:** LOBO all 268 sessions, 3-class macro-F1, accuracy and class recall, with the anchors scored by the model and,
  separately, by the capture-day rule; majority rate of each held-out batch.
- **Within-batch check:** fruit-grouped 5-fold CV on the primary partition and the mean over the ten partitions (session macro-F1, all
  268 sessions) for the physics model and for M1 of the classical and probe models, and on the primary partition only for the knock
  CNN (ten CNN partitions would need GPU refits), to show whether anchoring or the physics features cost within-batch accuracy. Anchor statistics in CV use all anchor captures of each batch (label-free by protocol).
- **Uncertainty:** fruit-cluster bootstrap (2,000 resamples, seed 42) of paired differences on the pooled endpoint sessions; fruit-level
  permutation test (20,000) for macro-F1 differences; McNemar for clustered pairs for accuracy.

## 5. Primary contrasts, decision rule and multiplicity

Three primary contrasts (Holm across the three; everything else exploratory and descriptive):

- **PA1** sound knock CNN: M1 versus M0.
- **PA2** thermal + sound + RGB mean rule with every component M1 versus all components M0.
- **PC1** physics-guided LR with M1 versus the knock CNN with M0.

A method is called **beneficial** only if its pooled endpoint macro-F1 (Ripe/Overripe) exceeds the comparator by at least 5 pp, the
fruit-cluster bootstrap 95% CI of the difference excludes zero, the Holm-adjusted permutation p < 0.05, and its fruit-grouped
within-batch macro-F1 is not more than 2 pp below the comparator's (ten-partition mean; primary partition where a knock CNN is
involved). Results that do not meet all four conditions
are reported as not beneficial, whatever their direction. No variant, threshold or subset will be chosen after seeing the results.

## 6. Checks

- Controls: M0 reproduces the shipped LOBO predictions (max |diff| 0 for refitted classical/probe models; CNN refit on CPU with 3 threads).
- Leakage audit: a test asserts that no endpoint-session label, end-of-storage weight, weight loss or storage-time variable enters any
  feature, normalization statistic or model.
- Anchor sensitivity (exploratory): M1 with 10, 20, 40 % and all anchor fruits of the held-out batch (5 random draws each).
- All outputs go to `results/v21/`, produced by `src/anchor_physics/` and checked by `run_all.sh` against the shipped files.
