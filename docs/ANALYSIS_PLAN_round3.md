# Analysis plan, round 3 (final): label-free Ripe/Overripe prior adaptation on the target batch's endpoint sessions

Status: **written and committed before any round-3 rule was computed.** The tag `prereg-round3` precedes every result file of this
round (`results/round3/`). Deviations will be listed in `results/round3/DEVIATIONS.md` and in the paper.

**This is the final round of method development on this dataset.** Whatever its result, the paper reports rounds 1, 2 and 3 in full,
together with their pre-registration and pre-evaluation tags:
- round 1: `prereg-anchor-physics`, `pre-evaluation-anchor-physics`;
- round 2: `prereg-round2`, `pre-evaluation-round2`;
- round 3: `prereg-round3`.

A positive result from this round would be exploratory and would require external data to confirm.

## 0. What is known before this plan

All scores are on the 106 LOBO endpoint sessions, macro-F1 over Ripe/Overripe.

**Round 1** (`results/v21/`):
- All three pre-registered primary contrasts were negative.
- Capture-day rule CDR-RO: 0.656.
- AdaBN on intake knocks (M5): 0.652.

**Round 2** (`results/round2/`). Candidates C1–C5; nested leave-one-batch-out selection.
- The procedure scored 0.679 against 0.656 for CDR-RO: +2.3 pp [−6.3, 11.7], Holm p = 0.936, not beneficial.
- The selections were C2 (Immature), C4 (Mature) and C4 (Overmature).
- Best single candidate: C4, 0.688.
- In the Overmature batch, Ripe recall was 0 for every rule except the two-class CNN C3 (0.231, Overripe recall 0.389).

The models thus rank a target batch's endpoint fruits but place the Ripe/Overripe boundary wrongly for that batch. The rules
below adapt only that boundary, without target labels. They were designed after seeing rounds 1–2 on the same sessions.

## 1. Data

As in rounds 1–2:
- 268 four-sensor sessions;
- the label-free intake rule: 162 anchors, 106 endpoint sessions;
- endpoint status known by protocol;
- common label.

No new model is trained. Every rule is computed from the round-2 session probabilities:
- `results/round2/cnn_sessions.parquet`;
- `inner_classical_sessions.parquet`;
- the round-1 M0 LOBO probabilities.

## 2. Candidates (fixed; no further candidates will be added)

**Bases**, as defined in round 2. Both are renormalized over Ripe and Overripe for each endpoint session.
- **C2 base**: the 3-class knock CNN with AdaBN on the target batch's intake knocks.
- **C4 base**: the mean of that CNN's and the peak + centroid + MFCC LR's renormalized Ripe/Overripe probabilities.

In every rule, anchors are predicted Unripe. Endpoints are assigned Ripe or Overripe as follows.

- **C2, C4** (carried over): the larger renormalized probability.
- **D1 = EM on C4.** Saerens et al. (2002) EM re-estimates the Ripe/Overripe prior on the target batch's endpoint sessions. The
  source prior is 0.5/0.5, because every base model was trained with class-balanced weights. Endpoints are then assigned by the
  adjusted argmax. Settings: 100 iterations, tolerance 1e-8, as in round 1.
- **D2 = quantile on C4.** Overripe is assigned to the top k = round(π_O × n) endpoint sessions of the target batch, ranked by
  log P(Overripe) − log P(Ripe). Here n is the number of target endpoint sessions, and π_O is the Overripe share among the
  endpoint sessions of the training batch(es); these are training labels. Ties are broken by session_id. Python's `round` is used
  (half to even).
- **D3 = EM on C2**, as D1.
- **D4 = quantile on C2**, as D2.

Reference rules:
- **CDR-RO** (round 1);
- **round-2 procedure**: the round-2 nested selection, i.e. C2/C4/C4.

## 3. Primary estimate: nested leave-one-batch-out selection

This is as in round 2, over the six candidates {C2, C4, D1, D2, D3, D4}.
- For each held-out batch, each candidate is scored on the two inner splits, using the existing single-batch fits. π_O comes from
  the inner training batch; EM and quantile use the inner test batch's endpoint sessions.
- The candidate with the highest pooled inner macro-F1 over Ripe/Overripe is selected. Ties go to the earlier candidate in the
  order C2, C4, D1, D2, D3, D4.
- The selected candidate is then applied to the held-out batch, with π_O from the two source batches.

The procedure's predictions on the three held-out batches are pooled.

## 4. Primary contrasts, decision rule and multiplicity

Holm adjustment across the two contrasts:
- **R3-1**: round-3 procedure vs CDR-RO;
- **R3-2**: round-3 procedure vs the round-2 procedure.

Statistics as in rounds 1–2:
- difference in pooled endpoint macro-F1 over Ripe/Overripe;
- fruit-cluster bootstrap 95% CI (2,000 resamples, seed 42);
- fruit-level permutation test (20,000, seed 42);
- clustered McNemar test for accuracy.

The procedure is called **beneficial** only if, against CDR-RO, all three hold:
- the gain is at least 3 pp;
- the bootstrap CI excludes zero;
- the Holm-adjusted p < 0.05.

## 5. Secondary (exploratory) and checks

**Secondary**, reported descriptively:
- per-candidate outer scores (pooled and per batch), with contrasts against CDR-RO and the round-2 procedure;
- class recall, including an Overmature Ripe-recall table;
- the inner score table;
- the target priors estimated by EM.

**Checks:**
- **Controls.**
  - C2 and C4 reproduce round 2 exactly.
  - The round-2 procedure's predictions are reproduced.
  - On synthetic data, D2/D4 assign exactly k Overripe sessions.
- **Leakage audit.** No target label enters a prior, threshold, ranking or selection. Inner selection never reads the outer batch.
