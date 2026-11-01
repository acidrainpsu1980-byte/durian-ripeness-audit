# Deviations from docs/ANALYSIS_PLAN_round2.md (tag `prereg-round2`, commit 385e454)

Written before `src/anchor_physics/round2.py evaluate` was first run, unless marked otherwise.

## Implementation choices the plan left open

1. **Training data of the single-batch (inner) fits.** All captures of the training batch (sessions of the 164 analysis fruits),
   as in the round-1 LOBO fits. Classical families, grids and inner fruit-grouped 3-fold CV are those of round 1.
2. **C3 training knocks.** All verified source knocks whose common label is Ripe or Overripe. This includes the knocks of the
   3 intake sessions labeled Ripe in the Overmature batch, because labels, not protocol status, define the training set of a
   two-class model. Class-balanced weights are computed over the two classes.
3. **C3 AdaBN reference.** Verified knocks of the test batch's non-intake recordings (all sessions of the analysis fruits). No
   labels are used.
4. **C4.** The peak + centroid + MFCC LR outputs three class probabilities. Its Ripe/Overripe probabilities are renormalized to
   sum to 1 before averaging with the renormalized C2 CNN probabilities.
5. **Selection ties** were to go to the lower candidate number, as pre-registered; whether any occurred is reported in
   `selection.csv`.

## Information seen before evaluation

6. The CNN training logs print knock-level accuracy on the test batch, as in round 1. During round 2 only the number of completed
   fits was read from the logs, not the accuracy values.

## Controls

- Default 3-class mode of `cnn_common.py` after the round-2 changes, refit of `lobo:Overmature`, vs round-1 M0: max |diff| 0, identical NaN pattern.
- C1 outer (AdaBN on intake knocks of the round-1 M0 networks) vs round-1 M5 session probabilities: max |diff| 0.
- AdaBN reference sets partition each test batch's verified knocks: anchor + endpoint knocks = 2680 + 1474, 1963 + 2392 and 2957 + 1112
  (Immature, Mature, Overmature), equal to the batch totals of round 1.
- Gated round-1 M0 vs round-1 CDR-RO endpoint macro-F1: asserted in `round2.py evaluate`
