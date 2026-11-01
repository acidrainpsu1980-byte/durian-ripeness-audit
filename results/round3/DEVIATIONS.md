# Deviations from docs/ANALYSIS_PLAN_round3.md (tag `prereg-round3`, commit 9a6d040)

Written before `src/anchor_physics/round3.py evaluate` was first run.

## Implementation choices the plan left open

1. **π_O.** π_O is the Overripe share among the endpoint sessions of the training batch(es) within the 268 four-sensor sessions.
   It is computed from training labels only. All 106 endpoint sessions are Ripe or Overripe, so the share is taken over Ripe +
   Overripe (asserted in the code).
2. **Ranking score.** The quantile rules rank by log(P(O) + 1e-12) − log(P(R) + 1e-12) on the renormalized probabilities. The
   constant only guards against log(0).
3. **Control on C2/C4.** The rule-level predictions are checked against `round2.candidates` from the same probabilities in every
   split, outer and inner. The round-2 procedure is reproduced from `results/round2/selection.csv`, and its pooled endpoint
   macro-F1 is asserted equal to the round-2 value.

## Checks run before evaluation

- `round3.py selftest`, run on synthetic data:
  - the quantile rule assigns exactly round(π × n) Overripe sessions for n ∈ {5, 31, 41} and π ∈ {0, 0.3, 0.46, 0.5, 1};
  - ties follow session_id order;
  - EM returns normalized probabilities.

## Tag correction

- A first push attempt created the tag `pre-evaluation-round3` on commit 9a6d040 (the plan commit), because the commit of the code
  failed. With the user's approval, the tag was moved to the commit that contains `round3.py` and this file, before evaluation.
