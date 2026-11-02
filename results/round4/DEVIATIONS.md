# Deviations from docs/ANALYSIS_PLAN_round4.md (tag `prereg-round4`, commit dc42c77)

Written before `src/anchor_physics/round4.py evaluate` was first run.

## Implementation choices the plan left open

1. **Training and test recordings of L1–L5 and T1.** Training uses the verified recordings of non-intake sessions labeled Ripe or
   Overripe whose fruit has a baseline (105 endpoint sessions with sound, of which 104 are scored). Every split predicts all non-intake
   sessions of the test fruits; scoring is restricted to the 104 pre-registered sessions.
2. **T1 elapsed time** = endpoint session date − the date of the fruit's intake session (label-free intake rule), in days.
3. **K6 donors** are drawn within (harvest batch, primary fruit fold) groups. Every fruit with a baseline received a donor other than
   itself (asserted).
4. **Thread count.** The feature-level fits use `OMP_NUM_THREADS=4`; a run with 1 thread gave identical session probabilities
   (max |diff| 0). The knock CNNs use 3 threads, as in the paper.

## Information seen before evaluation

5. `paired_cnn.py` prints only training-set sizes and run time per fit. No accuracy or score was printed or inspected before
   evaluation.

## Controls

- CDR-RO on the 104 sessions is recomputed from the round-1 M0 session probabilities (asserted in `round4.py evaluate`).
- The pairing sampler draws only intake knocks of the same fruit (K6: of its donor); this is asserted in every mini-batch.
