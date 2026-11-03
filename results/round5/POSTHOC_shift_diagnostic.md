# Round 5 post-hoc diagnostic: ranking signal vs decision-threshold shift (NOT pre-registered)

Computed 2026-11-03 after the round-5 evaluation, from `results/round5/probe_sessions.parquet`, the A1ft npy files and the round-1/4 references. LOBO, 104 endpoint sessions. Labels per batch (Ripe, Overripe): Immature (23, 11), Mature (21, 20), Overmature (12, 17). Everything here is hypothesis-generating; any rule derived from it must be pre-registered (round 6) or confirmed on a new harvest.

## 1. Within-batch AUC (Overripe vs Ripe) and oracle-threshold macro-F1

| Arm | AUC Imm | AUC Mat | AUC Ove | F1 Ove (argmax) | F1 Ove (oracle thr.) | mean p(Overripe) in Ove |
|---|---|---|---|---|---|---|
| CDR-RO (round-1 CNN) | 0.74 | 0.86 | **0.45** | 0.37 | 0.54 | 0.84 |
| K1 (round-4 CNN) | 0.66 | 0.89 | **0.39** | 0.37 | 0.53 | 0.82 |
| A1 BEATs frozen | 0.63 | 0.91 | **0.80** | 0.45 | 0.73 | 0.81 |
| A1ft BEATs LP-FT | 0.58 | 0.90 | 0.75 | 0.45 | 0.74 | 0.75 |
| A5 wav2vec2 | 0.72 | 0.93 | 0.79 | 0.65 | 0.76 | 0.67 |
| F1 fusion | 0.65 | 0.87 | **0.82** | 0.37 | 0.79 | 0.76 |
| F3 embedding fusion | 0.70 | 0.84 | **0.90** | 0.53 | 0.85 | 0.87 |
| V-ms DINO | 0.65 | 0.81 | 0.78 | 0.60 | 0.75 | 0.67 |
| V-rgb DINO | 0.71 | 0.66 | 0.59 | 0.51 | 0.58 | 0.84 |
| V-thermal DINO | 0.49 | 0.72 | 0.44 | 0.34 | 0.57 | 0.70 |

Reading: the task-specific CNNs carry no ranking information in the Overmature batch (AUC < 0.5); the foundation embeddings do (AUC 0.75–0.90), but their scores are shifted towards Overripe (mean p ≈ 0.8), so the 0.5 threshold assigns almost everything to Overripe. The Immature batch is the opposite case: AUC 0.6–0.75 for every arm, so even an oracle threshold gives only 0.62–0.75.

## 2. Label-free score-shift rules applied within the held-out batch (pooled macro-F1, Δ vs CDR-RO 0.661)

| Arm | argmax 0.5 | median-centering | train-prior quantile | prior-shift EM |
|---|---|---|---|---|
| A1 | 0.65 (−1.1) | 0.72 (+6.0 [−5.0, 17.0]) | 0.69 (+3.0) | 0.56 |
| A1ft | 0.73 (+7.0) | 0.68 (+2.2) | 0.65 (−0.9) | 0.66 |
| A5 | 0.74 (+7.9) | 0.74 (+8.0 [−2.7, 18.3]) | 0.73 (+6.9) | 0.48 |
| F1 | 0.73 (+7.0) | 0.70 (+4.1) | 0.67 (+1.1) | 0.57 |
| F3 | 0.69 (+3.2) | 0.72 (+6.0 [−6.1, 17.4]) | 0.71 (+4.9) | 0.65 |

Per batch, median-centering moves F3's Overmature F1 from 0.53 to 0.79 (Ripe recall 0.17 → 0.83) and F1's from 0.37 to 0.72, but costs 5–16 pp in the Immature batch, whose true prior (68 % Ripe) is far from 0.5. EM diverges to one class, as in round 3. No rule brings the pooled CI above zero at n = 104.
