# Power analysis for the confirmatory new-harvest study

Source data: round-5 out-of-batch (LOBO) session predictions (`results/round5/probe_sessions.parquet`), the round-1 knock-CNN reference
(`results/v21/lobo_cnn_sessions.parquet`, variant M0, CDR-RO decision = argmax over Ripe/Overripe) and `results/session_table.csv`.
Population: the 104 gated endpoint sessions used for round-5 `scores.csv` (split lobo, pooled) — Immature 34, Mature 41,
Overmature 29; 103 fruits (one fruit contributes two sessions); labels Ripe 56 / Overripe 48.
Metric: macro-F1 over Ripe/Overripe. Uncertainty: fruit-cluster percentile bootstrap, identical to `src/anchor_physics/evaluate.py::paired`.

## 1. Harness validation (step 1)
Re-implementing `paired()` on the F1 fusion (equal-weight mean of the Ripe/Overripe probabilities of A1 BEATs and the three DINOv2 probes, as in
ANALYSIS_PLAN_round5 §F1) versus CDR-RO gives F1 = 0.731, CDR-RO = 0.661, Δ = **+7.01 pp, 95% CI [−2.59, 16.80]** (2,000 resamples, seed 42),
reproducing the round-5 Q2 contrast (+7.0 [−2.6, 16.8]) to two decimals. Half-width at n = 104 sessions: 9.7 pp; bootstrap SD 4.9 pp.

## 2. How the CI half-width scales with sample size (step 2)
Fruits were resampled stratified by batch (per-batch proportions preserved): fractions 0.4–1.0 without replacement, and 1.5×, 2×, 3×, 4× with
replacement (duplicated fruits treated as distinct clusters). For each of 60 resamples per size the paired bootstrap (1,000 resamples) half-width of
F1 − CDR-RO was recorded (`power_halfwidth_scaling.csv`).

| fruits | sessions | mean half-width (pp) |
|---|---|---|
| 41 | 41 | 15.5 |
| 51 | 52 | 13.8 |
| 62 | 63 | 12.4 |
| 73 | 74 | 11.6 |
| 82 | 83 | 10.7 |
| 93 | 94 | 10.1 |
| 103 | 104 | 9.6 |
| 155 | 157 | 7.8 |
| 206 | 208 | 6.8 |
| 309 | 312 | 5.5 |
| 412 | 416 | 4.8 |

Free-exponent fit: half-width = 100.4 · n_fruits^(−0.506); fixing the exponent at −1/2 gives **half-width ≈ 97.6 / √n_fruits**
(≈ 98.1 / √n_sessions). The observed exponent (−0.51) confirms that the current fruit-cluster bootstrap behaves as a pure 1/√n statistic:
it contains no between-batch variance component (only 3 batches; batch is not a resampling unit).

## 3. Sample size for 80 % power
Analytic model: the paired estimate d̂ ~ N(δ, σ(n)²) with σ(n) = half-width(n)/1.96 = 49.8/√n_fruits pp. The pre-registered rule
requires the 95 % CI to exclude 0 in the beneficial direction (lower bound > 0) and Δ ≥ 3 pp. Power = Φ(δ/σ(n) − 1.96); 80 % power requires
δ/σ(n) = 1.96 + 0.84 = **2.80 standard errors**, i.e. δ ≈ 1.43 half-widths (the "≈2.8" figure in the task refers to standard errors, not half-widths).
Required fruits: n = (97.6 · 2.80 / (1.96 · δ))². Sessions = fruits × 104/103 (≈1 endpoint session per fruit, as now).

| true gain δ | fruits (single primary contrast) | endpoint sessions | fruits if Holm over 2 primary contrasts (α = 0.025) | sessions |
|---|---|---|---|---|
| +5 pp | 779 | **786** | 943 | 952 |
| +7 pp | 397 | **401** | 481 | 486 |
| +10 pp | 195 | **197** | 236 | 238 |

The analytic power was checked by simulation (`power_analysis.csv`, columns `power_*_sim`): candidate prediction vectors with imposed effects of
+5.1, +7.0 and +9.9 pp were built from the observed F1/CDR-RO paired structure (flipping randomly chosen F1 predictions correct→wrong or
wrong→correct until the pooled Δ matched the target; 40 variants per effect), fruits were resampled with replacement within batch to 1×–8× the current
size (200 replicates per cell), and the rule (CI lower bound > 0 and Δ ≥ 3 pp) was applied with a 1,000-resample bootstrap. Simulated and analytic
power agree within ±0.09 at every grid point (largest gap 0.44 vs 0.53 at n = 104, +10 pp; e.g. +7 pp at 416 sessions: 0.81 simulated vs 0.81 analytic; +5 pp at 832 sessions: 0.85 vs 0.82).

### Six-batch design (2 harvest sources × 3 maturity stages, leave-one-source-out)
The fruit-cluster bootstrap does not see batch-level variance, so the numbers above are the *fruit-only* requirement and hold regardless of
how fruits are distributed over batches. Round-5 per-batch results show that the F1-vs-CDR-RO gain is concentrated in one batch (Immature +11 pp,
Mature 0, Overmature 0), so a real between-batch component exists that the current CI understates. **Assumption:** if the 6-batch crossed design
halves that between-batch variance (two sources per stage instead of one, and stage no longer confounded with source), the total standard error
shrinks by up to √2 and the required n by up to 2×. Because the share of batch variance cannot be estimated from 3 batches, we report a range:

| true gain δ | endpoint sessions, 6-batch design (range) |
|---|---|
| +5 pp | 393–786 |
| +7 pp | 201–401 |
| +10 pp | 98–197 |

The lower end assumes batch variance dominates and is halved; the upper end assumes fruit-level variance dominates (what the present bootstrap
assumes). **Plan to the upper end**; the protocol in `docs/PROTOCOL_new_harvest.md` uses the +7 pp target: ~400 endpoint sessions
(~67 per batch × 6 batches) plus 10 % attrition → 446 fruits enrolled.

## 4. Assumptions (all stated)
1. The new-harvest data have the same per-fruit discordance structure between candidate and reference as the round-5 F1/CDR-RO pair (same class
   balance Ripe 54 % / Overripe 46 %, same per-batch proportions, ~1 endpoint session per fruit).
2. The sampling distribution of the paired macro-F1 difference is approximately normal at n ≥ 100 (supported by the simulation agreement).
3. The reference arm in round 6 has the same variance as CDR-RO; a different reference (e.g. a change-only knock arm) changes the constant c.
4. The candidate arm is frozen before data collection (no selection on the new data), so no optimism correction is needed.
5. Leave-one-source-out splits: the training set in each fold contains all 3 stages from the other source, so the per-fold model variance is
   comparable to the present LOBO (2 of 3 batches). Fewer, larger folds would reduce it slightly; not modelled.
6. The √2 benefit of the 6-batch design is an assumption, not an estimate (see §3).
7. Power is for the CI/Δ ≥ 3 pp part of the rule; the permutation p (and Holm over m primary contrasts) is approximately equivalent to the CI
   criterion when m = 1, and requires the larger n shown for m = 2.
