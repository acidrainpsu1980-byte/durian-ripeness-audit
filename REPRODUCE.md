# Reproducing the paper (audit package v2.3.1)

This package reproduces every table of *"Fruit-Grouped and Batch-Aware Evaluation of Thermal, Multispectral, Knocking-Sound and
RGB Sensing for Durian Ripeness Classification on a Public Longitudinal Dataset"* at three depths. Each depth regenerates its
outputs in a scratch directory and compares them with the shipped files; nothing is accepted on trust.

| Level | Starts from | Command | Wall time* | What is verified |
|---|---|---|---|---|
| A. Cached predictions | shipped out-of-fold predictions (`intermediate_*`) | `./run_all.sh` | 34 min | Every result table (`results/`, `results/v14`, `v15`, `v17`, `v19`), Supplementary Tables S1–S22, the audit tables, the prediction export and the 15 saved classical models; max \|diff\| ≤ 1e-9 required |
| B. Cached features | shipped feature tables and embeddings | `REFIT=1 ./run_all.sh` | 7 min more | The 20 classical candidates refitted on the primary partition (5 outer + 10 fold-pair fits each, 300 fits) reproduce the shipped predictions |
| C. Raw data | the four Zenodo archives (23.4 GB) | `DURIAN_DATA_ROOT=/path/to/zips RAW=1 CNN=1 ./run_all.sh` | 74 min (raw extraction 26 min, CNN retraining 48 min) more | Every feature table, the RGB cache and masks, both CNN embeddings and the log-mel arrays re-extracted from the archives; the five primary-partition knock CNNs retrained on CPU with weight export, reproducing the shipped predictions |

\*Measured for v1.8.0 in a fresh copy of the package on a 10-core Apple-silicon laptop (CPU only, `OMP_NUM_THREADS=4`), with all
three levels in one call (`audit/verification_log_v1.8.0.txt`).

**Which run checked which release.** Every run below was made in a fresh copy of the release tree on the same laptop and compared each
regenerated file with the shipped one (`ok` = max |diff| ≤ 1e-9). The largest difference was 3.3e-16 in the v1.8.0 run; in the
later logged runs every difference was 0.

| Release | Date | Command | What the run covered | Wall time | Exit | Evidence |
|---|---|---|---|---|---|---|
| v1.8.0 | 31 Oct 2026 | `REFIT=1 RAW=1 CNN=1 ./run_all.sh` | levels A, B and C | 1 h 54 min | 0 | `audit/verification_log_v1.8.0.txt` |
| v1.9.0 | 1 Nov 2026 | `./run_all.sh` | level A | 34 min | 0 | `audit/verification_log_v1.9.0.txt` |
| v1.9.1 | 1 Nov 2026 | `./run_all.sh` | level A | 36 min | 0 | `audit/verification_log_v1.9.1.txt` |
| v2.0.0 | 2 Nov 2026 | `LABELSENS=1 ANCHOR=1 ./run_all.sh` | level A, label-policy retraining (re-evaluation), rounds 1–3 | 47 min | 0 | `audit/verification_log_v2.0.0.txt` |
| v2.1.0 | — | — | never published (`results/round6/DEVIATIONS.md`, item 7) | — | — | none |
| v2.2.0 | 8 Nov 2026 | `LABELSENS=1 ANCHOR=1 ./run_all.sh` | level A, label-policy retraining (re-evaluation), rounds 1–6 | 54 min | 0 | the v2.2.0 tag and its GitHub release notes only; no log file was kept in the package |
| v2.3.0 | 10 Nov 2026 | `LABELSENS=1 ANCHOR=1 ./run_all.sh` | level A, label-policy retraining (re-evaluation), rounds 1–7, oracle-threshold bound, Figure 1c statistics, S1–S29 | about 55 min, in two parts (see log) | 0 | `audit/verification_log_v2.3.0.txt` (commit b074759) |

v2.3.1 changes only documentation and logs (this section, the log file names, version numbers); its code and results are those of v2.3.0.
Levels B and C were last run for v1.8.0; no later release repeated them.

**What the v2.3.0 run did, by kind of check.**

1. *Regenerated from cached predictions.* The fusion pipeline, including its stacking meta-learners, and the sensitivity tables
   (stage 1), nested selection over the ten partitions (stage 3), the development pipelines and matched candidate space (stage 5),
   Supplementary Tables S1–S29 and the Figure 1c statistics from the deposited characteristics file (stage 6), and the audit tables and
   out-of-fold prediction export (stage 7), all from the shipped out-of-fold predictions in `intermediate_*`.
2. *Re-evaluated from shipped outputs (no model refitted).* The label-policy retraining (`LABELSENS=1`: shipped retrained predictions in
   `results/v20/`; `LABELSENS_REFIT=1` was not set) and, under `ANCHOR=1`, every knock-CNN arm of rounds 1, 2 and 4 (shipped CNN
   predictions), the probe and fine-tune arms of rounds 5–6 (shipped predictions; batch decodability shipped, not recomputed), round 7
   (shipped `fit7` output `results/round7/rta_sessions.parquet`), the post hoc analyses of rounds 5–7 including the oracle-threshold
   bound, the power and harvest scenarios, and the 15 saved classical models (stage 7, `export_models.py check`).
3. *Refitted from cached features.* The targeted ablations (stage 4: thermal, multispectral-wavelength and RGB ablations, from the
   shipped feature tables and image embeddings), the four LightGBM comparison models on the common label (`[v2.2]` stage), and, under
   `ANCHOR=1`, the round-1 classical and physics-guided models, baselines, within-batch checks and anchor sensitivity and the classical
   candidates of rounds 2 and 4 (including L1–L5 and T1), from the shipped feature tables and `results/v21/physics_*.parquet`. The 300
   classical candidate refits of level B (`REFIT=1`) were not part of this run.
4. *Computed from raw data.* None. No feature table, embedding or log-mel array was re-extracted from the Zenodo archives, and no knock CNN,
   foundation-encoder probe or `fit7` model was retrained; level C (`RAW=1 CNN=1`) was last run for v1.8.0, and the GPU and
   foundation-embedding fits (`ANCHOR_CNN=1`, `src/foundation/extract_r5.py`, `round7.py fit7`) have not been rerun from raw data in a
   verification run.

## 1. Environment

```bash
conda env create -f environment.yml && conda activate durian      # Python 3.12, scikit-learn 1.9.1, LightGBM 4.7.0, PyTorch 2.14.1 (CPU)
```

- Exact versions of every installed package: `audit/requirements_lock_cpu.txt` (97 packages); key versions and the GPU stack in
  `audit/SOFTWARE.md`.
- `ffmpeg` decodes the iPhone `.m4a` recordings. conda-forge `soundfile` does not bundle `libsndfile`, and on macOS the
  `DYLD_*` search variables are removed when scripts start through `/usr/bin/env`; `src/sound/knock_features.py` therefore points
  the library lookup at `$CONDA_PREFIX/lib` itself, so no environment variable is needed.
- **Threads.** Keep `OMP_NUM_THREADS` ≥ 4 (the default in `run_all.sh` and the `Makefile`). With one BLAS thread, three
  logistic-regression fits solved by L-BFGS (multispectral LR, MobileNetV2 and EfficientNet-B0 probes) change their probabilities by up
  to 0.016 because the floating-point summation order changes; with 4 or 10 threads they are bit-identical to the shipped fits. The
  knock CNN on CPU is bit-reproducible with `CNN_THREADS=3` (the setting used for the paper).
- The knock-CNN refits on the ten partitions (Table 3, Table S13) were run on a GPU (NVIDIA GeForce RTX 3090 Ti, driver 595.99.02,
  PyTorch 2.14.1+cu130, cuDNN 9.24, deterministic cuDNN); they are shipped as predictions in `intermediate_v14/cnn/` and
  `intermediate_v15/dev/`. Re-running them (`src/nested_selection/cnn_part.py`, `src/ablations/cnn_dev.py`) needs a CUDA GPU, and a
  different GPU or driver may change the last digits; the primary-partition CNN (the "primary" columns) is CPU-reproducible bit for bit.

## 2. Level A — tables from cached predictions (no raw data)

```bash
./run_all.sh          # or: make reproduce
```

| Stage | Script(s) | Regenerates | Paper items |
|---|---|---|---|
| 1 | `src/fusion_pipeline.py` (config `src/fusion_config.json`), `src/verify_reproduction.py`, `src/sensitivity/sensitivity_tables.py` | `results/*.csv` | Tables 5–11; S1–S6, S9, S11 |
| 2 | `src/fusion_figures.py` | `figures/diagnostic/` | diagnostic plots of the data behind Figures 3-5 (the leakage plot uses the earlier table `results/leakage_table.csv`); the paper figures are shipped in `figures/paper/` (see `figures/README.md`) |
| 3 | `src/nested_selection/evaluate_v14.py` (`nsel.py`, `stats4.py`) | `results/v14/` | Tables 3 (selection procedures, ten partitions), 4; S12–S14 |
| 4 | `src/ablations/evaluate_v15.py`, `src/ablations/ms_wavelength_ablation.py` | `results/v15/` | Table 12; S15–S17, S19 |
| 5 | `src/nested_selection/matched_space.py`; `src/nested_selection/primary_metric.py` | `results/v17/` (incl. `matched_space_session_oof.parquet`); `results/v19/` | Tables 3 and 4 (fold-mean, pooled and across-partition macro-F1 / accuracy, class recall, paired comparisons); S20–S22 |
| 6 | `src/audit/build_supplementary_tables.py` | `audit/supplementary/` | Supplementary Tables S1–S24 as files |
| 7 | `src/audit/build_audit_tables.py`, `model_registry.py`, `export_predictions.py`, `export_models.py check` | `audit/` | exclusion ledger, sessions, partitions, registry, predictions, saved models |

One cached input is a fixed earlier computation rather than a stage output: the v1.3 nested-stacking probabilities
(`intermediate_common/nested/`, used by stage 1 for the stacking rows of Tables 6–11). Stage 3 recomputes nested stacking for the
primary partition with the v1.4 code (`nsel.py`); the two agree to within 0.0065 in probability and give the same predicted class
for all 268 sessions in all 11 subsets, so every reported metric is identical.

Further checks shipped with their output: `src/audit/table1.py` (Table 1 from `audit/fruits.csv`, `audit/sessions.csv` and the
feature tables) and `src/audit/check_manuscript_numbers.py MANUSCRIPT.md SUPPLEMENT.md`, which confirms that every number in the results tables of the
v2.2 manuscript (main Tables 3–7; Supplementary Tables A1, B1–B4, C1, C2, D1–D3, D5; 1,320 numbers) occurs in the result files mapped to
that table (`audit/manuscript_number_check.csv`; a presence check per table, not a cell-by-cell match). `audit/table_map.csv` lists, for every table and figure of the paper, the source file, script and stage.

## 3. Level B — predictions from cached features

```bash
REFIT=1 ./run_all.sh                        # 20 classical candidates x (5 outer + 10 fold-pair) fits on the primary partition
# all ten partitions (as in the paper; ~1 h on 4 cores):
for P in p0 p1 p2 p3 p4 p5 p6 p7 p8 p9; do
  python src/nested_selection/run_fits.py results_repro/fits $P $(python -c "import sys; sys.path.insert(0,'src/nested_selection'); import cands; print(' '.join(f'{m}:{c}' for m,l in cands.CANDIDATES.items() for c in l if c!='cnn'))")
  python src/audit/compare_fits.py intermediate_v14/fits/$P results_repro/fits/$P
done
python src/audit/search_log.py              # every grid point's inner-CV scores on p0 -> audit/hyperparameter_search_p0.csv
```

The candidate families, fixed settings, search grids, inner CV and seeds are defined in `src/nested_selection/cands.py` and listed
in `audit/model_registry.csv` (generated from the estimator objects by `src/audit/model_registry.py`); the configuration space of the
selection procedures (C26, C37) is in `audit/configuration_space.csv`.

## 4. Level C — everything from the raw Zenodo archives

```bash
export DURIAN_DATA_ROOT=/path/to/zips      # dataset_clean_{thermal,multispectral,sound,rgb}.zip from doi:10.5281/zenodo.18603796
md5sum -c data/zenodo_checksums.md5        # macOS: md5 -r <file>, compare with the table in README.md
RAW=1 CNN=1 ./run_all.sh
```

`RAW=1` runs `src/audit/verify_raw_features.py`, which executes in `work/raw/<modality>/`:

| Modality | Commands | Outputs compared with |
|---|---|---|
| Thermal | `src/thermal/run_extract.py` | `intermediate_common/features/features_thermal.parquet` (70 feature columns, 1,671 frames) |
| Multispectral | `src/multispectral/extract_all.py` | `features_multispectral.parquet` (136 columns, 2,222 TIFFs) |
| Sound | `src/sound/sound_pipeline.py features`, `src/common_label/prepare_sound_cnn_inputs.py` | `features_sound_recordings.parquet` (270 columns, 432 recordings), `intermediate_v14/features_sound_knocks.parquet` (15,709 knocks); log-mels and CNN label files |
| RGB | `src/rgb/build_cache.py`, `src/rgb/extract_features.py`, `src/rgb/rgb_models.py embed` | `features_rgb.parquet` (100 columns, 1,999 photographs), `intermediate_common/rgb/emb_{mobilenetv2,efficientnetb0}.npy` |

The shipped feature tables carry the v1.3 common session label, the re-extraction the file-name label; the number of rows whose
label differs is reported separately and equals the documented relabelling (22 multispectral captures, 18 RGB photographs,
16 sound recordings). `CNN=1` then trains the five primary-partition knock CNNs (`src/common_label/cnn_common.py`, 20 epochs, seed 42,
3 threads), writes their weights to `models/primary_p0/`, and checks both the predictions and the reloaded weights.

The per-modality model comparisons that preceded the common-label refits (the `make thermal|multispectral|sound|rgb` targets,
file-label OOF in `intermediate/`) are kept for provenance; the literal interactive cells are in `src/session_traces/`.

## 5. Optional stages (v2.0–v2.3)

```bash
LABELSENS=1 ./run_all.sh        # label-policy retraining (results/v20 -> results_repro/v20); LABELSENS_REFIT=1 also refits thermal/RGB
ANCHOR=1 ./run_all.sh           # the seven adaptation rounds (results/v21, round2-round7), post hoc diagnostics, docs/protocol (~32 min)
ANCHOR=1 ANCHOR_CNN=1 ./run_all.sh   # also retrains the 17 round-1 knock CNNs on CPU (~2.5 h on 9 cores; needs work/raw/sound)
```

Each stage regenerates its outputs from the shipped knock-CNN predictions (`results/v20/cnn`, `results/v21/cnn`,
`results/round2/cnn_sessions.parquet`) and compares every CSV/parquet with the shipped file. The round-2 knock CNNs (6 three-class and 6
two-class single-batch fits, 3 two-class LOBO fits) are documented in `docs/ANALYSIS_PLAN_round2.md` and `src/anchor_physics/round2_cnn.py`;
their weights are shipped in `results/round2/cnn/*_weights/`. Plans, deviations and the tags that fix them before evaluation are listed in
`README.md`. After the pre-evaluation tags, only output-path overrides (`ROUND_V21`, `ROUND2_DIR`, `ROUND3_DIR`, `ANCHOR_OUT`) were added to
the round scripts; the defaults reproduce the tagged behaviour.

**v2.2 additions.** Every run also refits the four LightGBM comparison models on the reconstructed (common) label
(`src/common_label/lgbm_common_leakage.py` -> `results/common_label/`, ~1 min; Table B3, Figure 4). The ten-partition summaries
(Tables 3, 4, S13, S14, S20–S22, D3) use one manifest: the CPU knock CNN on partition 0 (`intermediate_v14/cnn_cpu_p0/`) and the GPU refits
on partitions 1–9 (`intermediate_v14/cnn/`); the GPU refit of partition 0 is kept as a hardware check only. The ANCHOR stage additionally
regenerates rounds 5–6 from the shipped probe predictions (the 1.7 GB round-5 embeddings are not shipped; `src/foundation/extract_r5.py`
and `docs/round5/emb_md5_list.txt`), the post hoc diagnostics (`posthoc_shift.py`, `posthoc_auc_ci.py`, `per_batch_ci.py`) and the
planning files (`src/protocol/power_analysis.py`, `harvest_scenarios.py`).

## 6. Where each reviewer-requested item is

See `audit/README.md`.

**v2.3 additions.** The ANCHOR stage also evaluates round 7 (`src/foundation/round7.py evaluate7`) from the shipped `fit7` outputs
(`results/round7/rta_sessions.parquet`; `fit7` needs the round-5 embeddings and ran on the GPU host, CPU only), recomputes the post hoc
oracle-threshold bound (`src/foundation/posthoc_oracle_headroom.py`) and rebuilds Supplementary Tables S24–S29 from the regenerated rounds.
Stage 6 builds S1–S29 and recomputes the Figure 1c statistics (`src/audit/fig1c_reference.py` -> `results/fig1c/`, which also redraws panel c).
