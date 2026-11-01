# Changelog

## v1.9.1 — comparison roles relabelled (third review round, comment 3)

- H1–H3 are labelled `primary_retrospective` instead of `confirmatory` in `results/v19/comparisons_macro_f1.csv`, Table S22 and
  Table S14: they were designated during revision after the data had been examined, so they are retrospectively designated primary
  comparisons, interpreted as exploratory; Holm's adjustment controls multiplicity within H1–H3 only. Values are unchanged.
- Table S14: the columns `d_acc_pp_primary` / `d_f1_pp_primary` are renamed `..._primary_foldmean` (they are fold-mean differences
  on the primary partition; values unchanged). `results/v14/comparisons_v14.csv` is left as produced by v1.4.
- Comments in `cands.py`, `nsel.py` and `matched_space.py` no longer call the candidate list or the two fixed-configuration fusions
  "pre-specified": the candidates, their preprocessing and the two fusion configurations were developed on this dataset, so the
  nested estimates are conditional on them.

## v1.9.0 — primary metric aligned with the estimates and comparisons (third review round, comment 2)

- **Macro-F1 as the primary metric throughout Tables 3 and 4.** `src/nested_selection/matched_space.py` now also writes the
  session-level outer-fold probabilities of every estimate (`results/v17/matched_space_session_oof.parquet`; its other outputs are
  unchanged). The new stage `src/nested_selection/primary_metric.py` (`results/v19/`) summarizes every development pipeline and
  selection procedure as fold-mean, pooled and across-partition (mean, SD, range) macro-F1 and accuracy with pooled class recall,
  and tests the paired comparisons in macro-F1 and accuracy: pooled primary-partition differences with a fruit-cluster bootstrap
  and a fruit-level permutation test (macro-F1), fruit-clustered McNemar (accuracy), the corrected resampled test over 10 x 5
  folds, partition-level ranges and Holm across H1-H3. The bootstrap and corrected resampled values reproduce
  `results/v14/comparisons_v14.csv`.
- **Supplementary Tables S21 and S22** (`audit/supplementary/`); `audit/table_map.csv` updated.
- **Number check.** `src/audit/check_manuscript_numbers.py` now traces the one- and two-decimal numbers of Tables 3 and 4 as well.
- `CITATION.cff`: the placeholder DOI is removed (the package is cited by its GitHub release).

## v1.8.0 — reproducible audit package (second review round)

Everything the paper depends on is now a versioned file regenerated and checked by `run_all.sh` (see `REPRODUCE.md`, `audit/README.md`).

- **Three verified reproduction levels.** A: tables from cached predictions (`./run_all.sh`); B: predictions from cached features
  (`REFIT=1`); C: features, embeddings and the primary-partition knock CNNs from the raw Zenodo archives (`RAW=1 CNN=1`). Every
  stage regenerates into `results_repro/` or `work/` and is compared with the shipped files (`src/audit/compare_tables.py`,
  `compare_fits.py`).
- **Raw → feature path executed.** `src/audit/verify_raw_features.py` re-runs the per-modality extraction drivers on the archives
  (`audit/raw_verification.csv`). v1.0–v1.7 described these drivers as syntax-checked only.
- **Exclusions with evidence.** `data/raw_file_inventory.csv` (every archive member, size, CRC-32, role; `src/audit/inventory_raw.py`)
  and `audit/exclusion_ledger.csv` (archive members, fruits, sessions and recordings, each with reason and evidence; files named
  TAGAKn are matched to the fruit-log field tags and shown not to change the analysis set).
- **Sessions, labels, partitions.** `audit/sessions.csv`, `audit/fruits.csv`, `audit/fruit_partitions.csv`; Table 1 recomputed
  by `src/audit/table1.py`.
- **Models.** `audit/model_registry.csv` and `audit/configuration_space.csv` generated from the estimator objects;
  `audit/hyperparameter_search_p0.csv` (every grid point of every inner search on p0); fitted models of the four development pipelines
  in `models/primary_p0/` (CNN weights exported through the new optional `CNN_SAVE_WEIGHTS` hook in `cnn_common.py` and
  `cnn_part.py`, which does not touch the random-number stream); `audit/SOFTWARE.md`, `audit/requirements_lock_cpu.txt`.
- **Predictions.** `audit/predictions/` exports every capture- and session-level out-of-fold prediction of every candidate,
  configuration and selection procedure in all partitions, plus every per-fold choice.
- **Tables.** Supplementary Tables S1–S20 as files (`audit/supplementary/`; S1–S5 assembled for the first time);
  `src/sensitivity/sensitivity_tables.py` turns the sound-matching (Table 10, S9) and label-policy (Table 11, S11) analyses and the S6
  summary, previously interactive cells, into a stage; `audit/table_map.csv`; `src/audit/check_manuscript_numbers.py`.
- **Fixes found by the audit.** `run_all.sh` stage 5 compared every file in `results/v15/` including the multispectral ablation that
  it did not produce (passed only because an earlier output was on disk); `environment.yml` described a pip venv although the
  environment is conda-forge; `soundfile` could not find conda's libsndfile when the pipeline was started from `run_all.sh` on macOS
  (System Integrity Protection strips `DYLD_*`), so `src/sound/knock_features.py` now locates it in `$CONDA_PREFIX/lib`; the threaded knock
  extraction could crash intermittently (SIGSEGV) because `libsoxr` (OpenMP) was called from several threads at once, so resampling is
  now serialised with a lock (decoding stays parallel; features unchanged, verified bit for bit); the
  MobileNetV2 embeddings were outside the bundle (now `intermediate_common/rgb/emb_mobilenetv2.npy`); a byte-identical duplicate of the
  EfficientNet-B0 embeddings in `intermediate/rgb/` was removed. Manuscript corrections: 105 (not 104) of the 106 excluded Android
  files are byte-identical to the fruit's other-session file (104 counted duplicate groups); Table 1 thermal quantization now reports
  the exact 1st–99th percentiles of distinct values per frame instead of rounded ranges: Immature 9–33 (5–13 Sep) and 45–124
  (15–17 Sep), Mature 39–168, Overmature 47–239; pooled over all sessions after 13 Sep 40–236 (Section 3.6 text).
- Threads: keep `OMP_NUM_THREADS` >= 4; with one thread three L-BFGS logistic fits drift by up to 0.016 in probability.

## v1.7 — development pipelines versus selection procedures (second review round, comment 1)

- `src/nested_selection/matched_space.py`: on the same 26 candidates as the nested configuration search, the post hoc best
  configuration (sound + RGB mean; 0.908 primary, 0.893 held fixed over ten partitions), the nested configuration choice with fixed
  families (0.897; 0.881) and with selected families (0.897; 0.874); procedure versions of the pre-specified fusions (families selected
  per fold: T+S+R mean 0.888, four-sensor stacking 0.894 over ten partitions) and procedure-level comparisons with sound (fruit-clustered
  McNemar, corrected resampled t). Outputs in `results/v17/`.
- Manuscript: the nested, repeated-partition analysis is now Section 3.2 and Table 3, ahead of the development-pipeline results.

## v1.6 — multispectral band-mapping ablation (response to review, comment 8)

- `src/ablations/ms_wavelength_ablation.py`: the selected multispectral LR refitted without the three wavelength-dependent features,
  with alternative band centres, under 20 random tile orders (all order-dependent features recomputed from stored per-band values;
  published slope/curvature reproduced exactly) and with order-free PLS-DA on raw band means. Primary partition 0.740 (published),
  0.752 without slope/curvature, 0.732–0.762 under random orders; ten partitions 0.750 vs 0.753. Output:
  `results/v15/ms_wavelength_ablation_detail.csv`.

## v1.5 — targeted ablations of acquisition pathways (response to review, comment 6)

- **Thermal quantization** (`src/ablations/thermal_harmonize.py`): every radiometric frame re-quantized to a 0.5 °C grid, features
  recomputed with the original masks (segmentation columns identical); fruit-relative subset of 52 features (15 absolute
  temperature statistics removed). Quantization-only 0.803 → 0.589; thermal 0.840 → 0.818 (SVM) / 0.848 (LightGBM); LOBO still below
  majority on Immature and Overmature.
- **RGB masking** (`src/ablations/rgb_mask_embed.py`): EfficientNet-B0 re-embedding of fruit-mask, stalk-free fruit-body (41-px
  opening) and context-only crops; unmodified crops reproduce the stored embeddings (max diff 6e-7). Probe 0.827 → 0.784 / 0.761;
  context alone 0.736. Fruit-disjoint cross-camera transfer 0.55–0.68 for every variant.
- **Phones** (`src/ablations/cnn_dev.py`, GPU, 150 fits): iPhone-only, Android-only and iPhone-on-dual-session CNNs for ten partitions,
  scored with the dual-phone CNN on the 163 identical dual-phone sessions: dual 0.895, iPhone-only 0.902, Android-only 0.871
  (ten-partition means); iPhone → Android −7.9 pp.
- **Fusion with ablated inputs:** thermal + sound + RGB mean 0.916 / 0.862 (harmonized fruit-relative thermal + stalk-free RGB).
- `src/ablations/evaluate_v15.py` regenerates `results/v15/` from `intermediate_v15/` and the v1.4 intermediates (`run_all.sh` stage 5).

## v1.4 — nested model selection and dependence-aware inference (response to review, comment 4)

- **Nested model-family selection:** 21 candidate families, fixed a priori from the track scripts (`src/nested_selection/cands.py`;
  thermal CNN and RGB fine-tunes excluded), were refitted on every outer fold and every fold pair (`run_fits.py`). For each outer
  fold, each sensor's family and, separately, the best single sensor or mean/product fusion (26 configurations) were chosen from
  inner predictions of the four training folds only (`nsel.py`).
- **Ten fruit partitions:** the primary split plus nine more (same stratification, seeds 1–9; `partitions.csv`). The knock CNN was
  refitted 150 times on a GPU (`cnn_part.py`); on the primary partition GPU and CPU fits agree for 99.3% of sessions.
- **Tests (`stats4.py`):** fruit-clustered McNemar (Durkalski 2003), fruit sign-flip permutation, fruit bootstrap (conditional
  on fitted predictions), and Nadeau–Bengio corrected resampled t over 10 × 5 folds. A confirmatory set (H1–H3) is
  Holm-adjusted; everything else is exploratory.
- **Results** (`results/v14/`):
  - Sound: knock CNN chosen in 5/5 primary folds (0.899) and 36/50 overall; sound is the best single sensor in 50/50 folds.
  - Thermal and RGB: nested selection lowers them by 3.8 and 5.3 pp (post hoc selection optimism).
  - Across partitions: sound 0.885 [0.870–0.899]; T+S+R mean 0.898; 4-sensor stacking 0.900.
  - No confirmatory comparison is significant (Holm p = 1.0).
- **Correction:** the primary split is stratified by maturity × actual endpoint (not × size class, as previously written).
- `run_all.sh` stage 4 reproduces `results/v14/` exactly from `intermediate_v14/`.

## v1.3 — common session label (response to review, comment 3)

- **Found:** the README does not define the Unripe/Ripe/Overripe classes. In the fruit log, 45 of the 49 Ripe sessions
  with a logged ripening day fall exactly on that day (47 within 1 day), and 27 of 49 Overripe sessions fall on the logged
  capsule-splitting day (40 within 1 day). Unripe captures lie 0–4 days after the recorded harvest date (not all on it);
  13 Overmature Unripe sessions are dated 5 days before their recorded harvest date. The file-name ripeness code differs
  between modalities or between the two phones' originals in 12 sessions (9 in the 268-session analysis set).
- **New label policy (model-independent):** (i) unanimous file label kept; (ii) otherwise adjudicated from the fruit log
  with ±1-day tolerance (Overripe if status Overripe and day ≥ splitting − 1; Ripe if status Ripe and day ≥ ripening − 1);
  (iii) otherwise unresolved (none). Labels in `data/session_labels_common.csv` (Table S10). Changes vs v1.2: 2 session
  labels (146 U→O, 37 R→O); 18/1671 thermal, 22/2222 multispectral, 18/1999 RGB captures and 16/432 sound recordings.
- **Retrained on the common label:** the selected model of every modality (thermal SVM-RBF, multispectral LR, RGB probe,
  sound CNN), all nested-stacking refits, random-split arms and LOBO fits (`src/common_label/`). Comparison models
  (LightGBM rows), cross-device and cross-camera experiments keep the v1.2 file-label training and are scored against the
  common label.
- **Sensitivity:** principal comparisons repeated after excluding the 9 disputed sessions, the 7 log-inconsistent sessions
  and the 15 harvest-sign sessions (`results/label_policy_sensitivity.csv`, `results/label_sets.csv`; Table S11).
- **Effect (session accuracy / macro-F1):**
  - Sound: 0.889/0.810 → 0.899/0.828.
  - T+S+R mean rule: 0.899/0.832 → 0.909/0.850.
  - Nested 4-sensor stacking: 0.913/0.854 → 0.909/0.850.
  - Post hoc best: sound + RGB stacking 0.924/0.876.
  - Minimum McNemar p for a fusion gain over sound: 0.19 → 0.12.
  - Thermal (p = 0.005) and RGB (p = 0.008) are now also significantly below sound.
  - Majority rate 0.597 → 0.593.
  - Unchanged: the LOBO collapse and image-sensor leakage.
- The v1.2 tables are kept in `results_v1_2_file_labels/`, and the v1.2 config in `src/fusion_config_v1_2_file_labels.json`.

## v1.2 — sound recording provenance (response to review, comment 2)

- **Found:** the deposited sound metadata lists every iPhone × Android combination of a fruit's original recordings, so
  standardized Android file names were overwritten during release. 106 of the 269 Android recordings in the analysis set
  contain the fruit's OTHER session (opposite label); 104 of them are byte-identical (MD5) to that session's file.
  iPhone recordings are unaffected.
- **Verification:** each shipped file was matched to its original by byte size, recovering the original name and timestamp
  (iPhone timestamps +8 h). The recovered dates were compared with the fruit's thermal/multispectral/RGB capture dates: all 538
  fall exactly on an image capture date. See `data/sound_recording_manifest.csv` (Table S7, 562 files) and
  `data/sound_session_evidence.csv` (Table S8).
- **Fix:** the 106 recordings in `data/sound_exclude_wrong_session.txt` are excluded via `DURIAN_SOUND_EXCLUDE`, leaving 432.
  Retrained: the sound CNN (5 folds, 10 nested fold pairs, random split, cross-device, LOBO) and the classical sound models.
  The fusion tables were regenerated. A refit without the exclusion reproduces the earlier knock-level predictions exactly
  (`src/nested/cnn_clean.py`).
- **Effect:**
  - Sound: 0.876/0.803 → 0.889/0.810.
  - Nested 4-sensor stacking: 0.899/0.839 → 0.913/0.854.
  - T+S+R mean rule: 0.907/0.848 → 0.899/0.832.
  - No fusion gain over sound is significant (McNemar p ≥ 0.19).
  - The random-split sound "deflation" (up to −26 pp) disappears (±1.4 pp).
  - The iPhone–Android gap disappears: recording accuracy 0.88 vs 0.90; cross-device transfer 0.81–0.86.
  - Unchanged: LOBO collapse and image-sensor leakage.
- `results/sound_matching_sensitivity.csv` gives metrics by sound-evidence stratum and for the iPhone-only variant.
- The v1.1 tables are kept in `results_v1_1_pre_sound_audit/`.

## v1.1 — nested stacking (response to review)

- **Fixed:** the stacked logistic-regression meta-learner was trained on global out-of-fold base-model
  probabilities. Each OOF prediction for a training session came from a base model whose training set
  contained the outer test fold, so test labels influenced the stacker indirectly.
- **Now:** fully nested stacking (`src/nested/`). For each outer test fold k and each other fold j, base
  models are refit on the three remaining folds with their own inner GroupKFold grid search. This yields 10 fold-pair
  refits per sensor. The stacker for fold k sees only inner-OOF predictions and is tuned by GroupKFold(3).
  - sklearn refits reproduce the stored global OOF exactly (max diff 0.0).
  - CNN refits use the same architecture, epochs, and seed.
- **Effect:** four-modality stacking went from 0.913/0.857 to 0.899/0.839 (acc/macro-F1). The two-class
  four-modality stacking went from 0.989 to 0.978. Other stacking subsets moved by −4.4 to +3.0 pp
  (`results/nested_vs_global_stacking.csv`). The mean/product rules, early fusion, single sensors,
  leakage and LOBO tables are unchanged.
- The old tables are kept in `results_v1_global_stacking/` for audit.
- `intermediate/nested/` holds the fold-pair base-model probabilities, so `run_all.sh` needs no retraining.
  Regenerating them requires the raw archives and `python src/nested/base_models.py` and `python src/nested/cnn_pairs.py`
  (about 10 min per CNN fit on CPU), followed by `python src/nested/nested_stacking.py`.

## 1.0.0 - 2026-10-28  (bundle prepared for GitHub / Zenodo)
- First public bundle: frozen splits, cleaned analysis scripts, intermediate out-of-fold predictions, final
  result tables.  The fusion stage (`src/fusion_pipeline.py`) was re-run from this bundle and reproduces
  `results/fusion_results.csv` and all derived tables exactly (max |diff| = 0 on every numeric column).

### Fold table v2 (`data/durian_mm_capture_folds.csv`) - sound rows de-duplicated
- v1 of the frozen capture table listed each sound recording once per *thermal session* of the same fruit,
  giving 1002 sound rows for 538 physical recordings (the knock files carry no capture date, so a recording
  could not be assigned to one of the fruit's two sessions from its name).  v2 keeps one row per recording
  (538 rows).  Fold assignments (`fold_fruit`, `fold_random`), labels, fruit ids and all rows of the three
  image modalities are identical between v1 and v2 (verified cell by cell).
- The thermal and multispectral out-of-fold files in `intermediate/oof/` were produced with v1 and the RGB
  files with v2; because the non-sound rows are identical this has no effect on any result.
- Sound results were computed on `data/durian_mm_sound_recordings.csv` (538 rows): the de-duplicated table
  whose `date_captured` was assigned by matching each recording's label to the date on which the image
  modalities recorded that fruit with the same label (`date_src = matched_label`, 524 rows) or, for 14
  recordings whose label does not occur in the image modalities, the fruit's endpoint date
  (`date_src = endpoint_fallback`).  Column `date_captured_frozen` keeps the v1/v2 date for traceability.
- KNOWN DISCREPANCY (not corrected here, to keep the published table byte-identical to what the paper's
  per-modality runs used): the 538 sound rows of v2 carry `date_captured` values that match the image
  modalities' session dates for only 146 of 269 sound sessions; the recordings table above aligns all 269.
  Anyone re-running the sound track must use `durian_mm_sound_recordings.csv` (the scripts do), not the
  sound rows of the capture table.  A v3 that merges the matched dates into the capture table is planned.

### Known limitations of the analysis code
- The per-modality tracks were developed interactively; `src/session_traces/*.py` are the literal executed
  cells, and `src/<modality>/*_models.py` / `*_pipeline.py` are consolidated final-state drivers that were
  syntax-checked but not re-executed end-to-end for this bundle (they require the raw Zenodo archives and
  several hours of CPU time).  Their expected outputs are shipped in `intermediate/`.
- Scikit-learn/LightGBM/PyTorch results on CPU are deterministic for the seeds used (42; 42+k per CNN fold)
  on the machine where they were produced; bit-identical reproduction across BLAS builds is not guaranteed
  for the trained models, whereas the fusion stage is exactly reproducible from the shipped OOF files.
