#!/usr/bin/env bash
# Reproduce every table of the paper from the shipped cached features and predictions (no raw data needed; ~45 min on 4 cores).
# Each stage regenerates its outputs into results_repro/ and compares them with the shipped files; the script stops at the
# first difference larger than 1e-9.  See REPRODUCE.md for the table -> stage map and for the deeper levels:
#   REFIT=1  refit all 20 classical candidates on the primary partition from cached features and compare (~6 min)
#   RAW=1    re-extract every feature table from the Zenodo archives and compare (needs DURIAN_DATA_ROOT; ~30 min)
#   CNN=1    retrain the five primary-partition knock CNNs on CPU, export their weights and compare (~70 min; needs RAW=1
#            in the same call or an existing work/raw/sound directory, because the log-mel array is regenerated from raw audio)
#   LABELSENS=1  label-policy sensitivity with retraining, results/v20 (Table 11 right-hand columns, S23; LABELSENS_REFIT=1 refits)
#   ANCHOR=1     pre-registered intake-anchored calibration and physics-guided acoustics, results/v21 (~25 min; ANCHOR_CNN=1
#                retrains the 17 knock CNNs, ~2.5 h)
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}   # keep >= 4: with 1 thread three lbfgs logistic fits drift at the 1e-2 level
export PYTHONWARNINGS=ignore DURIAN_INTERMEDIATE=intermediate_common EMB_MOBILENET=intermediate_common/rgb/emb_mobilenetv2.npy
CMP="python src/audit/compare_tables.py"
mkdir -p results_repro
echo "[1/7] fusion pipeline and sensitivity tables -> results_repro/ (Tables 5-11; S1-S6, S9, S11)"
python -c "import json; c=json.load(open('src/fusion_config.json')); c['out_dir']='results_repro'; json.dump(c, open('fusion_config_repro.json','w'), indent=1)"
python src/fusion_pipeline.py --config fusion_config_repro.json > results_repro_run.log
python src/verify_reproduction.py results results_repro
python src/sensitivity/sensitivity_tables.py results_repro        # Tables 10, 11; S6, S9, S11 (were interactive cells before v1.8)
$CMP results results_repro label_sets.csv sound_matching_sensitivity.csv label_policy_sensitivity.csv nested_vs_global_stacking.csv
echo "[2/7] figures -> figures/"
DURIAN_RESULTS=results DURIAN_FIGURES=figures python src/fusion_figures.py
echo "[3/7] nested selection, ten partitions, dependence-aware tests -> results_repro/v14 (Tables 3-4, S12-S14)"
DURIAN_V14_OUT=results_repro/v14 python src/nested_selection/evaluate_v14.py
$CMP results/v14 results_repro/v14
echo "[4/7] targeted ablations -> results_repro/v15 (Table 12, S15-S17, S19)"
DURIAN_V15_OUT=results_repro/v15 python src/ablations/evaluate_v15.py
DURIAN_V15_OUT=results_repro/v15 python src/ablations/ms_wavelength_ablation.py
$CMP results/v15 results_repro/v15
echo "[5/7] development pipelines vs selection procedures, matched candidate space -> results_repro/v17; macro-F1 summaries and paired comparisons -> results_repro/v19 (Tables 3-4, S20-S22)"
DURIAN_V17_OUT=results_repro/v17 python src/nested_selection/matched_space.py
$CMP results/v17 results_repro/v17 "*.csv" "*.parquet"
DURIAN_V17_IN=results_repro/v17 DURIAN_V19_OUT=results_repro/v19 python src/nested_selection/primary_metric.py > /dev/null
$CMP results/v19 results_repro/v19
echo "[6/7] Supplementary Tables S1-S25 from the regenerated results -> results_repro/supplementary"
DURIAN_RESULTS=results_repro python src/audit/build_supplementary_tables.py results_repro/supplementary
$CMP audit/supplementary results_repro/supplementary
echo "[7/7] audit tables, model registry, out-of-fold prediction export, saved-model check -> results_repro/audit"
DURIAN_RESULTS=results_repro DURIAN_AUDIT_OUT=results_repro/audit python src/audit/build_audit_tables.py
DURIAN_AUDIT_OUT=results_repro/audit python src/audit/model_registry.py > /dev/null
DURIAN_RESULTS=results_repro DURIAN_AUDIT_OUT=results_repro/audit python src/audit/export_predictions.py
DURIAN_AUDIT_OUT=results_repro/audit python src/audit/table1.py > /dev/null
DURIAN_AUDIT_OUT=results_repro/audit CAND_NJ=2 python src/audit/search_log.py > /dev/null
$CMP audit results_repro/audit fruits.csv sessions.csv fruit_partitions.csv exclusion_ledger.csv model_registry.csv configuration_space.csv \
     table1_dataset_structure.csv hyperparameter_search_p0.csv "predictions/*.csv" "predictions/*.parquet"
DURIAN_AUDIT_OUT=results_repro/audit python src/audit/export_models.py check
if [ "${REFIT:-0}" = "1" ]; then
  echo "[REFIT] classical candidates on the primary partition from cached features -> results_repro/fits"
  JOBS=$(python -c "import sys; sys.path.insert(0,'src/nested_selection'); import cands; print(' '.join(f'{m}:{c}' for m,l in cands.CANDIDATES.items() for c in l if c!='cnn'))")
  CAND_NJ=2 python src/nested_selection/run_fits.py results_repro/fits p0 $JOBS
  python src/audit/compare_fits.py intermediate_v14/fits/p0 results_repro/fits/p0
fi
if [ "${RAW:-0}" = "1" ]; then
  echo "[RAW] feature tables from the Zenodo archives (DURIAN_DATA_ROOT=${DURIAN_DATA_ROOT:?set DURIAN_DATA_ROOT}) -> work/raw"
  python src/audit/verify_raw_features.py work/raw
fi
if [ "${CNN:-0}" = "1" ]; then
  echo "[CNN] primary-partition knock CNNs on CPU (3 threads, as in the paper) with weight export -> models/primary_p0"
  ( cd work/raw/sound && CNN_THREADS=3 OMP_NUM_THREADS=3 CNN_SAVE_WEIGHTS=../../../models/primary_p0 \
      python ../../../src/common_label/cnn_common.py outer:0 outer:1 outer:2 outer:3 outer:4 )
  python src/audit/compare_fits.py intermediate_v14/cnn_cpu_p0 work/raw/sound/handoff "cnn_common_outer_*.npy"
  DURIAN_MELS=work/raw/sound/handoff/mels.npy python src/audit/export_models.py check
fi
if [ "${LABELSENS:-0}" = "1" ]; then
  # Label-policy sensitivity with retraining (Table 11 right-hand columns, S23). Re-evaluates the shipped retrained predictions;
  # LABELSENS_REFIT=1 also refits the thermal and RGB components (~10 min). The knock-CNN retraining (6 x 5 CPU fits) is documented in
  # src/label_sensitivity/retrain_without_questionable.py and shipped as predictions in results/v20/cnn/.
  echo "[LABELSENS] retraining without questionable sessions + matched random-removal control -> results_repro/v20"
  O=results_repro/v20; rm -rf $O; mkdir -p $O
  cp -R results/v20/cnn results/v20/question_sets.csv results/v20/control_sessions.csv $O/
  if [ -d work/raw/sound/handoff ]; then python src/label_sensitivity/retrain_without_questionable.py prepare $O work/raw/sound > /dev/null
       $CMP results/v20 $O question_sets.csv distribution_evaluation_sessions.csv distribution_training_captures.csv
  fi
  cp -R results/v20/fits $O/                     # includes the random-removal control fits
  if [ "${LABELSENS_REFIT:-0}" = "1" ]; then python src/label_sensitivity/retrain_without_questionable.py classical $O > /dev/null; fi
  python src/label_sensitivity/retrain_without_questionable.py evaluate $O > /dev/null
  python src/label_sensitivity/retrain_without_questionable.py control_evaluate $O > /dev/null
  $CMP results/v20 $O results.csv comparisons.csv control_results.csv
fi
if [ "${ANCHOR:-0}" = "1" ]; then
  # Pre-registered intake-anchored calibration and physics-guided acoustics (docs/ANALYSIS_PLAN_anchor_physics.md, tag
  # prereg-anchor-physics; deviations in results/v21/DEVIATIONS.md). ~25 min on 4 cores from the shipped knock-CNN predictions.
  # ANCHOR_CNN=1 retrains the 17 knock CNNs on CPU (M0/M1/M2 LOBO, M1 rerun with weights, M1 primary partition; ~2.5 h on 9 cores,
  # needs work/raw/sound) and re-derives the AdaBN/logit-shift/EM variants and the CNN anchor sensitivity.
  echo "[ANCHOR] intake-anchored calibration and physics-guided acoustics -> results_repro/v21"
  O=results_repro/v21; rm -rf $O; mkdir -p $O; OA=$(cd $O && pwd)
  python src/anchor_physics/prepare.py anchors $O > /dev/null
  $CMP results/v21 $O anchor_sessions.csv; cmp results/v21/anchor_recordings.txt $O/anchor_recordings.txt
  if [ "${RAW:-0}" = "1" ]; then python src/anchor_physics/prepare.py physics $O > /dev/null; $CMP results/v21 $O "physics_*.parquet"
  else cp results/v21/physics_*.parquet $O/; fi
  python src/anchor_physics/anchor_sens.py $O draws > /dev/null; $CMP results/v21 $O anchor_draws.csv
  if [ "${ANCHOR_CNN:-0}" = "1" ]; then
    R=$(pwd); C=$OA/cnn
    ( cd work/raw/sound && export CNN_THREADS=3 OMP_NUM_THREADS=3 && L="lobo:Immature lobo:Mature lobo:Overmature"
      CNN_OUT_DIR=$C/M0 CNN_SAVE_WEIGHTS=$C/M0_weights python $R/src/common_label/cnn_common.py $L > $C.M0.log 2>&1 &
      CNN_NORM=anchor CNN_ANCHOR_FILE=$OA/anchor_recordings.txt CNN_OUT_DIR=$C/M1 CNN_SAVE_WEIGHTS=$C/M1_weights python $R/src/common_label/cnn_common.py $L > $C.M1.log 2>&1 &
      CNN_NORM=all CNN_OUT_DIR=$C/M2 python $R/src/common_label/cnn_common.py $L > $C.M2.log 2>&1 &
      wait
      CNN_NORM=anchor CNN_ANCHOR_FILE=$OA/anchor_recordings.txt CNN_OUT_DIR=$C/M1_outer python $R/src/common_label/cnn_common.py outer:0 outer:1 outer:2 outer:3 outer:4 > $C.M1_outer.log 2>&1
      ANCHOR_OUT=$OA python $R/src/anchor_physics/cnn_adapt.py $R > /dev/null
      ANCHOR_OUT=$OA python $R/src/anchor_physics/cnn_anchor_sens.py $R > /dev/null )
    for d in M0 M1 M2 M1_outer; do python src/audit/compare_fits.py results/v21/cnn/$d $C/$d "cnn_common_*.npy"; done
  else
    cp -R results/v21/cnn $O/; cp results/v21/lobo_cnn_sessions.parquet results/v21/anchor_sens_cnn.parquet results/v21/cnn_controls.json $O/
  fi
  python src/anchor_physics/lobo_classical.py $O > /dev/null
  python src/anchor_physics/baselines.py $O > /dev/null
  python src/anchor_physics/within_batch.py $O > /dev/null
  python src/anchor_physics/anchor_sens.py $O classical > /dev/null
  python src/anchor_physics/evaluate.py $O > /dev/null
  $CMP results/v21 $O "*.csv" "*.parquet"
  # rounds 2 and 3 (docs/ANALYSIS_PLAN_round2.md, _round3.md) from the shipped round-2 knock-CNN session probabilities
  # (src/anchor_physics/round2_cnn.py after the 16 CNN fits listed in the plan; not retrained by this script)
  O2=results_repro/round2; O3=results_repro/round3; rm -rf $O2 $O3; mkdir -p $O2 $O3
  cp results/round2/cnn_sessions.parquet $O2/
  ROUND_V21=$O ROUND2_DIR=$O2 python src/anchor_physics/round2.py classical > /dev/null
  ROUND_V21=$O ROUND2_DIR=$O2 python src/anchor_physics/round2.py evaluate > /dev/null
  $CMP results/round2 $O2 "*.csv" "*.parquet"
  ROUND_V21=$O ROUND2_DIR=$O2 ROUND3_DIR=$O3 python src/anchor_physics/round3.py evaluate > /dev/null
  $CMP results/round3 $O3 "*.csv"
  # round 4 (docs/ANALYSIS_PLAN_round4.md): fruit-specific baseline, from the shipped knock-CNN predictions (results/round4/cnn;
  # 48 CPU fits of src/anchor_physics/paired_cnn.py, ~2.5 h on 9 cores, not rerun here)
  O4=results_repro/round4; rm -rf $O4; mkdir -p $O4; cp -R results/round4/cnn $O4/
  ROUND_V21=$O ROUND4_DIR=$O4 python src/anchor_physics/round4.py classical > /dev/null
  ROUND_V21=$O ROUND4_DIR=$O4 python src/anchor_physics/round4.py evaluate > /dev/null
  $CMP results/round4 $O4 "*.csv" "*.parquet"
fi
echo "done: all regenerated tables are identical to the shipped ones"
