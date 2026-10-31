# Reproduction entry points.  Run from the repository root.  See REPRODUCE.md.
#   make reproduce - every table from cached predictions, checked against the shipped files (= ./run_all.sh)
#   make refit     - plus the 20 classical candidates refitted on the primary partition from cached features
#   make raw       - plus features, embeddings and primary-partition knock CNNs from the raw archives (DURIAN_DATA_ROOT)
#   make audit     - rebuild audit/ (exclusion ledger, sessions, registry, predictions, S-tables) in place
#   make fusion    - re-derive every table in results/ from the shipped out-of-fold predictions (~3 min, CPU)
#   make figures   - fusion / leakage / LOBO figures into figures/
#   make verify    - compare results_repro/ with the shipped results/ (prints max |diff| per table)
#   make thermal | multispectral | sound | rgb  - full per-modality re-runs (need DURIAN_DATA_ROOT, hours)
#   make all       - fusion + figures + verify
PY      ?= python
SRC     := $(abspath src)
export DURIAN_DATA_ROOT ?= $(abspath raw)
export DURIAN_N_JOBS    ?= 4
export OMP_NUM_THREADS  ?= 4
export DURIAN_INTERMEDIATE ?= intermediate_common
export EMB_MOBILENET ?= intermediate_common/rgb/emb_mobilenetv2.npy

.PHONY: all fusion figures verify thermal multispectral sound rgb clean reproduce refit raw audit

all: fusion figures verify

fusion:
	$(PY) $(SRC)/fusion_pipeline.py --config $(SRC)/fusion_config.json

results_repro/fusion_results.csv:
	$(PY) -c "import json; c=json.load(open('$(SRC)/fusion_config.json')); c['out_dir']='results_repro'; json.dump(c, open('fusion_config_repro.json','w'), indent=1)"
	$(PY) $(SRC)/fusion_pipeline.py --config fusion_config_repro.json

verify: results_repro/fusion_results.csv
	$(PY) $(SRC)/verify_reproduction.py results results_repro

figures:
	DURIAN_RESULTS=results DURIAN_FIGURES=figures $(PY) $(SRC)/fusion_figures.py

# ---- per-modality tracks (raw Zenodo archives required; outputs land in work/<modality>/results) ----
thermal:
	mkdir -p work/thermal && cd work/thermal && $(PY) $(SRC)/thermal/run_extract.py && $(PY) $(SRC)/thermal/thermal_models.py && $(PY) $(SRC)/thermal/run_cnn.py fold_fruit

multispectral:
	mkdir -p work/multispectral && cd work/multispectral && $(PY) $(SRC)/multispectral/extract_all.py && $(PY) $(SRC)/multispectral/ms_run_models.py

sound:
	mkdir -p work/sound && cd work/sound && $(PY) $(SRC)/sound/sound_pipeline.py features && $(PY) $(SRC)/sound/sound_pipeline.py classical \
	  && $(PY) $(SRC)/sound/cnn_train.py fold_fruit 20 && $(PY) $(SRC)/sound/cnn_train.py fold_random 20 \
	  && $(PY) $(SRC)/sound/cnn_extra.py lobo 20 && $(PY) $(SRC)/sound/cnn_extra.py crossdevice 20 \
	  && $(PY) $(SRC)/sound/sound_pipeline.py cnn_aggregate

rgb:
	mkdir -p work/rgb && cd work/rgb && $(PY) $(SRC)/rgb/build_cache.py && $(PY) $(SRC)/rgb/extract_features.py \
	  && $(PY) $(SRC)/rgb/rgb_models.py handcrafted && $(PY) $(SRC)/rgb/rgb_models.py embed && $(PY) $(SRC)/rgb/rgb_models.py probe

reproduce:
	./run_all.sh

refit:
	REFIT=1 ./run_all.sh

raw:
	REFIT=1 RAW=1 CNN=1 ./run_all.sh

audit:
	$(PY) src/audit/build_supplementary_tables.py
	$(PY) src/audit/build_audit_tables.py
	$(PY) src/audit/model_registry.py
	$(PY) src/audit/export_predictions.py
	$(PY) src/audit/table1.py
	$(PY) src/audit/search_log.py

clean:
	rm -rf results_repro fusion_config_repro.json
