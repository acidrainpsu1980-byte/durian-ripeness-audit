# durian-ripeness-audit (v2.3.0)

Reproducible audit package for *"Fruit-Grouped and Batch-Aware Evaluation of Thermal, Multispectral, Knocking-Sound and RGB
Sensing for Durian Ripeness Classification on a Public Longitudinal Dataset"* (prepared for *Computers and Electronics in Agriculture*, 2026).

The paper classifies the ripeness of a durian fruit **on the day of capture** (Unripe / Ripe / Overripe) from thermal images,
8-band NIR multispectral images, knocking sound and RGB photographs of the public dataset of Mesa-Satina et al. (2026,
doi:10.5281/zenodo.18603796), under fruit-grouped cross-validation on ten fruit partitions, leave-one-harvest-batch-out and
capture-level leakage controls. This repository contains everything needed to audit and reproduce it:

| Reviewer-requested item | Where |
|---|---|
| Recording exclusions with the evidence for each | `audit/exclusion_ledger.csv`, `data/raw_file_inventory.csv`, `data/sound_recording_manifest.csv` (S7), `data/sound_session_evidence.csv` (S8), `data/sound_exclude_wrong_session.txt` |
| Session identifiers, label decisions, the ten fruit partitions | `audit/sessions.csv`, `data/session_labels_common.csv` (S10), `audit/fruit_partitions.csv`, `audit/fruits.csv` |
| Candidate models, hyper-parameter searches, software versions, training settings | `audit/model_registry.csv`, `audit/configuration_space.csv`, `audit/hyperparameter_search_p0.csv`, `audit/SOFTWARE.md`, `audit/requirements_lock_cpu.txt`, `models/primary_p0/` |
| Out-of-fold predictions and scripts reproducing the principal tables | `audit/predictions/`, `audit/supplementary/` (S1–S29), `audit/table_map.csv`, `run_all.sh` |
| Commands from raw data and from cached features | `REPRODUCE.md` |

Details: `audit/README.md`. Version history: `CHANGELOG.md`.

## Quick start

```bash
conda env create -f environment.yml && conda activate durian
./run_all.sh                               # every table from cached predictions, compared with the shipped files (level A)
REFIT=1 ./run_all.sh                       # + classical candidates refitted from cached features (level B)
DURIAN_DATA_ROOT=/path/to/zips RAW=1 CNN=1 ./run_all.sh    # + everything from the raw Zenodo archives (level C)
```

## Main results (session level, 268 sessions of 164 fruits with all four sensors)

Primary estimate = selection procedure (model family and configuration chosen inside each outer training fold), mean over ten
fruit partitions (partition 0: CPU knock CNN; partitions 1–9: GPU refits): knocking sound session macro-F1 0.803 (range 0.767–0.835;
accuracy 0.875), best single sensor in all 50 outer folds; thermal 0.738, RGB 0.701, multispectral 0.663; fusions added 1.8–2.8 pp with
every confidence interval including zero. Under leave-one-harvest-batch-out every sensor and fusion
falls to about the majority rate. Development pipelines (families fixed after inspecting the primary partition) are reported
separately in the paper (sound 0.899 on the primary partition, 0.885 held fixed over ten partitions). All numbers are read from
`results/` (see `audit/table_map.csv`).

## Sequential cross-batch adaptation rounds (v2.0–v2.3)

Seven sequential, prospectively specified exploratory rounds of label-free adaptation to a held-out harvest batch. Each round's plan was
committed and tagged before fitting (`docs/ANALYSIS_PLAN_anchor_physics.md`, `docs/ANALYSIS_PLAN_round2.md` … `_round7.md`; tags
`prereg-anchor-physics`, `prereg-round2` … `prereg-round7`), and its predictions and deviation list were tagged before its single evaluation
(`pre-evaluation-anchor-physics`, `pre-evaluation-round2` … `pre-evaluation-round5`, `pre-evaluation-round7`; both round-7 tags were pushed
and verified on the remote before the next step). The round-6 pre-evaluation tag was lost before it was
pushed; `results/round6/DEVIATIONS.md` (item 7) gives the md5 sum and archive timestamps that document the order of events instead.
Each round after the first was designed after the earlier results on the same three batches had been seen, so the rounds are exploratory.

The reference is a protocol-gated acoustic classifier (CDR-RO): Unripe for intake scans, otherwise the more probable of Ripe and Overripe
under the unadapted knock CNN (endpoint macro-F1 0.656 on rounds 1–3, 0.661 on rounds 4–7). The endpoint task classifies sessions that the
study design identifies as endpoints; the dataset cannot evaluate later scans that are still Unripe. No round met its superiority
criterion; the best selected arm (round 6) reached 0.759 (+9.9 pp [−1.1, 20.2]), with per-batch differences of +13.9, −7.3 and +33.9 pp
(`results/round6/per_batch_ci.csv`, post hoc). Round 7 (Rank-then-Anchor: intake-centred embeddings with a within-batch ranking head)
scored 0.670 (+1.0 pp [−10.7, 11.7]) and met none of its pre-specified wording conditions. A post hoc oracle-threshold bound
(`src/foundation/posthoc_oracle_headroom.py`) gives the embedding fusion 0.798 with per-batch thresholds chosen on the batch's labels. Code: `src/anchor_physics/`, `src/foundation/`; results: `results/v21/` (round 1),
`results/round2/` … `results/round7/`; Supplementary Tables S24–S29. Planning material for a new-harvest study (fruit-level power
illustration and between-harvest scenarios): `docs/protocol/`, `src/protocol/`.

## Repository layout

```
REPRODUCE.md            reproduction levels A-C, stage -> table map, environment and tolerance notes
run_all.sh, Makefile    entry points (make reproduce | refit | raw | audit)
audit/                  audit tables, model registry, hyper-parameter search log, prediction export, S-tables, checks
models/primary_p0/      fitted models of the four development pipelines, one per outer fold (+ model_check.csv)
data/                   frozen split tables, fruit table, session labels, sound provenance manifests, raw-archive inventory,
                        literature extraction (Table S18), fruit log copied from the Zenodo record (data/zenodo/)
src/                    code: per-modality tracks (thermal/, multispectral/, sound/, rgb/), common_label/ (v1.3 refits),
                        nested_selection/ (candidates, refits, nested selection, partitions), ablations/, sensitivity/,
                        audit/ (inventory, ledger, registry, exports, checks), fusion_pipeline.py, session_traces/ (literal
                        interactive cells, provenance only)
intermediate_common/    v1.3 common-label features, embeddings, out-of-fold predictions, nested stacking
intermediate_v14/       refits of all candidates on ten partitions (fits/), knock-CNN refits (cnn/ GPU, cnn_cpu_p0/ CPU)
intermediate_v15/       ablation inputs (harmonized thermal features, masked-crop embeddings, device-specific CNNs)
intermediate/           v1.0-v1.2 file-label outputs still read by the fusion stage (leave-one-batch-out tables)
results/                result tables (v14, v15, v17, v19 subfolders by revision); results_v1_* = superseded versions
```

## Data

Mesa-Satina, A. R.; Kobayashi, V. B.; Bayogan, E. R.; Calag, V. B. (2026). *Multi-Modal Sensor Data for Durian Fruit Maturity
Classification and Ripeness Assessment*. Zenodo. https://doi.org/10.5281/zenodo.18603796 (CC BY 4.0; University of the
Philippines Mindanao; 189 fruits, Davao, 2025 season). The raw archives (23.4 GB) are not redistributed; only
`durian_characteristics_cleaned.csv` (the fruit log, 46 kB) is copied to `data/zenodo/` with attribution.

```bash
B=https://zenodo.org/api/records/18603796/files
for f in dataset_clean_thermal.zip dataset_clean_multispectral.zip dataset_clean_sound.zip dataset_clean_rgb.zip; do
  curl -L -C - --retry 5 -o "$DURIAN_DATA_ROOT/$f" "$B/$f/content"; done
(cd "$DURIAN_DATA_ROOT" && md5sum -c /path/to/repo/data/zenodo_checksums.md5)        # macOS: md5 -r <file>
```

| file | bytes | md5 |
|---|---|---|
| dataset_clean_thermal.zip | 536,341,191 | 7410bb66b1b6f697a01aae537ed06a08 |
| dataset_clean_multispectral.zip | 973,580,421 | ec527131bd147da45d1d7a0f1e008048 |
| dataset_clean_sound.zip | 1,624,075,464 | 728b19fd8f3447c506e0b66964e1d520 |
| dataset_clean_rgb.zip | 20,283,680,926 | a5dd18da3d124f151f4a8711ae6c45e1 |
| durian_characteristics_cleaned.csv | 45,923 | 536a80e41c28dd3e9e4d8159019d598c |

## Citing and licence

Please cite the paper and this package (`CITATION.cff`) **and** the dataset. Code: MIT licence (`LICENSE`). Derived tables and the
copied fruit log: CC BY 4.0, with attribution to the dataset authors.

Contact: [TODO: corresponding author and e-mail]
