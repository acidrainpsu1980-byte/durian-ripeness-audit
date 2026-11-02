# Analysis plan, round 5: pretrained audio and vision foundation encoders under batch-disjoint evaluation

Status: **written and committed before any round-5 model was fitted or scored.**
- The tag `prereg-round5` precedes every result file of this round (`results/round5/`).
- Deviations will be listed in `results/round5/DEVIATIONS.md` and in the paper.
- Frozen embeddings were extracted before this plan was tagged. Extraction is label-free: no label, score or classifier was computed. The embedding files are fixed by the md5 sums in Section 8.

## 0. Disclosure

**Prior rounds on the same sessions.** Rounds 1–4 (tags `prereg-anchor-physics`, `prereg-round2`, `prereg-round3`, `prereg-round4`) were evaluated on the same endpoint sessions before this plan was written.
- The round-3 plan called round 3 the final round; round 4 revised that statement.
- Round 5 is a further revision, made after round 4 was found not beneficial. The paper had not been submitted anywhere when this plan was written.
- The paper will report all five rounds.
- Round-5 results are exploratory and need a new harvest to confirm.

**Known before this plan** (104 LOBO endpoint sessions, macro-F1 over Ripe/Overripe, gated):
- CDR-RO 0.661;
- round-4 current-knock CNN K1 0.567;
- paired K4 0.548;
- change-only K5 0.642 (exploratory);
- shuffled-baseline K6 0.585;
- intake-only K2 0.541 and L2 0.610;
- Overmature Ripe recall 0 for K1, K3, K4, K6, CDR-RO and M1-RO.

The change-only signal (K5) has therefore already been seen; the change arms of this round are not confirmatory for it.

**Hypothesis.** General-purpose encoders pretrained on large audio or image corpora carry representations that transfer across harvest batches better than the task-specific knock CNN trained on about 70 endpoint fruits. Adjacent fields give mixed evidence (see `docs/novelty/novelty_memo_foundation.md`), so this round is designed to report a negative result.

## 1. Population and protocol information (unchanged from round 4)

- **Population.** The 104 endpoint sessions of round 4 (Immature 34, Mature 41, Overmature 29). Every one has all four sensors and a verified intake recording of the same fruit.
- **Gating.** Every method is gated: intake sessions are predicted Unripe; endpoint sessions are classified Ripe or Overripe. The false-Unripe rate is 0 by construction.
- **Excluded inputs.** No method uses storage time, the time between visits, end-of-storage weight, weight loss or any endpoint label of the test fruits.
- **Training data per split.** The non-intake sessions labelled Ripe or Overripe (common label) of the training fruits that have a baseline, as in round 4.
  - Audio: training units are verified recordings (knock embeddings averaged within the recording).
  - Vision: training units are images.
- **Session probability.** The mean over the session's units.
- **Splits.**
  - Primary: leave-one-batch-out (LOBO, train on two harvest batches, test on the third).
  - Secondary: fruit-grouped 5-fold CV on the primary partition (`fold_fruit`, p0).

## 2. Frozen embeddings (label-free; extracted before this plan)

**Audio, per knock.**
- **Knocks.** All 15,709 knocks of `features_sound_knocks.parquet`, at the same onsets as the paper's CNN.
- **Window.** Re-cut from the original-rate recordings (iPhone m4a, Android wav) and resampled once to each encoder's native rate.
- **Primary window W150 (150 ms).** Each encoder's own padding convention:
  - BEATs: zero-pad to 1.0 s with `padding_mask`;
  - CNN14: zero-pad to 1.0 s;
  - AST: feature-extractor zero-pad to 1,024 frames;
  - CLAP: `repeatpad` to 10 s;
  - wav2vec2/HuBERT: no padding.
- **Sensitivity window W500.** 0.5 s of real context.

| Code | Encoder | Checkpoint | Clip-level embedding |
|---|---|---|---|
| BEATS | BEATs iter3+ AS2M, self-supervised | `BEATs_iter3_plus_AS2M.pt` | mean of last-layer tokens where `padding_mask` is False (768) |
| CNN14 | PANNs CNN14 | `Cnn14_mAP=0.431.pth` (Zenodo 3987831) | built-in embedding (2048) |
| CLAP | LAION CLAP | `laion/clap-htsat-unfused` | `get_audio_features`, L2-normalised (512; file feature `repeatpad_projl2`) |
| AST | AST AudioSet | `MIT/ast-finetuned-audioset-10-10-0.4593` | `pooler_output` (768) |
| W2V | wav2vec 2.0 base | `facebook/wav2vec2-base` | mean over the 12 layers of frame-means (768) |
| HUB | HuBERT base | `facebook/hubert-base-ls960` | same rule (768) |

**Vision, per image.** Every image is padded to square and resized to 224.

| Code | Encoder | Checkpoint | Embedding |
|---|---|---|---|
| DINO | DINOv2 ViT-B/14 | `facebook/dinov2-base` | `pooler_output` (CLS, 768) |
| SIGLIP | SigLIP base/16-224 | `google/siglip-base-patch16-224` | vision `pooler_output` (768) |
| CLIP | CLIP ViT-B/16 (descriptive only) | `openai/clip-vit-base-patch16` | `get_image_features` (512) |

Image inputs:
- **Thermal.** °F → °C, clipped to a fixed 20–40 °C window, scaled to 0–255, grey replicated to 3 channels. A per-image min–max variant is used only as a sensitivity analysis.
- **Multispectral.** Each of the 8 bands is embedded separately, with a fixed per-band intensity range and grey replicated to 3 channels. The 8 band embeddings are concatenated in wavelength order (8 × D).
- **RGB.** Original image; the camera is kept in the index.

## 3. Models (fixed; no further candidates will be added)

**One head for every frozen arm.**
- **Estimator.** The paper's `embedding_logreg`: StandardScaler, then class-balanced L2 logistic regression.
- **Tuning.** C ∈ {0.001, 0.01, 0.1}, chosen by inner fruit-grouped 3-fold CV on the training fruits only (macro-F1), exactly as `lobo_classical.estimator("embedding_logreg")`.
- **No other dimensionality reduction**, so that no data-driven step precedes the head.

**Audio arms (W150 unless stated).**
- **A1 BEATs current: h(z_t). Primary audio arm.**
- A2 CNN14 current, A3 CLAP current, A4 AST current.
- A5 wav2vec2 current, A6 HuBERT current (descriptive; speech-pretraining contrast).
- A1c BEATs change-only, h(z_t − z_0). z_0 is the mean embedding of the fruit's intake recordings under the label-free intake rule (as in round 4); z_t is the endpoint recording embedding.
- A1p BEATs paired, h([z_t, z_t − z_0]).
- A1s BEATs change-only with a shuffled baseline. z_0 is replaced by the intake mean of a donor fruit drawn within the same (batch, `fold_fruit`) group, using the round-4 derangement (rng 42).
- A1b BEATs baseline-only, h(z_0) (diagnostic).
- A1w BEATs current on W500 (sensitivity).
- **A1ft BEATs LP-FT** (the only fine-tuned arm).
  - **Stage 1.** Fit a linear head on frozen knock embeddings.
  - **Stage 2.** Unfreeze the top 2 of 12 transformer blocks plus the head.
  - **Optimiser.** AdamW, lr 2e-5 for the encoder and 1e-3 for the head, weight decay 0.01, 8 epochs, batch 64 knocks, class-balanced cross-entropy, no early stopping.
  - **Data.** Knock-level training on the training split's verified endpoint recordings. Test knocks are averaged to the recording, then to the session.
  - **Seeds.** 0, 1 and 2; seed-averaged probabilities.
  - **Nesting.** Fold-nested: no audio of the held-out batch or fold enters any gradient step.

**Vision arms (per modality).** V-T thermal, V-M multispectral (8-band concatenation) and V-R RGB, each with DINO (primary) and SIGLIP.
- V-T also has the min–max sensitivity variant.
- CLIP is descriptive.

**Multimodal arms.**
- **F1: equal-weight mean of the R/O probabilities of A1, V-T DINO, V-M DINO and V-R DINO. Primary multimodal arm.**
- F2: as F1, with SIGLIP for the three image modalities.
- F3: embedding fusion at session level. The per-modality session embeddings (A1, DINO ×3) are each standardised on the training split and scaled by 1/√dim, so that every modality contributes equal total variance. They are then concatenated and fed to the same head. Only sessions that have all four modalities are used.
- F4: F1 without RGB (camera-confound control).
- F5: mean of the three DINO image arms (vision only).

**References (gated, same 104 sessions).**
- CDR-RO (round-1 M0 knock CNN);
- K1 (round-4 current-knock CNN, architecture of the paper);
- M1-RO.

## 4. Evaluation

**Primary score.** LOBO pooled macro-F1 over Ripe and Overripe on the 104 sessions.

**Secondary scores.**
- per-batch macro-F1, Ripe recall and Overripe recall, with Overmature Ripe recall reported explicitly;
- the p0 5-fold score;
- per-camera results for V-R and every fusion that includes RGB.

**Uncertainty.** As in rounds 1–4:
- fruit-cluster bootstrap (2,000 resamples, seed 42);
- fruit-level permutation (20,000, seed 42);
- clustered McNemar test for accuracy.

**Diagnostics (descriptive).**
- **Baseline-only.** A1b with a bootstrap CI.
- **Batch decodability of each frozen embedding.** Logistic regression on intake recordings/images only (all Unripe at harvest), predicting harvest batch with fruit-grouped 5-fold CV and reporting balanced accuracy. Batch is not the outcome; this measures how strongly each encoder encodes batch identity before ripening.
- **Cost.** GPU and CPU time per encoder for extraction and fine-tuning.

## 5. Primary contrasts, decision rule and multiplicity

Holm adjustment across two contrasts:
- **Q1: A1 (BEATs frozen, current) vs CDR-RO;**
- **Q2: F1 (four-modality foundation fusion) vs CDR-RO.**

Each hypothesis is decided separately. An arm is called **beneficial** only if its contrast shows all three of:
- a pooled LOBO gain of at least 3 pp;
- a bootstrap 95% CI excluding zero;
- a Holm-adjusted p < 0.05.

Otherwise it is reported as not beneficial, whatever its direction.

**Conditions for specific wording** (unadjusted, reported):
- **"transfers better than the task-specific CNN":** A1 − K1 ≥ 3 pp with a CI excluding zero;
- **"within-fruit change carries the signal":** A1c − A1s ≥ 3 pp with a CI excluding zero, and A1c − A1 ≥ 3 pp with a CI excluding zero;
- **"fine-tuning helps":** A1ft − A1 ≥ 3 pp with a CI excluding zero.

**Secondary family (Holm across four, reported):** A1, A2, A3 and A4, each vs CDR-RO.

**Exploratory (all others):**
- A5, A6;
- A1c/A1p/A1s/A1b/A1w;
- A1ft vs CDR-RO;
- F2–F5;
- the per-modality vision arms;
- all p0 contrasts.

## 6. Checks

**Controls.**
- CDR-RO and K1 on the 104 sessions are recomputed from the round-1 and round-4 session probabilities (asserted equal to the round-4 values 0.661 and 0.567).
- The knock count per recording in every audio embedding equals the count in `features_sound_knocks.parquet`.
- Every image of the 104 sessions and of their training sessions has an embedding (asserted).

**Leakage audit.** The audit asserts that no label, storage time, visit interval or weight variable enters any embedding, feature, pairing or normalisation; and that A1ft gradient steps see no held-out unit.

**Information seen before evaluation.** Fitting scripts print only sizes and run times.

**Outputs.**
- Location: `results/round5/`.
- Code: `src/foundation/round5.py` (probes, fusion, evaluation) and `src/foundation/beats_ft.py` (A1ft, run on the GPU host).

## 7. What a result would mean

| Result | Interpretation |
|---|---|
| Q1 or Q2 beneficial | Pretrained encoders transfer across these three batches better than the protocol-information reference. Exploratory; a new harvest is needed to confirm. |
| Neither beneficial | Batch shift on this dataset is not solved by larger pretrained representations. Combined with rounds 1–4, the limit is the data (three batches confounded with maturity), not the model class. |

## 8. Frozen embedding files (md5)

The round uses exactly these files. They were extracted on the GPU host `gpu` with `extract_r5.py`, in the `durian-fm` environment (Python 3.12, torch 2.8.0+cu128). The full description is in `docs/round5/README_embeddings.md`, and the environment, checkpoint shas and file md5s are in `docs/round5/manifest_env_models.json`.

**Extraction facts (label-free, recorded before the tag).**
- **Audio.** All 538 recordings are natively 48 kHz. The re-run onset detector reproduced every onset (max difference 1.9 µs). There were 0 decode failures.
- **Truncated knocks.** 48 knocks are truncated at the file end in W500; none are truncated in W150.
- **Minimum input lengths, observed.** BEATs 175 ms and CNN14 310 ms. The 1.0 s padding is therefore required.
- **BEATs checkpoint.** The file is the self-supervised `BEATs_iter3_plus_AS2M.pt` from the Hugging Face mirror `Bencr/beats-checkpoints@082fb18` (361,499,833 B, md5 `e455a4b81e5d4aa1d257fefb2efb770f`). It has no predictor head and gives 768-d output. It could not be checked against a Microsoft-published hash, which is a disclosed limitation.
- **Primary image variants.**
  - RGB `padsq`: repo decode, long side 512, zero-pad to a square, resized to 224. There are 2,219 M50 images and 499 R50 images.
  - Thermal `padsq_fixed20to40C`: the mean clipped fraction is 0.03 %.
  - Multispectral `band8`: fixed sensor 8-bit range, in the repo tile order B1_720 … B8_980.
- **Analysis images.** Images enter the analysis only through `data/durian_mm_capture_folds.csv` (joined by `zip_member`): RGB 1,999, thermal 1,671 CSV and multispectral 2,222.
- **First-pass extraction bugs, fixed before this plan.** 53 `*_thermal_03.csv` files were missed, and 2 multispectral TIFFs were LZW-compressed. Both modalities were fully re-extracted.
- **Determinism.** Image embeddings are bit-reproducible. Audio embeddings reproduce within 3.4e-5.

**Where each step runs.**
- `round5.py fit` and `round5.py diagnose` (CPU) and `beats_ft.py` (GPU) run on the host, where the embeddings are.
- Only `results/round5/probe_sessions.parquet`, `batch_decodability.csv` and the A1ft npy files come back.
- `round5.py evaluate` runs in the repository.

| file (host `~/durian_r5/emb/`) | md5 |
|---|---|
| `audio_ast_W150_pooler.npy` | `058e969a8b1d768ef9230bd024ecf450` |
| `audio_beats_W150_maskedmean.npy` | `d392f841dc4e2cbd51769969dfcff970` |
| `audio_beats_W500_maskedmean.npy` | `db33ae52a4b6e19b495356aa056302da` |
| `audio_clap_W150_repeatpad_projl2.npy` | `9cbc70a8194d428b6c69ea8c24f89a24` |
| `audio_hubert_W150_alllayers_mean.npy` | `c6c51214970510b73f47f8a464d77fac` |
| `audio_knocks_index.parquet` | `0e1669781803718b455ae7a271463b4a` |
| `audio_panns_W150_embedding.npy` | `1b77b4f1ef89839a6dc0b8d85bcd29a4` |
| `audio_wav2vec2_W150_alllayers_mean.npy` | `f10d78d4139add2f31a6f1ccb1060470` |
| `multispectral_clip_band8_index.parquet` | `a8bdaaecb87535b9b51af17508e33dc6` |
| `multispectral_clip_band8_proj.npy` | `2fa85acbe90779327a0d32ec2eb7cff7` |
| `multispectral_dinov2_band8_index.parquet` | `11b4503f0940bc6ff54e386aacaa984f` |
| `multispectral_dinov2_band8_pooler.npy` | `196f518f8e2659567e2210362cec3a6d` |
| `multispectral_siglip_band8_index.parquet` | `3f4e10806fe0879a40a1013dd771dcdf` |
| `multispectral_siglip_band8_pooler.npy` | `153e9cf5b579350a856cebae2e0cb80f` |
| `rgb_clip_padsq_index.parquet` | `ca3dce4546085651719145943912d4c5` |
| `rgb_clip_padsq_proj.npy` | `a724cf583e6ad4f015b98518ed1ba434` |
| `rgb_dinov2_padsq_index.parquet` | `43284badd705d13b7f9811f50132887c` |
| `rgb_dinov2_padsq_pooler.npy` | `7db9e72ccb8e3c6fe3b0b88ec1a18ebb` |
| `rgb_siglip_padsq_index.parquet` | `59cce2364287f3a2f8dc8a96178183a7` |
| `rgb_siglip_padsq_pooler.npy` | `5c21975ca3b95e9f289c2dacd186ca99` |
| `thermal_clip_padsq_fixed20to40C_index.parquet` | `358e30a779abbf1725b1848e09627073` |
| `thermal_clip_padsq_fixed20to40C_proj.npy` | `07833ecc16fae977feaccd45428d17f5` |
| `thermal_dinov2_padsq_fixed20to40C_index.parquet` | `31c27d2796bfd2698082cfc4bbc72bca` |
| `thermal_dinov2_padsq_fixed20to40C_pooler.npy` | `30e0ccd4ca4a8db2da6f27ee2c139d86` |
| `thermal_dinov2_padsq_minmax_index.parquet` | `dba41b1b275fc7f87da03f7475a44872` |
| `thermal_dinov2_padsq_minmax_pooler.npy` | `bd053939868a0f0ed64093d1fa0fa8e0` |
| `thermal_siglip_padsq_fixed20to40C_index.parquet` | `135639bd9a6aa900cd4e3e0e57e69f73` |
| `thermal_siglip_padsq_fixed20to40C_pooler.npy` | `2a90cb819d0249e1dd59e6d7fec8ac87` |
