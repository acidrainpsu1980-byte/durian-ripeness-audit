# Round-5 label-free frozen embeddings (durian ripeness, Zenodo 10.5281/zenodo.18603796)

Pre-registration-phase product. **No label column was read into any model, and no classifier, accuracy, F1 or other score was computed.** The only metadata used are file keys (zip member / file name) and, for aggregation only, the session keys `fruit_id` + `date_captured` from the repository manifests.

Extracted on 2026-11-02 on the ssh host `gpu` (RTX 3090 Ti, driver 595.99.02). The GPU was shared, and the jobs used at most 2.9 GB of GPU memory against a 7 GB budget (`torch.cuda.set_per_process_memory_fraction`). Code: `~/durian_r5/extract_r5.py` (a copy is in `emb/extract_r5.py`). Driver scripts: `run_r5.sh`, `run_audio_sidecar.sh`, `run_thermal_fix.sh`, `run_ms_fix.sh`.

## Location (embeddings stay on the host)

```
~/durian_r5/
  zips/            dataset_clean_{sound,thermal,multispectral,rgb}.zip, README.md, durian_characteristics_cleaned.csv,
                   checksums_expected.md5 (all 6 files md5 OK; see md5_verify.log)
  repo/durian_ripeness_repo/   repo kit (src/, data/, work/raw/sound/features_sound_knocks.parquet ...)
  models/          BEATs_iter3_plus_AS2M.pt, Cnn14_mAP=0.431.pth, beats_code/ (unilm BEATs.py, backbone.py, modules.py), model_sources.json
  hf/              Hugging Face cache (HF_HOME) incl. hf/open_clip/
  emb/             FINAL embeddings, float32 throughout  <-- use this folder
    aggregated/    recording- and session-level means
  emb_audio/       audio-only run that produced emb/audio_* (byte-identical copies)
  emb_pack/        transfer copy of emb/ (only *_patchmean*.npy downcast to float16); r5_embeddings_host.tgz = tar.gz of emb_pack
  emb_dry/         dry run (3 sound files, 4 images per modality) - not for analysis
  logs/            smoke_report.json, manifest_env_models.json, md5_verify.log, pip_freeze.txt, conda_explicit.txt, download logs
```

Environment: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate durian-fm`, plus `export HF_HOME=~/durian_r5/hf DURIAN_DATA_ROOT=~/durian_r5/zips NUMBA_CACHE_DIR=~/durian_r5/numba_cache`.

## File naming

`emb/{modality}_{encoder}_{window|variant}_{feature}.npy` (rows aligned with) `emb/{modality}_{encoder}_{window|variant}_index.parquet`.
All features of one encoder × window/variant share one index. `emb/audio_knocks_index.parquet` and `emb/{rgb,thermal,multispectral}_images_index.parquet` are the shared master indexes.
`emb/aggregated/{prefix}_{feature}_{recording|session}.npy` with `{prefix}_{level}_index.parquet` (`key`, `n_items`); row order = sorted keys.

## Audio: 15,709 knocks, 538 files

* **Knock list.** These are all rows of `work/raw/sound/features_sound_knocks.parquet` (`file`, `knock_idx`, `onset_s`), in parquet file order. The per-file knock count equals the parquet count for every file (asserted). The paper's own onset detector, re-run on this platform, reproduced every file's count and every onset (maximum difference 1.9 µs). There were 0 decode failures.
* **Decoding.** Decoding is identical to `src/sound/knock_features.py::decode_bytes` (wav via libsndfile, m4a via ffmpeg f32le, mono mean). All 538 recordings are natively 48 kHz. No high-pass filter and no peak normalisation are applied, so raw decoded amplitude is kept. `peak_abs_*` and `file_peak_abs` (in `audio_file_decode_log.csv`) are recorded.
* **Resampling.** Each recording is resampled ONCE from its native rate to the encoder rate (16 / 32 / 48 kHz) with `librosa.resample(res_type="soxr_hq")`, with no detour through 22.05 kHz. The window is then cut at `round(onset_s * sr)`.
  * W150 = 150 ms, the paper's CNN window (primary).
  * W500 = 500 ms, the exploratory sensitivity window. It is truncated at the file end and zero-filled; 48 knocks are affected, and none are affected for W150. `nvalid_{W}_{sr}` holds the real sample count.

| encoder (checkpoint) | sr | encoder input | features (dim) - **primary in bold** |
|---|---|---|---|
| BEATs iter3+ AS2M, **self-supervised** (not fine-tuned) | 16k | zero-pad to 1.0 s, `padding_mask=True` on padded samples | **`maskedmean`** (768) = mean of last-layer tokens with returned padding_mask False; `meanall` (768) |
| PANNs CNN14 (Zenodo 3987831) | 32k | zero-pad to 1.0 s | **`embedding`** (2048) built-in fc1(max_t+mean_t); equals library forward (max diff 0) |
| AST `MIT/ast-finetuned-audioset-10-10-0.4593` | 16k | feature-extractor default (log-mel zero-padded to 1024 frames = 10.24 s) | **`pooler`** (768) = mean(CLS, distill); `cls`; `patchmean` (all 12×101 patches); `patchmean_signal` (patch columns starting inside the real window: 2 for W150, 5 for W500) |
| LAION CLAP `laion/clap-htsat-unfused` | 48k | feature extractor to 10 s; `repeatpad` = model default (primary), `pad` = zeros (sensitivity); truncation rand_trunc never triggers | **`repeatpad_projl2`** (512, L2-normalised get_audio_features); `repeatpad_pooler` (768, pre-projection); `zeropad_projl2` (512) |
| wav2vec2-base `facebook/wav2vec2-base` | 16k | none (do_normalize=True from its preprocessor) | **`alllayers_mean`** (768) = mean over the 12 transformer layers of each layer's frame-mean; `last_mean`; `mid6_mean` (hidden_states[6]) |
| HuBERT-base `facebook/hubert-base-ls960` | 16k | none | same three as wav2vec2 |

**Minimum input length, checked by binary search on unpadded inputs.** BEATs needs 2,800 samples (175 ms; 16 fbank frames for one 16×16 patch). CNN14 needs 9,920 samples (310 ms; five 2×2 pools). A 150 ms window therefore fails unpadded in both, and the pre-specified 1.0 s padding is required. Note that BEATs maps the sample-level mask onto tokens in contiguous chunks (`forward_padding_mask … .all(-1)`); it was used as released.

## Images (all image files in the zips, `__MACOSX` excluded); vision encoders at 224×224 with no further resize or centre-crop

Encoders, with primary features in bold:

* DINOv2 `facebook/dinov2-base`: **`pooler`** (CLS after the final layernorm, 768) and `patchmean`. ImageNet mean/std.
* SigLIP `google/siglip-base-patch16-224` (vision tower): **`pooler`** (MAP head, 768) and `patchmean`. Normalisation 0.5/0.5.
* CLIP ViT-B/16 OpenAI via open_clip 3.3.0 `ViT-B-16-quickgelu` (secondary): `proj` (512, un-normalised image embedding) and `patchmean` (768).

Variants by modality:

| modality | files | variants (primary first) |
|---|---|---|
| RGB | 2,718 (Canon EOS M50m2 2,219; EOS R50 499; `camera` column in index) | **`padsq`**: repo `build_cache` decode (EXIF transpose, long side 512, zero pad to a 512 square), then INTER_AREA to 224. `crop`: repo fruit crop (`seg_hsv` mask, 128-px round trip, `fruit_crop` margin 0.12) at 224; 1 image fell back to padsq (`crop_fallback_full`). |
| thermal | 2,174 CSV (396×528 °F → °C; includes 53 `*_thermal_03.csv`) | **`padsq_fixed20to40C`**: full frame clipped to a FIXED 20–40 °C window, scaled to 0–255 uint8, pad to 528² with 0, resize 224, grey ×3, no colormap. `padsq_minmax`: per-image min–max (sensitivity). `crop_fixed20to40C` / `crop_minmax`: repo `segment` + `crop_image` geometry (1.3× bbox, edge-padded) at 224, no z-score; segmentation fell back to the centred disc for 44 images (`seg_fallback`). Clipping (label-free): 0 pixels below 20 °C; mean fraction above 40 °C 0.00029 (full frame) / 0.00012 (crop); 1 image has >1 % clipped. |
| multispectral | 2,348 TIFF (uint8 512×1024 montage) | **`band8`**: each of the 8 tiles (256²) embedded separately. The fixed per-band range is the sensor's 8-bit range (DN/255; observed max 255), resized to 224, grey ×3. Stored as (N, 8, D) in the repo's tile order B1_720 … B8_980 nm (ascending wavelength order is the repo's datasheet assumption; the files carry no band tags). |

**Join to `data/durian_mm_capture_folds.csv` by `zip_member`.** Every fold-table image row is present: RGB 1,999/1,999, thermal 1,671/1,671 CSV rows, multispectral 2,222/2,222. The thermal fold table's other 1,671 rows are `.jpg` previews, which were not embedded. The extra zip files are not in the fold table (`in_capture_folds = 0`): RGB 719, thermal 503, multispectral 126. These are the fruits outside the analysis set and tree-code-unlinked files listed in `raw_file_inventory.csv`. They were embedded and flagged, and they map to `UNMATCHED` in the session aggregates. No image failed in the final run.

## Aggregates (`emb/aggregated/`)

* Audio, recording level: the mean over the knocks of each file (538).
* Audio, session level: the mean over all knocks of the session's recordings (269 sessions; `session_id = fruit_id_date` from `data/durian_mm_sound_recordings.csv`). No exclusion list was applied.
* Images, session level: the mean over the images of a session, per modality and variant. There are 313 RGB, 276 thermal and 316 multispectral groups, each including one `UNMATCHED` group.

## Determinism

Settings were fp32, `eval()`, `no_grad`, cuDNN deterministic, TF32 off, `use_deterministic_algorithms(warn_only)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and eager attention. Re-running the first 100 knocks (64 images) with the same batches gave:

* images: bit-identical (max abs diff 0);
* audio: not bit-identical, max abs diff ≤ 3.4e-5 (wav2vec2 W150 `last_mean`) and ≤ 2.2e-5 for every other encoder. Values per feature are in `audio_report.json`.

## Provenance / caveats

* **BEATs checkpoint.** The official unilm download links (OneDrive/Azure SAS) cannot be scripted, so the file came from the HF mirror `datasets/Bencr/beats-checkpoints@082fb18`: 361,499,833 B, md5 `e455a4b81e5d4aa1d257fefb2efb770f`. A second mirror, `tsinghua-ee/WAVE-7B@7d51cda`, has a file with the same name but different bytes (363,145,291 B, md5 `3e6a432f…`). That size equals the published size of the AudioSet-fine-tuned cpt1 checkpoint, so that file was **not** used. The model we loaded has no predictor head and returns 768-d features, which is consistent with the self-supervised checkpoint. Its md5 could not be checked against a Microsoft-published hash. BEATs code: microsoft/unilm @ `732d834db70ee0fc3886b4bcbcfb4ce7fb829be2`.
* **CNN14.** `Cnn14_mAP=0.431.pth`, md5 `541141fa2ee191a88f24a3219fff024e`, which equals the Zenodo record checksum.
* **Hugging Face models.** Commit shas and per-file md5s for every HF model are in `manifest_env_models.json`.
* **First-pass bugs, fixed and re-extracted.** (a) The thermal filename filter missed 53 `*_thermal_03.csv` files. (b) Two LZW-compressed multispectral TIFFs needed `imagecodecs`. Every `thermal_*` and `multispectral_*` file in `emb/` comes from the corrected rerun (see `thermal_rerun_note` and `multispectral_rerun_note` in `images_report.json`). `images_report_thermal.json` and `images_report_multispectral.json` are the rerun reports.
* **Not attempted.** EAT (optional) was not tried.
* **Large files not transferred.** The 1.7 GB archive was not copied to the local workspace because the host link ran at about 47 KB/s. Use the host copy and the per-file md5 list `emb_md5_list.txt`.
