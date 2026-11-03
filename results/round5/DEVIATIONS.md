# Deviations from docs/ANALYSIS_PLAN_round5.md (tag `prereg-round5`, commit 6810a38)

Written before `src/foundation/round5.py evaluate` was first run.

## Execution

1. **Where the fits ran.** `round5.py fit` and `round5.py diagnose` ran on the GPU host, CPU only, with `CUDA_VISIBLE_DEVICES=""`.
   - Environment: `durian-fm`, with python 3.12.16, scikit-learn 1.9.1, numpy 2.5.4 (scipy-openblas 0.3.34), pandas 3.1.0, scipy 1.18.1 and pyarrow 26.0.0.
   - Inputs: the frozen embeddings in `~/durian_r5/emb/` (md5 in plan Section 8), as planned.
   - Code: unchanged from the tag (`round5.py` md5 `3b29a9b2f52250e991b6beae0bffaacd`).
   - Wall time: `fit` 39.7 s and `diagnose` 86.7 s.
   - Output: `probe_sessions.parquet` md5 `ea115a5089ebe470613e910cb190dd40`, 5,200 rows, 22 models × 2 splits.
2. **Thread count.** `OMP_NUM_THREADS=4` was set. CPU use was 5.5–6.2 cores on average, because numpy's bundled OpenBLAS starts its own threads.
   - Effect: none expected on the fits (lbfgs logistic regression).
   - Not tested: no 1-thread rerun was made in this round.
3. **A1ft run.** `beats_ft.py` ran on the GPU host: 8 jobs × 3 seeds, all 15,709 knocks; see `logs/beats_ft.log`.
   - Software and hardware: torch 2.8.0+cu128, CUDA 12.8, cuDNN 91002, RTX 3090 Ti.
   - Wall time: 2,712 s. Peak GPU memory: about 2.5 GB.
   - Train/test knock counts per job are identical to the round-4 paired-CNN logs.
   - The 8 output npy files match the host md5 list (`logs/beats_ft_md5.txt`).
4. **A1ft crash fix (one line).** The first attempt crashed during decoding, with a multiprocessing `PicklingError` (`logs/beats_ft_attempt1_crash.log`).
   - Cause: the kit's `src/foundation/extract_r5.py`, which is md5-identical to `$R5_EXTRACT`, was imported under the same module name in the worker processes.
   - Fix: `sys.modules["extract_r5"] = X5` was added before the module is executed. No model, data, hyperparameter or seed changed.
   - md5 of `beats_ft.py`: d5c531dc05841ad00d6c7b3e8de37001 as tagged, 99e1112246aa7fbdd17f83fc78a761fe as run. The repository now holds the version that was run.
5. **Shipped files.** The frozen embeddings themselves (1.7 GB) are not shipped in the repository. They are reproducible with `src/foundation/extract_r5.py` from the Zenodo zips, and their md5 sums are listed in `docs/round5/emb_md5_list.txt`.

## Information seen before evaluation

6. **Printed output.** The fitting scripts printed only arm names and run times. The fusion track checked only the row count and the model/split names of `probe_sessions.parquet`, and did not read `batch_decodability.csv`. No probability or score was inspected before evaluation.
8. **Evaluate runtime fix (added after the pre-evaluation commit 6ab67cb, before any score was written).** The first `evaluate` call stopped with `FileNotFoundError`, before any output was written.
   - Cause: the per-camera step loaded the RGB embedding array, which is not present locally, although it needs only the RGB index (camera column).
   - Fix: `image_embeddings(..., index_only=True)` now reads only `rgb_dinov2_padsq_index.parquet`, copied from the shipped index set to `work/r5_embeddings/`.
   - Effect: no model, prediction or contrast changed.
   - Push status: `pre-evaluation-round5` exists as a local commit/tag only. The push was not made because the user did not answer "Yes".

7. **Structural smoke test.** Before the tag, the full pipeline (`fit`, `diagnose`, `evaluate`) was run once locally on random embeddings of the real shapes. Its purpose was to check the code path, the coverage counts and the reference assertions (CDR-RO and K1 reproduce the round-4 values). Its scores are meaningless and were not inspected.
