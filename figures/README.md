# figures/

The three PNGs here were generated from this bundle with `make figures`; `make figures` (or `python src/fusion_figures.py`, run from the repository root after the fusion
stage) writes the three fusion/leakage/LOBO panels here at 300 dpi:

- `fig_fusion_ablation.png` - session macro-F1 / accuracy per modality subset and fusion rule (5-fold mean, 95 % t-CI)
- `fig_leakage.png`         - fruit-grouped vs capture-level random split per sensor and per fusion
- `fig_lobo.png`            - leave-one-harvest-batch-out per sensor and fusion vs majority baseline

The per-modality figures of the paper (segmentation examples, spectra, confusion matrices, Grad-CAM) were
rendered from the session code in `src/session_traces/` and the paper's figure set is produced separately;
they are not part of `make all`.
