# Software and hardware

## CPU analysis environment (all results except the ten-partition knock-CNN refits)

Apple-silicon laptop, 10 cores, 32 GB RAM, macOS, CPU only. conda-forge builds; `environment.yml` creates it, and
`requirements_lock_cpu.txt` lists the exact version of every installed package (97 packages).

```
python 3.12.16
platform macOS-27.0.1-arm64-arm-64bit
numpy 2.5.4
pandas 3.1.0
scipy 1.18.1
sklearn 1.9.1
lightgbm 4.7.0
torch 2.14.1
torchvision 0.29.1
librosa 1.0.0
soundfile 0.14.0
numba 0.68.0
cv2 5.0.0
skimage 0.26.0
tifffile 2026.10.15
PIL 13.0.0
pyarrow 25.0.0
joblib 1.6.0
matplotlib 3.11.2
libsndfile 1.2.2
ffmpeg version 9.0.2 Copyright (c) 2000-2026 the FFmpeg developers
```

Threads: `OMP_NUM_THREADS=4` for all scikit-learn / LightGBM fits (≥ 4 is required for bit-identical L-BFGS logistic fits);
`CNN_THREADS=3` for the CPU knock CNN of the primary partition. joblib uses the threading backend (the loky backend is not used).

## GPU environment (knock-CNN refits on the ten fruit partitions and the device-specific CNNs)

```
GPU            NVIDIA GeForce RTX 3090 Ti, 24 GB, driver 595.99.02
OS             Linux 7.0.0-38-generic x86_64, glibc 2.43
Python         3.12.15
PyTorch        2.14.1+cu130 (CUDA 13.0), cuDNN 9.24 (torch.backends.cudnn.deterministic = True, benchmark = False)
NumPy 2.5.4, pandas 3.0.6, pyarrow 25.0.0
```
Scripts: `src/nested_selection/cnn_part.py` (150 fits: 10 partitions x 15), `src/ablations/cnn_dev.py` (150 fits). Their predictions
are shipped (`intermediate_v14/cnn/`, `intermediate_v15/dev/`); every table is reproduced from them by `run_all.sh`.

## Training settings

Every candidate model, fixed hyper-parameter, search grid, inner CV and seed: `model_registry.csv`. Inner search results of every
grid point on the primary partition: `hyperparameter_search_p0.csv`. Configuration space of the selection procedures:
`configuration_space.csv`.
