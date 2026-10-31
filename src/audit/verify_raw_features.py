"""v1.8: re-extract every feature table from the raw Zenodo archives in a clean work directory and compare it with the shipped tables.

usage (repo root):  DURIAN_DATA_ROOT=/path/to/zips python src/audit/verify_raw_features.py WORKDIR [thermal multispectral sound rgb rgb_embed]
                    (default: all five stages; about 25 min on a 10-core laptop, of which ~20 min are the two CNN embeddings)
Runs, each in WORKDIR/<modality>/:
  thermal        src/thermal/run_extract.py                     -> thermal_feat_raw.parquet
  multispectral  src/multispectral/extract_all.py               -> features_multispectral.parquet
  sound          src/sound/sound_pipeline.py features           -> features_sound_recordings.parquet, features_sound_knocks.parquet,
                 src/common_label/prepare_sound_cnn_inputs.py      handoff/mels.npy, handoff/file_label_common.csv
  rgb            src/rgb/build_cache.py, src/rgb/extract_features.py -> cache/rgb512.npy, cache/rgb_masks128.npy, features_rgb.parquet
  rgb_embed      src/rgb/rgb_models.py embed                    -> cache/emb_mobilenetv2.npy, cache/emb_efficientnetb0.npy
and writes audit/raw_verification.csv: per table, rows matched, feature columns compared, max |diff| over feature columns, and
the number of rows whose label differs (the shipped tables carry the v1.3 common session label; the extraction carries the
file-name label, so these counts must equal the documented relabelling: 22 multispectral, 18 RGB, 16 sound recordings).
"""
import os, sys, subprocess
import numpy as np, pandas as pd
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WORK = os.path.abspath(sys.argv[1]); STAGES = sys.argv[2:] or ["thermal", "multispectral", "sound", "rgb", "rgb_embed"]
S = lambda p: os.path.join(REPO, "src", p); IC = os.path.join(REPO, "intermediate_common")
env = dict(os.environ); env.setdefault("OMP_NUM_THREADS", "4")
CI = {"Unripe": 0, "Ripe": 1, "Overripe": 2}
lab = pd.read_csv(f"{REPO}/data/session_labels_common.csv").set_index("session_id").label_common
man = pd.read_csv(f"{REPO}/data/sound_recording_manifest.csv").set_index("file")

def run(cwd, *cmd):
    os.makedirs(cwd, exist_ok=True)
    print("[raw]", cwd, " ".join(os.path.basename(c) for c in cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=cwd, env=env, check=True)

def compare(name, new, old, key):
    a = new.set_index(key); b = old.set_index(key); idx = b.index.intersection(a.index)
    feat = [c for c in a.columns if c in b.columns and c != "y" and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    d = 0.0; nan_mis = 0
    for c in feat:
        x = a.loc[idx, c].astype(float).values; z = b.loc[idx, c].astype(float).values
        nan_mis += int((np.isnan(x) != np.isnan(z)).sum())
        if np.isfinite(x).any(): d = max(d, float(np.nanmax(np.abs(x - z))))
    yd = int((a.loc[idx, "y"].values != b.loc[idx, "y"].values).sum()) if "y" in a and "y" in b else None
    return dict(table=name, rows_reextracted=len(a), rows_shipped=len(b), rows_matched=len(idx), feature_columns=len(feat),
                max_abs_diff=d, nan_mismatch=nan_mis, label_rows_differing=yd)

rows = []
if "thermal" in STAGES:
    w = f"{WORK}/thermal"; run(w, S("thermal/run_extract.py"))
    rows.append(compare("features_thermal", pd.read_parquet(f"{w}/thermal_feat_raw.parquet"), pd.read_parquet(f"{IC}/features/features_thermal.parquet"), "zip_member"))
if "multispectral" in STAGES:
    w = f"{WORK}/multispectral"; run(w, S("multispectral/extract_all.py"))
    rows.append(compare("features_multispectral", pd.read_parquet(f"{w}/features_multispectral.parquet"), pd.read_parquet(f"{IC}/features/features_multispectral.parquet"), "zip_member"))
if "sound" in STAGES:
    w = f"{WORK}/sound"; run(w, S("sound/sound_pipeline.py"), "features"); run(w, S("common_label/prepare_sound_cnn_inputs.py"))
    rows.append(compare("features_sound_recordings", pd.read_parquet(f"{w}/features_sound_recordings.parquet"), pd.read_parquet(f"{IC}/features/features_sound_recordings.parquet"), "file"))
    kn = pd.read_parquet(f"{w}/features_sound_knocks.parquet"); ko = pd.read_parquet(f"{REPO}/intermediate_v14/features_sound_knocks.parquet")
    for d_ in (kn, ko): d_["knock_key"] = d_.file + "#" + d_.knock_idx.astype(str)
    rows.append(compare("features_sound_knocks", kn, ko, "knock_key"))
if "rgb" in STAGES:
    w = f"{WORK}/rgb"; run(w, S("rgb/build_cache.py")); run(w, S("rgb/extract_features.py"))
    rows.append(compare("features_rgb", pd.read_parquet(f"{w}/features_rgb.parquet"), pd.read_parquet(f"{IC}/features/features_rgb.parquet"), "file"))
if "rgb_embed" in STAGES:
    w = f"{WORK}/rgb"; run(w, S("rgb/rgb_models.py"), "embed")
    for net in ["mobilenetv2", "efficientnetb0"]:
        a = np.load(f"{w}/cache/emb_{net}.npy"); b = np.load(f"{IC}/rgb/emb_{net}.npy")
        rows.append(dict(table=f"emb_{net}", rows_reextracted=a.shape[0], rows_shipped=b.shape[0], rows_matched=min(len(a), len(b)),
                         feature_columns=int(np.prod(a.shape[1:])), max_abs_diff=float(np.abs(a.astype(np.float64) - b).max()),
                         nan_mismatch=0, label_rows_differing=None))
out = pd.DataFrame(rows)
f = f"{REPO}/audit/raw_verification.csv"
if os.path.exists(f):                                   # keep rows of stages not re-run this time
    old = pd.read_csv(f); out = pd.concat([old[~old.table.isin(out.table)], out], ignore_index=True)
out.to_csv(f, index=False); print(out.to_string())
