"""Thermal quantization harmonization (v1.5, reviewer comment 6).
Every radiometric frame is requantized to a common 0.5 degC grid (coarser than the coarsest export step observed,
about 0.22 degC), keeping the ORIGINAL fruit mask, and the 67 features are recomputed on the requantized frame.
usage: python thermal_harmonize.py SLICE NSLICES OUT.parquet"""
import sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "durian_ripeness_repo")
sys.path[0:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "src", "thermal")]
os.chdir(os.path.join(REPO, "src", "thermal"))
import thermal_features as tf
STEP = 0.5
i, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
feat = pd.read_parquet(os.path.join(REPO, "intermediate_common", "features", "features_thermal.parquet"))
members = feat.zip_member.values[i::n]
rows = []
for m in members:
    M = tf.load_matrix(m)
    mask, info = tf.segment(M)
    Mq = (np.round(M / STEP) * STEP).astype(np.float32)
    f = tf.features(Mq, mask, info); f["zip_member"] = m
    f["n_unique_orig"] = int(np.unique(np.round(M, 3)).size)
    rows.append(f)
pd.DataFrame(rows).to_parquet(out)
print("done", i, len(rows))
