"""v1.3: assemble intermediate_common/ for the fusion pipeline under ONE common session label.
- fruit-grouped OOF: thermal/multispectral/RGB from sk_common.py refits; sound CNN from cnn_common.py outer fits
  (knock -> recording mean, verified recordings only).
- random-split OOF, other leakage-model OOFs and feature parquets: label columns replaced by the common label
  (these models were trained with per-modality file labels, which differ from the common label for 18/1671 thermal frames (1.1%),
  22/2222 multispectral (1.0%), 18/1999 RGB (0.9%) and 16/432 verified sound recordings (3.7%); 9/268 four-sensor sessions (3.4%) are affected).
- sound LOBO probabilities from cnn_common.py lobo fits.
Writes ../durian_ripeness_repo/intermediate_common/ and src/fusion_config_common.json."""
import os, json, shutil, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "..", "durian_ripeness_repo")
IC = os.path.join(REPO, "intermediate_common"); os.makedirs(f"{IC}/oof", exist_ok=True); os.makedirs(f"{IC}/features", exist_ok=True); os.makedirs(f"{IC}/sound", exist_ok=True)
CI = {"Unripe": 0, "Ripe": 1, "Overripe": 2}; CL = ["Unripe", "Ripe", "Overripe"]
SL = pd.read_csv(f"{REPO}/data/session_labels_common.csv").set_index("session_id").label_common
EXCL = set(open(f"{REPO}/data/sound_exclude_wrong_session.txt").read().split())

REC = pd.read_csv(f"{REPO}/data/durian_mm_sound_recordings.csv")
REC_SID = (REC.fruit_id.astype(float).astype(int).astype(str) + "_" + REC.date_captured.astype(str)).values
FILE2SID = pd.Series(REC_SID, index=REC.file.values)

def relabel(df):
    if "file" in df and df.file.astype(str).str.endswith((".m4a", ".wav")).all():
        sid = df.file.map(FILE2SID)          # sound: matched session dates, never the frozen dates in the feature table
    elif "session_id" in df:
        sid = df.session_id.astype(str)
    else:
        sid = df.fruit_id.astype(float).astype(int).astype(str) + "_" + df.date_captured.astype(str)
    lab = sid.map(SL); assert lab.notna().all(), int(lab.isna().sum())
    df = df.copy(); df["label_capture"] = lab.values; df["y"] = lab.map(CI).values
    return df

# 1. fruit-grouped OOF of the selected image models
REFIT = set()
for f in glob.glob(f"{HERE}/handoff_common/oof_*.csv"):          # fruit-grouped AND random-split refits of the selected models
    shutil.copy(f, f"{IC}/oof/{os.path.basename(f)}"); REFIT.add(os.path.basename(f))
# 2. sound CNN fruit-grouped OOF from the common-label outer fits
knock = pd.read_parquet(f"{HERE}/features_sound_knocks.parquet", columns=["file"])
P = np.full((len(knock), 3), np.nan)
for k in range(5):
    a = np.load(f"{HERE}/handoff/cnn_common_outer_{k}.npy"); m = ~np.isnan(a[:, 0]); assert np.isnan(P[m, 0]).all(); P[m] = a[m]
assert not np.isnan(P).any()
t = pd.DataFrame(P, columns=CL); t["file"] = knock.file.values; rp = t.groupby("file")[CL].mean()
o = pd.read_csv(f"{REPO}/intermediate/oof_clean/oof_sound__cnn__fold_fruit.csv")
assert not o.file.isin(EXCL).any()
for c in CL: o[f"p_{c}"] = rp.loc[o.file, c].values
relabel(o).to_csv(f"{IC}/oof/oof_sound__cnn__fold_fruit.csv", index=False); REFIT.add("oof_sound__cnn__fold_fruit.csv")
# 2b. sound CNN random-split OOF from the common-label random-split fits
P = np.full((len(knock), 3), np.nan)
for k in range(5):
    a = np.load(f"{HERE}/handoff/cnn_common_rand_{k}.npy"); m = ~np.isnan(a[:, 0]); assert np.isnan(P[m, 0]).all(); P[m] = a[m]
assert not np.isnan(P).any()
t = pd.DataFrame(P, columns=CL); t["file"] = knock.file.values; rp = t.groupby("file")[CL].mean()
o = pd.read_csv(f"{REPO}/intermediate/oof_clean/oof_sound__cnn__fold_random.csv")
for c in CL: o[f"p_{c}"] = rp.loc[o.file, c].values
relabel(o).to_csv(f"{IC}/oof/oof_sound__cnn__fold_random.csv", index=False); REFIT.add("oof_sound__cnn__fold_random.csv")
# 3. every other OOF used by the pipeline (random splits, leakage models, ablations): labels replaced only
srcs = glob.glob(f"{REPO}/intermediate/oof/*.csv") + glob.glob(f"{REPO}/intermediate/oof_clean/*.csv")
for f in srcs:
    b = os.path.basename(f)
    if b in REFIT:
        continue
    d = pd.read_csv(f)
    if "sound" in b: d = d[~d.file.isin(EXCL)]
    relabel(d).to_csv(f"{IC}/oof/{b}", index=False)
# 4. feature parquets (labels used by LOBO refits and early fusion)
for f in glob.glob(f"{REPO}/intermediate/features/*.parquet"):
    d = pd.read_parquet(f)
    if {"label_capture", "y"} <= set(d.columns) and "fruit_id" in d.columns:
        if "sound" in os.path.basename(f): d = d[~d.file.isin(EXCL)]
        d = relabel(d)
    d.to_parquet(f"{IC}/features/{os.path.basename(f)}", index=False)
# 5. sound LOBO recording probabilities
lobo = {}
for bname in ["Immature", "Mature", "Overmature"]:
    a = np.load(f"{HERE}/handoff/cnn_common_lobo_{bname}.npy"); m = ~np.isnan(a[:, 0])
    t = pd.DataFrame(a[m], columns=CL); t["file"] = knock.file.values[m]
    t = t[~t.file.isin(EXCL)]
    lobo[bname] = t.groupby("file")[CL].mean().reset_index().to_dict(orient="list")
json.dump(lobo, open(f"{IC}/sound/cnn_lobo_recprobs.json", "w"))
os.makedirs(f"{IC}/rgb", exist_ok=True)
relabel(pd.read_csv(f"{REPO}/intermediate/rgb/rgb512_manifest.csv")).to_csv(f"{IC}/rgb/rgb512_manifest.csv", index=False)
if not os.path.exists(f"{IC}/rgb/emb_efficientnetb0.npy"):
    shutil.copy(f"{REPO}/intermediate/rgb/emb_efficientnetb0.npy", f"{IC}/rgb/emb_efficientnetb0.npy")
cfg = json.load(open(f"{REPO}/src/fusion_config.json")); txt = json.dumps(cfg)
for a, b in [("intermediate/oof_clean/", "intermediate_common/oof/"), ("intermediate/oof/", "intermediate_common/oof/"),
             ("intermediate/features/", "intermediate_common/features/"), ("intermediate/rgb/", "intermediate_common/rgb/"),
             ("intermediate/sound_clean/cnn_lobo_recprobs.json", "intermediate_common/sound/cnn_lobo_recprobs.json"),
             ("intermediate/nested_clean/", "intermediate_common/nested/")]:
    txt = txt.replace(a, b)
cfg = json.loads(txt); cfg["out_dir"] = "results_common"; cfg["session_labels"] = "data/session_labels_common.csv"
json.dump(cfg, open(f"{REPO}/src/fusion_config_common.json", "w"), indent=1)
print("built", IC)
