"""v1.3: capture-level random-split refits of the three selected image models with the common session label, so that the
leakage comparison (fruit-grouped vs random) uses the same training labels in both arms."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "..", "durian_ripeness_repo")
os.environ.setdefault("DURIAN_INTERMEDIATE", os.path.join(REPO, "intermediate"))
sys.path.insert(0, os.path.join(REPO, "src", "nested")); import base_models as bm
bm.N_JOBS = int(os.environ.get("SK_JOBS", "2"))
CI = {"Unripe": 0, "Ripe": 1, "Overripe": 2}; CL = ["Unripe", "Ripe", "Overripe"]
SL = pd.read_csv(os.path.join(REPO, "data", "session_labels_common.csv")).set_index("session_id").label_common
MODEL = {"thermal": "svm_rbf", "multispectral": "logreg", "rgb": "efficientnetb0_probe"}
OUT = os.path.join(HERE, "handoff_common")
for m in ["thermal", "multispectral", "rgb"]:
    feat, _, fit_predict = bm.LOADERS[m]()
    lab = feat.session_id.astype(str).map(SL); assert lab.notna().all()
    feat["y"] = lab.map(CI).values
    orig = pd.read_csv(os.path.join(REPO, "intermediate", "oof", f"oof_{m}__{MODEL[m]}__fold_random.csv"))
    fr = pd.Series(orig.fold_random.values, index=orig.file.values).loc[feat.file].values.astype(int)
    P = np.full((len(feat), 3), np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(fr != k), np.flatnonzero(fr == k); P[te] = fit_predict(tr, te)
    pos = pd.Series(np.arange(len(feat)), index=feat.file.values); idx = pos.loc[orig.file].values
    for i, c in enumerate(CL): orig[f"p_{c}"] = P[idx, i]
    orig["label_capture"] = lab.values[idx]; orig["y"] = feat.y.values[idx]
    orig.to_csv(os.path.join(OUT, f"oof_{m}__{MODEL[m]}__fold_random.csv"), index=False)
    print(m, "random-split capture acc", round(float((P.argmax(1) == feat.y.values).mean()), 3), flush=True)
print("DONE")
