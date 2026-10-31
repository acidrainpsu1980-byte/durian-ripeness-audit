"""v1.3: refit the thermal SVM-RBF, multispectral LR and RGB EfficientNet-B0 probe with ONE common session label
(data/session_labels_common.csv) for every capture: 5 outer fruit-grouped folds (OOF csv in the track format) and the
10 fold-pair refits used by nested stacking.  Procedure otherwise identical to base_models.py."""
import os, sys, itertools, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "durian_ripeness_repo")
os.environ.setdefault("DURIAN_INTERMEDIATE", os.path.join(REPO, "intermediate"))
sys.path.insert(0, os.path.join(REPO, "src", "nested"))
import base_models as bm
bm.N_JOBS = int(os.environ.get("SK_JOBS", "1"))
CI = {"Unripe": 0, "Ripe": 1, "Overripe": 2}; CL = ["Unripe", "Ripe", "Overripe"]
SL = pd.read_csv(os.path.join(REPO, "data", "session_labels_common.csv")).set_index("session_id")
OUT = os.path.join(HERE, "handoff_common"); os.makedirs(OUT, exist_ok=True)
MODEL = {"thermal": "svm_rbf", "multispectral": "logreg", "rgb": "efficientnetb0_probe"}
for m in sys.argv[1:] or ["thermal", "multispectral", "rgb"]:
    feat, _, fit_predict = bm.LOADERS[m]()
    lab = feat.session_id.astype(str).map(SL.label_common)
    assert lab.notna().all()
    n_changed = int((lab.map(CI).values != feat.y.values).sum())
    feat["y"] = lab.map(CI).values; feat["label_capture"] = lab.values      # closures read feat.y at call time
    fold = feat.fold_fruit.values.astype(int)
    t = time.time(); P = np.full((len(feat), 3), np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k); P[te] = fit_predict(tr, te)
    oof = pd.read_csv(os.path.join(REPO, "intermediate", "oof", f"oof_{m}__{MODEL[m]}__fold_fruit.csv"))
    pos = pd.Series(np.arange(len(feat)), index=feat.file.values)
    idx = pos.loc[oof.file].values
    for i, c in enumerate(CL): oof[f"p_{c}"] = P[idx, i]
    oof["label_capture"] = lab.values[idx]; oof["y"] = feat.y.values[idx]
    oof.to_csv(os.path.join(OUT, f"oof_{m}__{MODEL[m]}__fold_fruit.csv"), index=False)
    acc = (P.argmax(1) == feat.y.values).mean()
    print(f"{m}: {n_changed} capture labels changed; outer OOF capture acc {acc:.3f} ({time.time()-t:.0f}s)", flush=True)
    for a, b in itertools.combinations(range(5), 2):
        tr, te = np.flatnonzero(~np.isin(fold, [a, b])), np.flatnonzero(np.isin(fold, [a, b]))
        Q = np.full((len(feat), 3), np.nan, np.float32); Q[te] = fit_predict(tr, te)
        np.save(os.path.join(OUT, f"{m}_pair_{a}{b}.npy"), Q)
    print(f"{m}: pairs done ({time.time()-t:.0f}s)", flush=True)
print("DONE", flush=True)
