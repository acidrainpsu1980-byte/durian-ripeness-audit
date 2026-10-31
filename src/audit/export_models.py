"""v1.8: fitted models of the four development pipelines on the primary partition (p0), one per outer fold.

usage (repo root):
  python src/audit/export_models.py fit       refit thermal SVM-RBF, multispectral logistic regression and the RGB EfficientNet-B0
                                              probe on each outer training set of p0 and save them to models/primary_p0/
                                              (knock-CNN weights are written by cnn_common.py with CNN_SAVE_WEIGHTS; see REPRODUCE.md)
  python src/audit/export_models.py check     reload every saved model (including the five knock CNNs), predict the held-out fold
                                              and compare with the shipped outer-fold predictions (intermediate_v14/fits/p0/*.npz,
                                              intermediate_v14/cnn_cpu_p0/*.npy); prints max |diff| per model
Inputs: intermediate_common/ feature tables, intermediate_v14/features_sound_knocks.parquet and the log-mel array
(work/sound/handoff/mels.npy from `sound_pipeline.py features`, path in DURIAN_MELS) for the CNN check.
"""
import os, sys, json, hashlib
import numpy as np, pandas as pd, joblib
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src", "nested_selection"))
os.environ.setdefault("DURIAN_INTERMEDIATE", "intermediate_common")
import cands
from joblib import parallel_backend
OUT = "models/primary_p0"; os.makedirs(OUT, exist_ok=True)
SEL = cands.SELECTED_V13                     # thermal svm_rbf, multispectral logreg, sound cnn, rgb efficientnetb0_probe
part = pd.read_csv("src/nested_selection/partitions.csv").set_index("fruit_id")["p0"]
mode = sys.argv[1] if len(sys.argv) > 1 else "check"
report = []
for mod in ["thermal", "multispectral", "rgb"]:
    cand = SEL[mod]; meta, X = cands.load(mod); fold = meta.fruit_id.map(part).values.astype(int)
    ref = np.load(f"intermediate_v14/fits/p0/{mod}__{cand}.npz", allow_pickle=True)["outer"]
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        fn = f"{OUT}/{mod}__{cand}__outer{k}.joblib"
        xs, est = cands._FACT[mod](cand); Xm = X[xs]; y = meta.y.values; g = meta.fruit_id.values
        if mode == "fit":
            with parallel_backend("threading", n_jobs=cands.NJ):
                est.fit(Xm[tr], y[tr], groups=g[tr]) if hasattr(est, "param_grid") else est.fit(Xm[tr], y[tr])
            joblib.dump({"model": est, "sensor": mod, "candidate": cand, "feature_set": xs, "partition": "p0", "outer_fold": k,
                         "train_files": meta.file.values[tr].astype(str), "classes": ["Unripe", "Ripe", "Overripe"]}, fn, compress=3)
        obj = joblib.load(fn)
        p = obj["model"].predict_proba(Xm[te])
        best = getattr(obj["model"], "best_params_", {})
        report.append(dict(model=os.path.basename(fn), n_train=len(tr), n_test=len(te), best_params=json.dumps(best),
                           max_abs_diff_vs_shipped=float(np.abs(p - ref[te]).max())))
# ---- knock CNN (weights from cnn_common.py, CNN_SAVE_WEIGHTS=models/primary_p0)
mels = os.environ.get("DURIAN_MELS")
if mels and os.path.exists(mels):
    import torch, torch.nn as nn
    sys.argv = sys.argv[:1]
    class Net(nn.Module):                        # identical to src/common_label/cnn_common.py
        def __init__(s, nc=3):
            super().__init__(); ch = [1, 16, 32, 64, 96]
            s.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(),
                                                    nn.Conv2d(ch[i+1], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(),
                                                    nn.MaxPool2d(2)) for i in range(4)])
            s.drop = nn.Dropout(0.3); s.fc = nn.Linear(ch[-1], nc)
        def forward(s, x):
            for b in s.blocks: x = b(x)
            return s.fc(s.drop(x.mean((2, 3))))
    torch.set_num_threads(int(os.environ.get("CNN_THREADS", "3")))
    MELS = np.load(mels)
    for k in range(5):
        fn = f"{OUT}/cnn_common_outer_{k}.pt"
        if not os.path.exists(fn):
            continue
        ck = torch.load(fn); net = Net(); net.load_state_dict(ck["state_dict"]); net.eval()
        ref = np.load(f"intermediate_v14/cnn_cpu_p0/cnn_common_outer_{k}.npy"); te = np.flatnonzero(~np.isnan(ref[:, 0]))
        X = torch.tensor((MELS[te] - ck["mel_mean"]) / ck["mel_std"]).unsqueeze(1)
        with torch.no_grad():
            p = np.concatenate([torch.softmax(net(X[i:i+512]), 1).numpy() for i in range(0, len(te), 512)])
        report.append(dict(model=os.path.basename(fn), n_train="", n_test=len(te), best_params="",
                           max_abs_diff_vs_shipped=float(np.abs(p - ref[te]).max())))
R = pd.DataFrame(report)
for f in sorted(os.listdir(OUT)):
    if f.endswith((".joblib", ".pt")):
        R.loc[R.model == f, "sha256"] = hashlib.sha256(open(f"{OUT}/{f}", "rb").read()).hexdigest()
CHK = os.environ.get("DURIAN_AUDIT_OUT", OUT); os.makedirs(CHK, exist_ok=True)
R.to_csv(f"{CHK}/model_check.csv", index=False)
print(R.drop(columns=["sha256"], errors="ignore").to_string())
