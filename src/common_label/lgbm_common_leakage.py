"""v2.2 (review 2, comment 7): refit the LightGBM comparison models of the leakage and single-sensor tables on the reconstructed
(common) label. Same features, model settings (src/nested_selection/cands.py) and fold assignments (fold_fruit and capture-level
fold_random of the original out-of-fold files) as the shipped rows; only the training label changes. A validation pass rescores
the shipped predictions with the same code and must reproduce results/leakage_table.csv.
usage (repo root): python src/common_label/lgbm_common_leakage.py -> results/common_label/leakage_lgbm_common.csv,
  leakage_table_v22.csv (results/leakage_table.csv with the four LightGBM rows replaced by the refit) and oof_<mod>__<model>__refit.csv
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(0, "src/nested_selection"); import cands
OUT = os.environ.get("COMMON_LABEL_OUT", "results/common_label"); os.makedirs(OUT, exist_ok=True)
OOF = "intermediate_common/oof"
JOBS = [("thermal", "lgbm", "lgbm"), ("multispectral", "lightgbm", "lightgbm"), ("sound", "lgbm", "lgbm"), ("rgb", "hc_lgbm", "hc_lgbm")]
CL = ["p_Unripe", "p_Ripe", "p_Overripe"]
LT = pd.read_csv("results/leakage_table.csv")
def foldstats(y, P, fold):
    a = [accuracy_score(y[fold == k], P[fold == k].argmax(1)) for k in range(5)]
    f = [f1_score(y[fold == k], P[fold == k].argmax(1), average="macro") for k in range(5)]
    return dict(fruit_acc_mean=np.mean(a), fruit_acc_sd=np.std(a, ddof=1), fruit_f1_mean=np.mean(f), fruit_f1_sd=np.std(f, ddof=1))
def score(files, sess, y, P_f, P_r, ff=None):
    out = {}
    for lev in ["capture", "session"]:
        if lev == "capture":
            yy, pf, pr = y, P_f, P_r
        else:
            d = pd.DataFrame(dict(s=sess, y=y)); g = d.groupby("s", sort=True)
            yy = g.y.first().values
            pf = pd.DataFrame(P_f).groupby(sess).mean().loc[g.y.first().index].values
            pr = pd.DataFrame(P_r).groupby(sess).mean().loc[g.y.first().index].values
        a_f, a_r = accuracy_score(yy, pf.argmax(1)), accuracy_score(yy, pr.argmax(1))
        f_f, f_r = f1_score(yy, pf.argmax(1), average="macro"), f1_score(yy, pr.argmax(1), average="macro")
        out[lev] = dict(n=len(yy), fruit_pooled_acc=a_f, random_pooled_acc=a_r, fruit_pooled_f1=f_f, random_pooled_f1=f_r,
                        delta_acc_pp=100 * (a_r - a_f), delta_f1_pp=100 * (f_r - f_f))
        if lev == "capture" and ff is not None: out[lev].update(foldstats(y, P_f, ff))
    return out
rows = []
for mod, cand, oname in JOBS:
    of = pd.read_csv(f"{OOF}/oof_{mod}__{oname}__fold_fruit.csv"); orr = pd.read_csv(f"{OOF}/oof_{mod}__{oname}__fold_random.csv").set_index("file").loc[of.file].reset_index()
    meta, X = cands.load(mod)
    pos = pd.Series(np.arange(len(meta)), index=meta.file.values); idx = pos.loc[of.file].values      # evaluated population = shipped rows
    y = meta.y.values[idx].astype(int); assert (y == of.y.values).all(), mod                         # shipped y is the common label
    sess = of.session_id.values; ff = of.fold_fruit.values.astype(int); fr = orr.fold_random.values.astype(int)
    # validation: shipped (file-label-trained) predictions rescored with this code reproduce the shipped leakage rows
    ship = score(of.file, sess, y, of[CL].values, orr[CL].values, ff)
    for lev, v in ship.items():
        L = LT[(LT.modality == mod) & (LT.model == oname) & (LT.level == lev)]
        assert len(L) == 1 and abs(L.fruit_pooled_f1.iloc[0] - v["fruit_pooled_f1"]) < 1e-9 and abs(L.random_pooled_f1.iloc[0] - v["random_pooled_f1"]) < 1e-9, (mod, lev)
        if lev == "capture":
            for c in ["fruit_acc_mean", "fruit_acc_sd", "fruit_f1_mean", "fruit_f1_sd"]:
                assert abs(L[c].iloc[0] - v[c]) < 1e-6, (mod, c, L[c].iloc[0], v[c])
    Xs = {k: v[idx] for k, v in X.items()}; m2 = meta.iloc[idx].reset_index(drop=True)
    P = {}
    for arm, fold in [("fruit", ff), ("random", fr)]:
        p = np.full((len(idx), 3), np.nan)
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]; p[te] = cands.fit_predict(mod, cand, m2, Xs, tr, te)
        P[arm] = p
    new = score(of.file, sess, y, P["fruit"], P["random"], ff)
    pd.DataFrame({"file": of.file, "session_id": sess, "y": y, "fold_fruit": ff, "fold_random": fr,
                  **{f"{c}_fruit": P["fruit"][:, i] for i, c in enumerate(CL)}, **{f"{c}_random": P["random"][:, i] for i, c in enumerate(CL)}}
                 ).to_csv(f"{OUT}/oof_{mod}__{oname}__refit.csv", index=False)
    for lev in ["capture", "session"]:
        rows.append(dict(modality=mod, model=oname, level=lev, training_label="file-name (shipped)", **ship[lev]))
        rows.append(dict(modality=mod, model=oname, level=lev, training_label="reconstructed (refit)", **new[lev]))
    print(mod, "done", flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/leakage_lgbm_common.csv", index=False)
L2 = LT.copy(); L2["training_label"] = "reconstructed"
for _, r in D[D.training_label.str.startswith("reconstructed")].iterrows():
    k = (L2.modality == r.modality) & (L2.model == r.model) & (L2.level == r.level); assert k.sum() == 1
    for c in [c for c in L2.columns if c in D.columns and c not in ("modality", "model", "level", "training_label")]:
        L2.loc[k, c] = r[c]
    for c in ["random_acc_mean", "random_acc_sd", "random_f1_mean", "random_f1_sd"]:
        if c in L2.columns: L2.loc[k, c] = np.nan
    if r.level == "session":
        for c in ["fruit_acc_mean", "fruit_acc_sd", "fruit_f1_mean", "fruit_f1_sd"]:
            if c in L2.columns: L2.loc[k, c] = np.nan
    L2.loc[k, "training_label"] = "reconstructed (refit v2.2)"
L2.loc[L2.model.str.contains("gradient", na=False), "training_label"] = "file-name (LightGBM members, earlier analysis)"
L2.to_csv(f"{OUT}/leakage_table_v22.csv", index=False)
print(D.round(3).to_string(index=False))
