"""Pre-registered analysis (docs/ANALYSIS_PLAN_anchor_physics.md §4, within-batch check): fruit-grouped 5-fold CV on the ten
fruit partitions (src/nested_selection/partitions.csv; p0 = the paper's primary partition = fold_fruit) for the thermal SVM-RBF, the
RGB probe and the physics-guided LR, each with M0 (no standardization) and M1 (per-batch standardization with the batch's anchor
captures; anchor statistics use all anchor captures of each batch, label-free by protocol). Same families, grids and inner
fruit-grouped 3-fold CV as lobo_classical.py. Captures of all 315 sessions of the training fruits are used for training.
Writes OUT/within_batch_<model>.parquet (session probabilities per variant and partition).
usage (repo root): python src/anchor_physics/within_batch.py OUT [model ...]
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import lobo_classical as lc

PC = lc.PC; CL = lc.CL


def run(mod, anchor_sessions, parts):
    d, cols, fam = lc.load(mod)
    X = d[cols].astype(float).values; X = np.where(np.isfinite(X), X, np.nan)
    y = d.label_capture.map({c: i for i, c in enumerate(CL)}).values; batch = d.Maturity_Stage.values
    anc = d.session_id.isin(anchor_sessions).values; rows = []
    for var in ["M0", "M1"]:
        Xv = X if var == "M0" else lc.standardize(X, batch, anc)
        for p in [f"p{i}" for i in range(10)]:
            fold = d.fruit_id.map(parts[p]).values.astype(int); P = np.full((len(d), 3), np.nan)
            for k in range(5):
                te = fold == k; tr = ~te
                med = np.nanmedian(Xv[tr], axis=0); Xtr = np.where(np.isnan(Xv[tr]), med, Xv[tr]); Xte = np.where(np.isnan(Xv[te]), med, Xv[te])
                est, grid = lc.estimator(fam)
                gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="f1_macro", n_jobs=int(os.environ.get("WB_JOBS", "1")))
                P[te] = gs.fit(Xtr, y[tr], groups=d.fruit_id.values[tr]).predict_proba(Xte)
            t = pd.DataFrame(P, columns=PC); t["session_id"] = d.session_id.values
            s = t.groupby("session_id")[PC].mean().reset_index(); s["model"] = mod; s["variant"] = var; s["partition"] = p; rows.append(s)
            print(mod, var, p, flush=True)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    OUT = sys.argv[1]; lc.OUT = OUT
    mods = sys.argv[2:] or ["physics", "thermal", "rgb"]
    A = pd.read_csv(f"{OUT}/anchor_sessions.csv"); anchor_sessions = set(A.session_id[A.anchor])
    parts = pd.read_csv("src/nested_selection/partitions.csv").set_index("fruit_id")
    for m in mods:
        run(m, anchor_sessions, parts).to_parquet(f"{OUT}/within_batch_{m}.parquet", index=False)
