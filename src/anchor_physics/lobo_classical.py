"""Pre-registered analysis (docs/ANALYSIS_PLAN_anchor_physics.md), step 2: leave-one-harvest-batch-out (LOBO) refits of the classical
and probe models with the adaptation variants of plan §3.

Models: thermal SVM-RBF (67 features), RGB EfficientNet-B0 probe, sound LR on the 265 recording features (secondary), physics-guided LR
(f_dom, f_first, centroid, decay20, 1/Q, stiffness, device). Each is refitted exactly as fusion_pipeline.lobo_refit does (same family,
grid, inner fruit-grouped 3-fold CV, median imputation), with an optional per-batch standardization applied to the features first:
  M0 none | M1 anchored: per-batch mean/SD of that batch's anchor captures | M2 unanchored: per-batch mean/SD of all captures.
Post-hoc on M0: M3 anchored logit shift, M4 prior-shift EM (Saerens et al. 2002).
Control: M0 of the three shipped models must reproduce results/lobo_table.csv (session accuracy and macro-F1 per held-out batch).
usage (repo root): python src/anchor_physics/lobo_classical.py OUT
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import fusion_pipeline as fp

CL = fp.CLASSES; PC = fp.PCOLS; SEED = fp.SEED
CFG = json.load(open("src/fusion_config.json"))["modalities"]


def f2s(mod):
    o = fp.load_oof(CFG[mod]["oof_fruit"]); return dict(zip(o.file, o.session_id))


def load(mod):
    """-> DataFrame with file, fruit_id, session_id, label_capture, Maturity_Stage and feature matrix X, model family."""
    if mod == "rgb":
        d = fp.embedding_table(CFG["rgb"]["embeddings"], CFG["rgb"]["manifest"], f2s("rgb"))
        cols = [c for c in d.columns if c.startswith("e") and c[1:].isdigit()]; fam = "embedding_logreg"
    elif mod in ("thermal", "sound_lr"):
        m = "thermal" if mod == "thermal" else "sound"
        d = pd.read_parquet(CFG[m]["features"]); d["session_id"] = fp.session_key(d)
        if m == "sound":
            man = pd.read_csv("data/sound_recording_manifest.csv").set_index("file"); d["session_id"] = d.file.map(man.session_id)
        cols = [c for c in d.columns if c not in fp.META_COLS and c not in set(CFG[m].get("feature_exclude", [])) and pd.api.types.is_numeric_dtype(d[c])]
        fam = "svm_rbf" if mod == "thermal" else "logreg"
    elif mod == "physics":
        s = pd.read_parquet(CFG["sound"]["features"])[["file", "fruit_id", "label_capture", "Maturity_Stage"]]      # the 432 verified recordings
        p = pd.read_parquet(f"{OUT}/physics_recordings.parquet").set_index("file")
        man = pd.read_csv("data/sound_recording_manifest.csv").set_index("file")
        d = s.copy(); d["session_id"] = d.file.map(man.session_id)
        for c in ["f_dom", "f_first", "centroid", "decay20", "inv_q", "stiffness"]: d[c] = d.file.map(p[c]).astype(float)
        d["device_iphone"] = (d.file.map(p.device) == "iphone").astype(float)
        cols = ["f_dom", "f_first", "centroid", "decay20", "inv_q", "stiffness", "device_iphone"]; fam = "logreg"
        assert d[cols].notna().all().all()
    d["fruit_id"] = d.fruit_id.astype(float).astype(int)
    return d.reset_index(drop=True), cols, fam


def standardize(X, batch, mask):
    """Per-batch (x - mean)/SD with statistics over the rows in `mask` of that batch (NaN-aware); constant columns -> SD 1."""
    Z = X.copy()
    for b in np.unique(batch):
        rb = batch == b; ref = X[rb & mask]
        mu = np.nanmean(ref, 0); sd = np.nanstd(ref, 0); sd = np.where((sd > 0) & np.isfinite(sd), sd, 1.0); mu = np.where(np.isfinite(mu), mu, 0.0)
        Z[rb] = (X[rb] - mu) / sd
    return Z


def estimator(fam):
    if fam == "svm_rbf":
        return Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))]), \
               {"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.003, 0.01]}
    if fam == "embedding_logreg":
        return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]), {"clf__C": [0.001, 0.01, 0.1]}
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))]), {"clf__C": [0.01, 0.1, 1, 10]}


def saerens_em(P, prior_src, n_iter=100, tol=1e-8):
    w = prior_src.copy()
    for _ in range(n_iter):
        Q = P * (w / prior_src); Q /= Q.sum(1, keepdims=True); w_new = Q.mean(0)
        if np.abs(w_new - w).max() < tol: w = w_new; break
        w = w_new
    Q = P * (w / prior_src); return Q / Q.sum(1, keepdims=True)


def lobo(mod, anchor_sessions):
    d, cols, fam = load(mod)
    X = d[cols].astype(float).values; X = np.where(np.isfinite(X), X, np.nan)
    y = d.label_capture.map({c: i for i, c in enumerate(CL)}).values; batch = d.Maturity_Stage.values
    anc = d.session_id.isin(anchor_sessions).values
    out = {}
    for var in ["M0", "M1", "M2"]:
        Xv = X if var == "M0" else standardize(X, batch, anc if var == "M1" else np.ones(len(d), bool))
        P = np.full((len(d), 3), np.nan); Psrc_anchor = {}; prior = {}
        for b in sorted(np.unique(batch)):
            te = batch == b; tr = ~te
            med = np.nanmedian(Xv[tr], axis=0); Xtr = np.where(np.isnan(Xv[tr]), med, Xv[tr]); Xte = np.where(np.isnan(Xv[te]), med, Xv[te])
            est, grid = estimator(fam)
            gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="f1_macro", n_jobs=1).fit(Xtr, y[tr], groups=d.fruit_id.values[tr])
            P[te] = gs.predict_proba(Xte)
            if var == "M0":
                Psrc_anchor[b] = gs.predict_proba(Xtr[anc[tr]]); prior[b] = np.bincount(y[tr], minlength=3) / tr.sum()
        out[var] = P
        if var == "M0":
            P3 = P.copy(); P4 = P.copy()
            for b in sorted(np.unique(batch)):
                te = batch == b
                shift = np.log(Psrc_anchor[b] + 1e-9).mean(0) - np.log(P[te & anc] + 1e-9).mean(0)       # M3
                L = np.log(P[te] + 1e-9) + shift; L = np.exp(L - L.max(1, keepdims=True)); P3[te] = L / L.sum(1, keepdims=True)
                P4[te] = saerens_em(P[te], prior[b])                                                     # M4
            out["M3"], out["M4"] = P3, P4
        print(mod, var, "done", flush=True)
    rows = []
    for var, P in out.items():
        t = pd.DataFrame(P, columns=PC); t["session_id"] = d.session_id.values; t["file"] = d.file.values
        s = t.groupby("session_id")[PC].mean().reset_index(); s["model"] = mod; s["variant"] = var; rows.append(s)
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    OUT = sys.argv[1]
    A = pd.read_csv(f"{OUT}/anchor_sessions.csv"); anchor_sessions = set(A.session_id[A.anchor])
    R = pd.concat([lobo(m, anchor_sessions) for m in ["thermal", "rgb", "sound_lr", "physics"]], ignore_index=True)
    R.to_parquet(f"{OUT}/lobo_classical_sessions.parquet", index=False)
    # control: M0 reproduces the shipped LOBO table for the three shipped models
    ST = pd.read_csv("results/session_table.csv", low_memory=False); T = ST[ST.n_modalities == 4].set_index("session_id")
    LT = pd.read_csv("results/lobo_table.csv")
    for mod, name in [("thermal", "thermal"), ("rgb", "rgb")]:
        S = R[(R.model == mod) & (R.variant == "M0")].set_index("session_id")
        for b in ["Immature", "Mature", "Overmature"]:
            idx = T.index[(T.Maturity_Stage == b) & T.index.isin(S.index)]
            yv = T.loc[idx, "y"].values.astype(int); pr = S.loc[idx, PC].values.argmax(1)
            ref = LT[(LT.held_out_batch == b) & (LT.modality == name)].iloc[0]
            da, df_ = abs(accuracy_score(yv, pr) - ref.session_acc), abs(f1_score(yv, pr, average="macro") - ref.session_macro_f1)
            print("control", mod, b, round(da, 12), round(df_, 12)); assert da < 1e-12 and df_ < 1e-12
