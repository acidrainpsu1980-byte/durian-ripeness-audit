"""Fully nested stacking for the durian multimodal fusion.

Outer loop: the frozen fruit-grouped 5-fold split (fold k = outer test fold).
For each outer fold k and each sensor subset:
  1. Stacker TRAINING inputs: for every outer-training fold j != k, the session probabilities of fold j
     come from base models refitted on the three folds outside {k, j} (cnn_pairs.py and the sklearn
     pair fits; each base model reruns its own inner GridSearch on those three folds only).
     No capture of fold k enters any base-model fit, hyper-parameter search or stacker fit.
  2. Stacker hyper-parameter C: chosen by fruit-grouped 3-fold CV on the outer-training sessions only.
  3. Stacker TEST inputs: session probabilities of fold k from base models trained on the four folds
     != k (the standard out-of-fold predictions, which never saw fold k).
The previous (non-nested) stacker, trained on the global out-of-fold probabilities, is rerun with the
identical stacker and C grid for comparison.

Writes results/nested_stacking_probs.npz (subset -> (n_sessions, 3), rows in session order of
results/nested_session_order.csv), results/nested_vs_global_stacking.csv and a per-fold table.
"""
import itertools, json, os, sys, warnings
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE); os.chdir(_HERE)
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import base_models as bm

warnings.filterwarnings("ignore")
CL = ["Unripe", "Ripe", "Overripe"]; PC = [f"p_{c}" for c in CL]
MODS = ["thermal", "multispectral", "sound", "rgb"]
C_GRID = {"C": [0.03, 0.1, 0.3, 1, 3, 10]}
os.makedirs("results", exist_ok=True)


def load_session_table(path):
    ST = pd.read_parquet(path)
    if "in_main_analysis" in ST: ST = ST[ST.in_main_analysis]
    return ST.reset_index(drop=True)


def capture_tables():
    """Per modality: capture metadata (session_id, fold) aligned with the pair-fit prediction arrays."""
    out = {}
    for m in ["thermal", "multispectral", "rgb"]:
        feat = bm.LOADERS[m]()[0]
        out[m] = (feat.session_id.astype(str).values, feat.fold_fruit.values.astype(int), None)
    rec2 = pd.read_parquet("handoff/rec2_meta.parquet")
    rec2["session_id"] = rec2.fruit_id.astype(float).astype(int).astype(str) + "_" + rec2.date_captured.astype(str)
    knock = pd.read_parquet("features_sound_knocks.parquet")
    out["sound"] = (rec2.session_id.values, rec2.fold_fruit.values.astype(int), (knock.file.values, rec2.file.values))
    return out


def pair_array(m, a, b):
    a, b = min(a, b), max(a, b)
    return np.load(f"handoff/{'cnn' if m == 'sound' else m}_pair_{a}{b}.npy")


def to_session(m, P, cap):
    """Capture (or knock) probabilities -> session mean, exactly as in the tracks."""
    sid, _, sound = cap
    if sound is not None:                          # knock -> recording mean -> session mean
        kfile, rfile = sound
        t = pd.DataFrame(P, columns=CL); t["file"] = kfile
        P = t.groupby("file")[CL].mean().reindex(rfile).values
    t = pd.DataFrame(P, columns=CL); t["session_id"] = sid
    return t.dropna().groupby("session_id")[CL].mean()


def inner_session_probs(ST, caps, available):
    """inner[(m, k)] = DataFrame of session probs for every session NOT in fold k, from fits excluding {k, j}."""
    inner = {}
    for m in available:
        for k in range(5):
            parts = []
            for j in range(5):
                if j == k: continue
                s = to_session(m, pair_array(m, k, j), caps[m])
                fold_j = ST.loc[ST.fold_fruit == j, "session_id"]
                parts.append(s.reindex(fold_j))
            inner[(m, k)] = pd.concat(parts)
    return inner


def fit_stacker(X, y, groups):
    gs = GridSearchCV(LogisticRegression(max_iter=5000, class_weight="balanced"), C_GRID,
                      cv=GroupKFold(3), scoring="f1_macro", n_jobs=1)
    gs.fit(X, y, groups=groups)
    return gs


def run(ST, available):
    caps = capture_tables()
    inner = inner_session_probs(ST, caps, available)
    y = ST.y.values.astype(int); fold = ST.fold_fruit.values.astype(int); groups = ST.fruit_id.values
    G = {m: ST[[f"{c}__{m}" for c in PC]].values for m in MODS}      # global OOF session probs (outer-test inputs)
    probs_nested, probs_global, chosen = {}, {}, []
    for r in range(2, len(MODS) + 1):
        for sub in itertools.combinations(MODS, r):
            if not set(sub) <= set(available): continue
            name = "+".join(sub)
            Pn = np.zeros((len(y), 3)); Pg = np.zeros((len(y), 3))
            Xg = np.hstack([G[m] for m in sub])
            for k in range(5):
                te = fold == k; tr = ~te
                sid_tr = ST.session_id.values[tr]
                Xtr = np.hstack([inner[(m, k)].reindex(sid_tr).values for m in sub])
                assert not np.isnan(Xtr).any(), (name, k)
                gs = fit_stacker(Xtr, y[tr], groups[tr]); Pn[te] = gs.predict_proba(Xg[te])
                chosen.append(dict(subset=name, fold=k, scheme="nested", C=gs.best_params_["C"]))
                gg = fit_stacker(Xg[tr], y[tr], groups[tr]); Pg[te] = gg.predict_proba(Xg[te])
                chosen.append(dict(subset=name, fold=k, scheme="global_oof", C=gg.best_params_["C"]))
            probs_nested[name], probs_global[name] = Pn, Pg
    return probs_nested, probs_global, pd.DataFrame(chosen), inner


def per_fold(y, P, fold, two=False):
    rows = []
    for k in range(5):
        m = fold == k
        if two:
            yt = (y[m] > 0).astype(int); yp = ((P[m, 1] + P[m, 2]) > 0.5).astype(int)
        else:
            yt = y[m]; yp = P[m].argmax(1)
        rows.append(dict(fold=k, accuracy=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro")))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    ST = load_session_table(sys.argv[1])
    available = [m for m in MODS if all(os.path.exists(f"handoff/{'cnn' if m == 'sound' else m}_pair_{a}{b}.npy")
                                       for a, b in itertools.combinations(range(5), 2))]
    print("modalities with complete inner fits:", available, flush=True)
    Pn, Pg, chosen, inner = run(ST, available)
    y = ST.y.values.astype(int); fold = ST.fold_fruit.values.astype(int)
    rows, pf_rows = [], []
    for name in Pn:
        for scheme, P in [("nested", Pn[name]), ("global_oof", Pg[name])]:
            for two in (False, True):
                pf = per_fold(y, P, fold, two)
                pf_rows.append(pf.assign(subset=name, scheme=scheme, task="2class" if two else "3class"))
                rows.append(dict(subset=name, scheme=scheme, task="2class" if two else "3class",
                                 accuracy_mean=pf.accuracy.mean(), accuracy_sd=pf.accuracy.std(ddof=1),
                                 macro_f1_mean=pf.macro_f1.mean(), macro_f1_sd=pf.macro_f1.std(ddof=1)))
    res = pd.DataFrame(rows)
    res.to_csv("results/nested_vs_global_stacking.csv", index=False)
    pd.concat(pf_rows).to_csv("results/nested_vs_global_stacking_per_fold.csv", index=False)
    chosen.to_csv("results/nested_stacking_chosen_C.csv", index=False)
    np.savez("results/nested_stacking_probs.npz", **{k.replace("+", "__"): v for k, v in Pn.items()})
    ST[["session_id", "fruit_id", "y", "fold_fruit"]].to_csv("results/nested_session_order.csv", index=False)
    # inner-fit quality vs standard OOF (same sessions): sanity check that the 3-fold base models are not degenerate
    q = []
    for (m, k), s in inner.items():
        ss = s.dropna(); g = ST.set_index("session_id").loc[ss.index]
        q.append(dict(modality=m, outer_fold=k, n_sessions=len(ss),
                      inner_acc=accuracy_score(g.y.astype(int), ss.values.argmax(1)),
                      oof_acc_same_sessions=accuracy_score(g.y.astype(int), g[[f"{c}__{m}" for c in PC]].values.argmax(1))))
    pd.DataFrame(q).to_csv("results/nested_inner_fit_quality.csv", index=False)
    piv = res.pivot_table(index=["task", "subset"], columns="scheme", values=["accuracy_mean", "macro_f1_mean"]).round(3)
    print(piv.to_string())
