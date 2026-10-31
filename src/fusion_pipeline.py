#!/usr/bin/env python
"""Late/early fusion, leakage, confound and error analysis for durian multimodal ripeness.

Unit of analysis = SESSION = (fruit_id, date_captured).  Each modality contributes a per-capture
out-of-fold (OOF) probability file; the session-level probability of a modality is the mean OOF
probability vector over that session's captures/recordings.

Usage
-----
    python fusion_pipeline.py --config fusion_config.json

The config lists modalities by name; adding a 4th modality (e.g. RGB) is a matter of adding one
entry with its OOF csv (fold_fruit), optional fold_random OOF csv, optional features parquet and
optional per-sample LOBO probability file.  Nothing in the code is specific to three modalities.

Required OOF columns: p_Unripe, p_Ripe, p_Overripe, fruit_id, date_captured (or session_id),
label_capture, fold_fruit (or fold_random), file.
"""
import argparse, itertools, json, os, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

warnings.filterwarnings("ignore")
CLASSES = ["Unripe", "Ripe", "Overripe"]
PCOLS = [f"p_{c}" for c in CLASSES]
SEED = 42
N_JOBS = 1        # sklearn GridSearch (loky workers are blocked in the sandbox); LightGBM uses LGB_THREADS
LGB_THREADS = 6
T975_DF4 = stats.t.ppf(0.975, 4)  # 2.776

# ----------------------------------------------------------------------------- helpers

def session_key(df):
    if "session_id" in df.columns and df.session_id.notna().all():
        return df.session_id.astype(str)
    return df.fruit_id.astype(float).astype(int).astype(str) + "_" + df.date_captured.astype(str)


def load_oof(path):
    d = pd.read_csv(path)
    d["session_id"] = session_key(d)
    d["fruit_id"] = d.fruit_id.astype(float).astype(int)
    return d


def oof_to_session(d, fold_col="fold_fruit"):
    """Mean OOF probability per session.  For fold_random the session has no single fold (NaN)."""
    g = d.groupby("session_id")
    out = g[PCOLS].mean()
    out["n_captures"] = g.size()
    out["fruit_id"] = g.fruit_id.first()
    out["date_captured"] = g.date_captured.first()
    out["label"] = g.label_capture.agg(lambda s: s.mode().iloc[0])
    if fold_col == "fold_fruit":
        nf = g.fold_fruit.nunique()
        assert (nf == 1).all(), "fold_fruit must be constant within a session"
        out["fold"] = g.fold_fruit.first()
    else:
        out["fold"] = np.nan
    return out


def metrics(y, P):
    yhat = np.asarray(P).argmax(1)
    return dict(accuracy=accuracy_score(y, yhat), macro_f1=f1_score(y, yhat, average="macro"),
                n=len(y))


def metrics2(y, P):
    """2-class task: Unripe (0) vs Ripe+Overripe (1), derived from 3-class probabilities."""
    p1 = np.asarray(P)[:, 1] + np.asarray(P)[:, 2]
    y2 = (np.asarray(y) > 0).astype(int)
    yhat = (p1 > 0.5).astype(int)
    return dict(accuracy=accuracy_score(y2, yhat), macro_f1=f1_score(y2, yhat, average="macro"), n=len(y))


def per_fold_eval(y, P, fold, two_class=False):
    """Per-fold acc / macro-F1 plus mean, SD, 95 % t-CI (df = n_folds-1) and pooled metrics."""
    mf = metrics2 if two_class else metrics
    rows = []
    for k in sorted(pd.unique(fold)):
        m = fold == k
        r = mf(np.asarray(y)[m], np.asarray(P)[m]); r["fold"] = int(k); rows.append(r)
    pf = pd.DataFrame(rows)
    nf = len(pf); tq = stats.t.ppf(0.975, nf - 1)
    summ = {}
    for met in ["accuracy", "macro_f1"]:
        mu, sd = pf[met].mean(), pf[met].std(ddof=1)
        summ[f"{met}_mean"] = mu; summ[f"{met}_sd"] = sd
        summ[f"{met}_ci_lo"] = mu - tq * sd / np.sqrt(nf); summ[f"{met}_ci_hi"] = mu + tq * sd / np.sqrt(nf)
    pooled = mf(y, P)
    summ["pooled_accuracy"] = pooled["accuracy"]; summ["pooled_macro_f1"] = pooled["macro_f1"]
    summ["n_sessions"] = len(y)
    if not two_class:
        yhat = np.asarray(P).argmax(1)
        summ["confusion"] = json.dumps(confusion_matrix(y, yhat, labels=[0, 1, 2]).tolist())
        summ["recall_per_class"] = json.dumps(recall_score(y, yhat, average=None, labels=[0, 1, 2]).round(4).tolist())
    return pf, summ


# ----------------------------------------------------------------------------- session table

def build_session_table(mod_oof, fruit_table):
    """mod_oof: {modality: oof_fruit_csv}.  Returns (session_table, per-modality session frames)."""
    per_mod = {m: oof_to_session(load_oof(p), "fold_fruit") for m, p in mod_oof.items()}
    allses = sorted(set().union(*[set(s.index) for s in per_mod.values()]))
    T = pd.DataFrame(index=pd.Index(allses, name="session_id"))
    for m, s in per_mod.items():
        T = T.join(s[PCOLS].add_suffix(f"__{m}"))
        T[f"n_captures__{m}"] = s.n_captures.reindex(T.index)
    meta = pd.concat([s[["fruit_id", "date_captured", "label", "fold"]].assign(modality=m) for m, s in per_mod.items()])
    g = meta.groupby(level=0)
    T["fruit_id"] = g.fruit_id.first(); T["date_captured"] = g.date_captured.first()
    # majority label across modalities (value_counts is stable, so ties fall to the first-listed modality)
    lab = g.label.agg(lambda s: s.value_counts(sort=True).index[0])
    T["label_capture"] = lab
    T["label_disagreement"] = g.label.nunique() > 1
    T["label_by_modality"] = g.apply(lambda d: ";".join(f"{r.modality}={r.label}" for r in d.itertuples()))
    T["y"] = T.label_capture.map({c: i for i, c in enumerate(CLASSES)})
    T["fold_fruit"] = g.fold.first()
    T["fold_disagreement"] = g.fold.nunique() > 1
    T["modalities"] = g.modality.agg(lambda s: "+".join(sorted(s)))
    T["n_modalities"] = g.modality.nunique()
    ft = pd.read_csv(fruit_table).set_index("fruit_id")
    for c in ["Maturity_Stage", "Class", "actual", "Dry_Matter_AVE", "TSS_AVE", "Firmness_AVE", "Harvest_Weight", "Day_Harvested"]:
        if c in ft.columns:
            T[c] = T.fruit_id.map(ft[c])
    # endpoint session = latest date per fruit
    last = T.groupby("fruit_id").date_captured.transform("max")
    T["is_endpoint_session"] = T.date_captured == last
    T = T.reset_index()
    return T, per_mod


# ----------------------------------------------------------------------------- late fusion

def fuse(P_list, rule):
    if rule == "mean":
        return np.mean(P_list, axis=0)
    if rule == "product":
        L = np.sum([np.log(np.clip(P, 1e-6, 1)) for P in P_list], axis=0) / len(P_list)
        E = np.exp(L); return E / E.sum(1, keepdims=True)
    raise ValueError(rule)


def stack(P_list, y, fold, groups, seed=SEED):
    """Multinomial LR stacker on concatenated OOF probabilities, trained on 4 folds / tested on the 5th.
    C chosen by inner fruit-grouped 3-fold CV (macro-F1)."""
    X = np.hstack(P_list)
    out = np.zeros((len(y), 3))
    for k in sorted(pd.unique(fold)):
        te = fold == k; tr = ~te
        gs = GridSearchCV(LogisticRegression(max_iter=5000, class_weight="balanced"),
                          {"C": [0.03, 0.1, 0.3, 1, 3, 10]}, cv=GroupKFold(3), scoring="f1_macro", n_jobs=1)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        out[te] = gs.predict_proba(X[te])
    return out


def load_nested_stacking(spec, T):
    """Nested-stacking session probabilities produced by nested_stacking.py (see its docstring).
    spec = {"probs": npz path, "order": session-order csv}.  Returns {subset_name: array aligned with T}."""
    order = pd.read_csv(spec["order"]).session_id.astype(str).values
    z = np.load(spec["probs"])
    out = {}
    for k in z.files:
        d = pd.DataFrame(z[k], index=order)
        assert set(T.session_id) <= set(order), "nested stacking file does not cover the main session set"
        out[k.replace("__", "+")] = d.loc[T.session_id.astype(str)].values
    return out


def late_fusion_all_subsets(T, mods, rules=("mean", "product", "stacking"), nested=None):
    """T restricted to sessions having all `mods`.  Returns per-fold table, summary table, fused prob dict.
    nested: optional {subset: probs} from fully nested stacking; when given, it replaces the stacker that is
    trained on global out-of-fold probabilities (which lets labels of the outer test fold reach the stacker's
    training inputs through the base models)."""
    y = T.y.values; fold = T.fold_fruit.values.astype(int); groups = T.fruit_id.values
    P = {m: T[[f"{c}__{m}" for c in PCOLS]].values for m in mods}
    pf_rows, summ_rows, fused = [], [], {}
    for r in range(1, len(mods) + 1):
        for sub in itertools.combinations(mods, r):
            name = "+".join(sub)
            for rule in (rules if r > 1 else ["single"]):
                if r == 1:
                    F = P[sub[0]]
                elif rule == "stacking":
                    F = nested[name] if nested is not None else stack([P[m] for m in sub], y, fold, groups)
                else:
                    F = fuse([P[m] for m in sub], rule)
                fused[(name, rule)] = F
                for two in (False, True):
                    pf, s = per_fold_eval(y, F, fold, two_class=two)
                    task = "2class" if two else "3class"
                    pf = pf.assign(subset=name, rule=rule, n_modalities=r, task=task)
                    pf_rows.append(pf)
                    summ_rows.append(dict(subset=name, rule=rule, n_modalities=r, task=task, **s))
    return pd.concat(pf_rows, ignore_index=True), pd.DataFrame(summ_rows), fused


# ----------------------------------------------------------------------------- early fusion

META_COLS = {"modality", "file", "zip_member", "fruit_id", "session_id", "orientation", "device", "date_captured",
             "label_capture", "y", "actual", "Maturity_Stage", "Class", "fold_fruit", "fold_random", "err",
             "cleaned_flag", "read_ok", "n_rows_in_fold_table", "n_sessions_in_fold_table"}


def session_features(parquet, exclude, prefix, file_to_session=None):
    """Session-level (mean over captures/recordings) numeric features.  `file_to_session` (dict file -> session_id,
    taken from the modality's OOF file) guarantees the same session keys as the probability tables."""
    d = pd.read_parquet(parquet)
    if file_to_session is not None:
        d["session_id"] = d.file.map(file_to_session)
        d = d[d.session_id.notna()]
    else:
        d["session_id"] = session_key(d)
    cols = [c for c in d.columns if c not in META_COLS and c not in set(exclude) and pd.api.types.is_numeric_dtype(d[c])]
    X = d.groupby("session_id")[cols].mean().astype(float)
    X.columns = [f"{prefix}::{c}" for c in cols]
    return X


def early_fusion(T, feats, mods, seed=SEED, pca=None, rule_suffix="", require=None):
    """feats: {modality: session-feature DataFrame}.  LightGBM + LR (in-fold standardisation; inner CV for C).
    pca: {modality: n_components} — that modality's block is standardised and PCA-reduced INSIDE each training fold.
    require: only evaluate subsets containing all these modalities (used for the embedding-PCA variant)."""
    import lightgbm as lgb
    from sklearn.decomposition import PCA
    pca = pca or {}
    y = T.y.values; fold = T.fold_fruit.values.astype(int); groups = T.fruit_id.values
    pf_rows, summ_rows, probs = [], [], {}
    for r in range(1, len(mods) + 1):
        for sub in itertools.combinations(mods, r):
            if require and not set(require) <= set(sub): continue
            name = "+".join(sub)
            blocks = {m: np.where(np.isfinite(feats[m].reindex(T.session_id.values).values.astype(float)), feats[m].reindex(T.session_id.values).values.astype(float), np.nan) for m in sub}
            for model in ["early_logreg" + rule_suffix, "early_lgbm" + rule_suffix]:
                out = np.zeros((len(y), 3)); nfeat = 0
                for k in sorted(pd.unique(fold)):
                    te = fold == k; tr = ~te
                    Xtr_parts, Xte_parts = [], []
                    for m in sub:
                        B = blocks[m]; med = np.nanmedian(B[tr], axis=0); Btr = np.where(np.isnan(B[tr]), med, B[tr]); Bte = np.where(np.isnan(B[te]), med, B[te])
                        if m in pca:
                            sc = StandardScaler().fit(Btr); pc = PCA(n_components=min(pca[m], Btr.shape[0] - 1, Btr.shape[1]), random_state=seed).fit(sc.transform(Btr))
                            Btr, Bte = pc.transform(sc.transform(Btr)), pc.transform(sc.transform(Bte))
                        Xtr_parts.append(Btr); Xte_parts.append(Bte)
                    Xtr, Xte = np.hstack(Xtr_parts), np.hstack(Xte_parts); nfeat = Xtr.shape[1]
                    if model.startswith("early_logreg"):
                        pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
                        gs = GridSearchCV(pipe, {"clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1]}, cv=GroupKFold(3), scoring="f1_macro", n_jobs=N_JOBS)
                        gs.fit(Xtr, y[tr], groups=groups[tr]); out[te] = gs.predict_proba(Xte)
                    else:
                        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                                 class_weight="balanced", random_state=seed, n_jobs=LGB_THREADS, verbose=-1)
                        clf.fit(Xtr, y[tr]); out[te] = clf.predict_proba(Xte)
                probs[(name, model)] = out
                for two in (False, True):
                    pf, s = per_fold_eval(y, out, fold, two_class=two)
                    task = "2class" if two else "3class"
                    pf_rows.append(pf.assign(subset=name, rule=model, n_modalities=r, task=task))
                    summ_rows.append(dict(subset=name, rule=model, n_modalities=r, task=task, n_features=nfeat, **s))
    return pd.concat(pf_rows, ignore_index=True), pd.DataFrame(summ_rows), probs


# ----------------------------------------------------------------------------- statistics

def mcnemar_exact(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b)); c = int(np.sum(~correct_a & correct_b))
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))
    return b, c, p


def bootstrap_fruit(T, correct_dict, n_boot=1000, seed=SEED, y=None, preds=None):
    """Resample fruits with replacement; pooled session accuracy and macro-F1 CI for each prediction set, and differences."""
    rng = np.random.default_rng(seed)
    fruits = T.fruit_id.values; uf = np.unique(fruits)
    idx_by_fruit = {f: np.where(fruits == f)[0] for f in uf}
    names = list(preds); res = {n: {"acc": [], "f1": []} for n in names}
    for _ in range(n_boot):
        samp = rng.choice(uf, size=len(uf), replace=True)
        idx = np.concatenate([idx_by_fruit[f] for f in samp])
        for n in names:
            res[n]["acc"].append(accuracy_score(y[idx], preds[n][idx]))
            res[n]["f1"].append(f1_score(y[idx], preds[n][idx], average="macro"))
    out = {}
    for n in names:
        for met in ["acc", "f1"]:
            a = np.array(res[n][met]); out[(n, met)] = (a.mean(), np.percentile(a, 2.5), np.percentile(a, 97.5))
    if len(names) == 2:
        for met in ["acc", "f1"]:
            d = np.array(res[names[1]][met]) - np.array(res[names[0]][met])
            out[("diff", met)] = (d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5))
    return out


def embedding_table(npy_path, manifest_csv, file_to_session):
    """Per-capture frozen CNN embeddings (rows aligned with the manifest; views averaged) -> per-capture DataFrame."""
    E = np.load(npy_path); man = pd.read_csv(manifest_csv)
    assert len(man) == E.shape[0], "embedding rows must align with the manifest"
    if E.ndim == 3: E = E.mean(axis=1)  # average over deterministic views (as in the RGB track's probe)
    d = pd.DataFrame(E, columns=[f"e{i}" for i in range(E.shape[1])])
    d["file"] = man.file.values; d["session_id"] = d.file.map(file_to_session)
    for c in ["fruit_id", "label_capture", "Maturity_Stage", "date_captured"]: d[c] = man[c].values
    return d[d.session_id.notna()].reset_index(drop=True)


def session_embeddings(npy_path, manifest_csv, file_to_session, prefix):
    d = embedding_table(npy_path, manifest_csv, file_to_session)
    cols = [c for c in d.columns if c.startswith("e") and c[1:].isdigit()]
    X = d.groupby("session_id")[cols].mean(); X.columns = [f"{prefix}::{c}" for c in cols]; return X


# ----------------------------------------------------------------------------- LOBO re-fits

def lobo_refit(parquet, exclude, model, batch_col="Maturity_Stage", seed=SEED, embedding=None):
    """Leave-one-harvest-batch-out per-capture probabilities, re-fitted with the track's model family and grid
    (thermal: SVM-RBF; multispectral: LR on features; rgb: LR 'probe' on frozen embeddings, C in {0.001,0.01,0.1}),
    inner fruit-grouped 3-fold CV.  embedding = (npy, manifest, file_to_session) for model == 'embedding_logreg'."""
    if model == "embedding_logreg":
        d = embedding_table(*embedding); cols = [c for c in d.columns if c.startswith("e") and c[1:].isdigit()]
    else:
        d = pd.read_parquet(parquet); d["session_id"] = session_key(d)
        cols = [c for c in d.columns if c not in META_COLS and c not in set(exclude) and pd.api.types.is_numeric_dtype(d[c])]
    d["fruit_id"] = d.fruit_id.astype(float).astype(int)
    X = d[cols].astype(float).values; X = np.where(np.isfinite(X), X, np.nan)
    y = d.label_capture.map({c: i for i, c in enumerate(CLASSES)}).values
    P = np.full((len(d), 3), np.nan)
    for b in sorted(d[batch_col].unique()):
        te = (d[batch_col] == b).values; tr = ~te
        med = np.nanmedian(X[tr], axis=0); Xtr = np.where(np.isnan(X[tr]), med, X[tr]); Xte = np.where(np.isnan(X[te]), med, X[te])
        if model == "svm_rbf":
            est = Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed))])
            grid = {"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.003, 0.01]}
        elif model == "embedding_logreg":
            est = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
            grid = {"clf__C": [0.001, 0.01, 0.1]}
        else:
            est = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
            grid = {"clf__C": [0.01, 0.1, 1, 10]}
        gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="f1_macro", n_jobs=N_JOBS)
        gs.fit(Xtr, y[tr], groups=d.fruit_id.values[tr]); P[te] = gs.predict_proba(Xte)
    out = d[["file", "fruit_id", "session_id", "date_captured", "label_capture", batch_col]].copy()
    out[PCOLS] = P
    return out



# ----------------------------------------------------------------------------- orchestration

def run_all(cfg):
    """cfg keys: fruit_table, out_dir, modalities {name: {oof_fruit, oof_random?, features?, feature_exclude?,
    best_model, lobo_refit_model? ('svm_rbf'|'logreg'), lobo_probs_json? (sound-style per-recording probabilities)}},
    leakage_models {modality: {model: {fruit, random}}}, ablation_refs {label: oof_csv}."""
    out = cfg.get("out_dir", "results"); os.makedirs(out, exist_ok=True)
    mods = list(cfg["modalities"]); M = cfg["modalities"]
    log = {}
    # 1. session table
    T_all, per_mod = build_session_table({m: M[m]["oof_fruit"] for m in mods}, cfg["fruit_table"])
    log["sessions_by_n_modalities"] = T_all.n_modalities.value_counts().to_dict()
    log["label_disagreements"] = int(T_all.label_disagreement.sum()); log["fold_disagreements"] = int(T_all.fold_disagreement.sum())
    T = T_all[T_all.n_modalities == len(mods)].reset_index(drop=True)
    y = T.y.values
    # 2+3. single modality (common denominator) and late fusion, 3-class and 2-class
    nested = load_nested_stacking(cfg["nested_stacking"], T) if cfg.get("nested_stacking") else None
    log["stacking_scheme"] = "nested (inner out-of-fold base-model refits)" if nested is not None else "global out-of-fold"
    pf_late, sm_late, fused = late_fusion_all_subsets(T, mods, nested=nested)
    # 4. early fusion
    oofdf = {m: load_oof(M[m]["oof_fruit"]) for m in mods}
    f2s = {m: dict(zip(d.file, d.session_id)) for m, d in oofdf.items()}
    feats = {m: session_features(M[m]["features"], M[m].get("feature_exclude", []), m, f2s[m]) for m in mods if M[m].get("features")}
    if len(feats) == len(mods):
        pf_early, sm_early, probs_early = early_fusion(T, feats, mods)
        emb_mods = [m for m in mods if M[m].get("embeddings")]
        if emb_mods:  # variant: that modality's hand-crafted features replaced by in-fold PCA of its frozen CNN embeddings
            feats_e = dict(feats)
            for m in emb_mods: feats_e[m] = session_embeddings(M[m]["embeddings"], M[m]["manifest"], f2s[m], m)
            pf_e, sm_e, pr_e = early_fusion(T, feats_e, mods, pca={m: int(M[m].get("embedding_pca", 50)) for m in emb_mods}, rule_suffix="_embpca", require=emb_mods)
            pf_early = pd.concat([pf_early, pf_e], ignore_index=True); sm_early = pd.concat([sm_early, sm_e], ignore_index=True); probs_early.update(pr_e)
    else:
        pf_early, sm_early, probs_early = pf_late.iloc[0:0], sm_late.iloc[0:0], {}
    pf_all = pd.concat([pf_late, pf_early], ignore_index=True); sm_all = pd.concat([sm_late, sm_early], ignore_index=True)
    probs = {**fused, **probs_early}
    s3 = sm_all[sm_all.task == "3class"]
    best_single = s3[s3.rule == "single"].sort_values("macro_f1_mean").iloc[-1]
    best_late = s3[s3.rule.isin(["mean", "product", "stacking"])].sort_values("macro_f1_mean").iloc[-1]
    BF = (best_late.subset, best_late.rule)  # best fusion = best late-fusion rule by mean session macro-F1
    fr = pd.concat([pf_all.assign(record="per_fold"), sm_all.assign(record="summary")], ignore_index=True)
    fr["best_fusion_flag"] = (fr.subset == BF[0]) & (fr.rule == BF[1])
    fr.to_csv(f"{out}/fusion_results.csv", index=False)
    pf_all.to_csv(f"{out}/fusion_results_per_fold.csv", index=False)
    single = fr[fr.rule == "single"].copy(); single["modality"] = single.subset; single.to_csv(f"{out}/single_modality_common_denominator.csv", index=False)
    fr[fr.task == "2class"].to_csv(f"{out}/two_class_results.csv", index=False)
    # 5. statistics
    def pf_vec(k, met):
        d = pf_all[(pf_all.subset == k[0]) & (pf_all.rule == k[1]) & (pf_all.task == "3class")].sort_values("fold"); return d[met].values
    all3 = "+".join(mods)
    comps = [("best_fusion_vs_best_single", (best_single.subset, "single"), BF), ("all_mean_vs_best_single", (best_single.subset, "single"), (all3, "mean")),
             ("all_stacking_vs_best_single", (best_single.subset, "single"), (all3, "stacking")), ("all_mean_vs_best_fusion", BF, (all3, "mean"))]
    if probs_early:
        best_early = s3[s3.rule.str.startswith("early")].sort_values("macro_f1_mean").iloc[-1]; BE = (best_early.subset, best_early.rule)
        comps += [("best_early_vs_best_single", (best_single.subset, "single"), BE), ("best_early_vs_best_fusion", BF, BE), ("best_early_vs_all_mean", (all3, "mean"), BE)]
    if len(mods) > 2:  # best full-set fusion vs best single and vs best (k-1)-modality fusion
        late = s3[s3.rule.isin(["mean", "product", "stacking"])]
        bfull = late[late.n_modalities == len(mods)].sort_values("macro_f1_mean").iloc[-1]
        bkm1 = late[late.n_modalities == len(mods) - 1].sort_values("macro_f1_mean").iloc[-1]
        comps += [("best_full_fusion_vs_best_single", (best_single.subset, "single"), (bfull.subset, bfull.rule)),
                  ("best_full_fusion_vs_best_kminus1_fusion", (bkm1.subset, bkm1.rule), (bfull.subset, bfull.rule))]
    comps += [(f"{m}_vs_best_single", (best_single.subset, "single"), (m, "single")) for m in mods if m != best_single.subset]
    comps = list(dict.fromkeys(comps))
    rows = []
    for name, A, B in comps:
        r = dict(comparison=name, reference="|".join(A), candidate="|".join(B))
        for met in ["accuracy", "macro_f1"]:
            a, b = pf_vec(A, met), pf_vec(B, met)
            r[f"{met}_ref_mean"] = a.mean(); r[f"{met}_cand_mean"] = b.mean(); r[f"{met}_diff_mean"] = (b - a).mean()
            r[f"{met}_paired_t_p"] = stats.ttest_rel(b, a).pvalue
            try: r[f"{met}_wilcoxon_p"] = stats.wilcoxon(b, a).pvalue
            except ValueError: r[f"{met}_wilcoxon_p"] = np.nan
        pa, pb = probs[A].argmax(1), probs[B].argmax(1); ca, cb = pa == y, pb == y
        b_, c_, p = mcnemar_exact(ca, cb)
        r.update(mcnemar_ref_only_correct=b_, mcnemar_cand_only_correct=c_, mcnemar_exact_p=p, pooled_acc_ref=ca.mean(), pooled_acc_cand=cb.mean())
        bs = bootstrap_fruit(T, None, cfg.get("n_boot", 1000), SEED, y=y, preds={"ref": pa, "cand": pb})
        for nm in ["ref", "cand", "diff"]:
            for met in ["acc", "f1"]:
                m_, lo, hi = bs[(nm, met)]; r[f"boot_{met}_{nm}_mean"] = m_; r[f"boot_{met}_{nm}_ci_lo"] = lo; r[f"boot_{met}_{nm}_ci_hi"] = hi
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{out}/stats_tests.csv", index=False)
    # 6. leakage
    def block(d, fold_col):
        yv = d.label_capture.map({c: i for i, c in enumerate(CLASSES)}).values; P = d[PCOLS].values
        o = dict(pooled_accuracy=metrics(yv, P)["accuracy"], pooled_macro_f1=metrics(yv, P)["macro_f1"], n=len(d))
        if fold_col in d and d[fold_col].notna().all():
            pf, _ = per_fold_eval(yv, P, d[fold_col].values.astype(int))
            o.update(accuracy_mean=pf.accuracy.mean(), accuracy_sd=pf.accuracy.std(ddof=1), macro_f1_mean=pf.macro_f1.mean(), macro_f1_sd=pf.macro_f1.std(ddof=1))
        return o
    lrows = []; rand_best = {}
    for m, models in cfg.get("leakage_models", {}).items():
        for mdl, pr in models.items():
            df_f, df_r = load_oof(pr["fruit"]), load_oof(pr["random"]); assert set(df_f.file) == set(df_r.file)
            sf, sr = oof_to_session(df_f, "fold_fruit"), oof_to_session(df_r, "fold_random"); sf["label_capture"] = sf.label; sr["label_capture"] = sr.label
            if mdl == M[m]["best_model"]: rand_best[m] = sr
            for level, (a, b, fc) in {"capture": (df_f, df_r, "fold_fruit"), "session": (sf, sr, "fold_fruit")}.items():
                bf, br = block(a, fc), block(b, "fold_random")
                r = dict(modality=m, model=mdl, level=level, n=bf["n"], fruit_pooled_acc=bf["pooled_accuracy"], random_pooled_acc=br["pooled_accuracy"], fruit_pooled_f1=bf["pooled_macro_f1"], random_pooled_f1=br["pooled_macro_f1"])
                if level == "capture":
                    r.update(fruit_acc_mean=bf["accuracy_mean"], fruit_acc_sd=bf["accuracy_sd"], random_acc_mean=br["accuracy_mean"], random_acc_sd=br["accuracy_sd"], fruit_f1_mean=bf["macro_f1_mean"], fruit_f1_sd=bf["macro_f1_sd"], random_f1_mean=br["macro_f1_mean"], random_f1_sd=br["macro_f1_sd"])
                r["delta_acc_pp"] = 100 * (r["random_pooled_acc"] - r["fruit_pooled_acc"]); r["delta_f1_pp"] = 100 * (r["random_pooled_f1"] - r["fruit_pooled_f1"]); lrows.append(r)
    for rr in range(2, len(mods) + 1):
        for sub in itertools.combinations(mods, rr):
            if not all(m in rand_best for m in sub): continue
            name = "+".join(sub); Pf = fused[(name, "mean")]
            Pr = np.mean([rand_best[m].reindex(T.session_id.values)[PCOLS].values for m in sub], axis=0)
            mf, mr = metrics(y, Pf), metrics(y, Pr)
            lrows.append(dict(modality=name, model="mean-rule fusion of best models", level="session", n=len(T), fruit_pooled_acc=mf["accuracy"], random_pooled_acc=mr["accuracy"], fruit_pooled_f1=mf["macro_f1"], random_pooled_f1=mr["macro_f1"], delta_acc_pp=100 * (mr["accuracy"] - mf["accuracy"]), delta_f1_pp=100 * (mr["macro_f1"] - mf["macro_f1"])))
    lg = cfg.get("leakage_gbm_models")
    if lg:
        Pf = np.mean([oof_to_session(load_oof(cfg["leakage_models"][m][k]["fruit"]), "fold_fruit").reindex(T.session_id.values)[PCOLS].values for m, k in lg.items()], axis=0)
        Pr = np.mean([oof_to_session(load_oof(cfg["leakage_models"][m][k]["random"]), "fold_random").reindex(T.session_id.values)[PCOLS].values for m, k in lg.items()], axis=0)
        mf, mr = metrics(y, Pf), metrics(y, Pr)
        lrows.append(dict(modality="+".join(lg), model="mean-rule fusion of gradient-boosted models", level="session", n=len(T), fruit_pooled_acc=mf["accuracy"], random_pooled_acc=mr["accuracy"], fruit_pooled_f1=mf["macro_f1"], random_pooled_f1=mr["macro_f1"], delta_acc_pp=100 * (mr["accuracy"] - mf["accuracy"]), delta_f1_pp=100 * (mr["macro_f1"] - mf["macro_f1"])))
    pd.DataFrame(lrows).to_csv(f"{out}/leakage_table.csv", index=False)
    # 7. LOBO
    Tm = T.set_index("session_id"); lobo_sess = {}
    for m in mods:
        if M[m].get("lobo_refit_model"):
            emb = (M[m]["embeddings"], M[m]["manifest"], f2s[m]) if M[m]["lobo_refit_model"] == "embedding_logreg" else None
            d = lobo_refit(M[m].get("features"), M[m].get("feature_exclude", []), M[m]["lobo_refit_model"], embedding=emb); d["session_id"] = d.file.map(f2s[m])
            lobo_sess[m] = d.groupby("session_id")[PCOLS].mean()
        elif M[m].get("lobo_probs_json"):
            jj = json.load(open(M[m]["lobo_probs_json"])); parts = []
            for b, dd in jj.items():
                r = pd.DataFrame({"file": dd["file"], "p_Unripe": dd["Unripe"], "p_Ripe": dd["Ripe"], "p_Overripe": dd["Overripe"]}); r["session_id"] = r.file.map(f2s[m]); parts.append(r.groupby("session_id")[PCOLS].mean())
            lobo_sess[m] = pd.concat(parts)
        M[m]["_lobo_source"] = ("refit from frozen embeddings" if M[m].get("lobo_refit_model") == "embedding_logreg" else "refit from features parquet") if M[m].get("lobo_refit_model") else os.path.basename(M[m].get("lobo_probs_json", ""))
    def lobo_row(name, model, source, sp, idx, b, note=""):
        yv = Tm.loc[idx, "y"].values; P = sp.loc[idx, PCOLS].values; r = metrics(yv, P); yhat = P.argmax(1); maj = pd.Series(yv).value_counts(normalize=True).max()
        return dict(held_out_batch=b, modality=name, model=model, source=source, n_sessions=len(idx), session_acc=r["accuracy"], session_macro_f1=r["macro_f1"], majority_rate=maj, acc_minus_majority=r["accuracy"] - maj,
                    recall_per_class=json.dumps(recall_score(yv, yhat, average=None, labels=[0, 1, 2]).round(3).tolist()), confusion=json.dumps(confusion_matrix(yv, yhat, labels=[0, 1, 2]).tolist()), note=note)
    lrows = []; batches = sorted(Tm[cfg.get("batch_col", "Maturity_Stage")].dropna().unique())
    for b in batches:
        idx = Tm.index[Tm[cfg.get("batch_col", "Maturity_Stage")] == b]
        for m in lobo_sess:
            idx_m = idx[idx.isin(lobo_sess[m].index)]
            lrows.append(lobo_row(m, M[m]["best_model"], M[m]["_lobo_source"], lobo_sess[m], idx_m, b))
        for rr in range(2, len(lobo_sess) + 1):
            for sub in itertools.combinations(list(lobo_sess), rr):
                idx_s = idx[np.all([idx.isin(lobo_sess[m].index) for m in sub], axis=0)]
                Pm = pd.DataFrame(np.mean([lobo_sess[m].loc[idx_s, PCOLS].values for m in sub], axis=0), index=idx_s, columns=PCOLS)
                lrows.append(lobo_row("+".join(sub), "mean-rule fusion", "mean of modality LOBO probabilities", Pm, idx_s, b))
    for nm, pth in cfg.get("ablation_refs", {}).items():
        s = oof_to_session(load_oof(pth), "fold_fruit")
        for b in batches + ["ALL (fruit-grouped CV, not LOBO)"]:
            idx = Tm.index[(Tm[cfg.get("batch_col", "Maturity_Stage")] == b) if not b.startswith("ALL") else slice(None)]; idx = idx[idx.isin(s.index)]
            lrows.append(lobo_row(nm, "ablation reference", "fold_fruit OOF restricted to batch (not a LOBO fit)", s, idx, b, note="reference: fruit-grouped CV score within this batch"))
    # tracks' own LOBO metric files (their own session sets) appended for cross-check
    for m, spec in cfg.get("lobo_track_files", {}).items():
        d = pd.read_csv(spec["path"])
        if spec.get("model_filter"): d = d[d.model == spec["model_filter"]]
        for _, r in d.iterrows():
            lrows.append(dict(held_out_batch=r.held_out_batch, modality=m, model=spec.get("model", ""), source=f"{os.path.basename(spec['path'])} (track, own session set)",
                              n_sessions=r[spec["n_col"]], session_acc=r[spec["acc_col"]], session_macro_f1=r[spec["f1_col"]], confusion=r.get(spec.get("confusion_col", ""), np.nan)))
    pd.DataFrame(lrows).to_csv(f"{out}/lobo_table.csv", index=False)
    # 8. error analysis — on a pre-specified model (config "error_analysis_model"), not the post hoc best fusion
    EA = tuple(cfg.get("error_analysis_model", BF)); log["error_analysis_model"] = "|".join(EA)
    Pbf = probs[EA]; T["pred_best_fusion"] = [CLASSES[i] for i in Pbf.argmax(1)]; T["pmax_best_fusion"] = Pbf.max(1)
    for m in mods:
        T[f"pred__{m}"] = [CLASSES[i] for i in probs[(m, "single")].argmax(1)]; T[f"correct__{m}"] = T[f"pred__{m}"] == T.label_capture
    T["correct_best_fusion"] = T.pred_best_fusion == T.label_capture
    T["n_modalities_wrong"] = sum((~T[f"correct__{m}"]).astype(int) for m in mods)
    T["pred_all_mean"] = [CLASSES[i] for i in probs[(all3, "mean")].argmax(1)]; T["correct_all_mean"] = T.pred_all_mean == T.label_capture
    err = []
    for c in ["Maturity_Stage", "actual", "Class", "is_endpoint_session", "label_capture"]:
        ct = pd.crosstab(T[c], T.correct_best_fusion)
        for lvl, r in ct.iterrows():
            err.append(dict(section="crosstab_best_fusion", variable=c, level=str(lvl), n_wrong=int(r.get(False, 0)), n_correct=int(r.get(True, 0)), error_rate=r.get(False, 0) / r.sum()))
    ep = T[T.is_endpoint_session]
    for cls in [None, "Ripe", "Overripe"]:
        sub = ep if cls is None else ep[ep.label_capture == cls]
        for v in ["Dry_Matter_AVE", "TSS_AVE", "Firmness_AVE", "Harvest_Weight"]:
            if v not in sub: continue
            a = sub.loc[sub.correct_best_fusion, v].dropna(); b = sub.loc[~sub.correct_best_fusion, v].dropna()
            if len(b) >= 3 and len(a) >= 3:
                u = stats.mannwhitneyu(a, b)
                err.append(dict(section="mannwhitney_endpoint_sessions_correct_vs_wrong", variable=v if cls is None else f"{v} | endpoint={cls}", n_correct=len(a), n_wrong=len(b), median_correct=a.median(), median_wrong=b.median(), mean_correct=a.mean(), mean_wrong=b.mean(), mannwhitney_U=u.statistic, p=u.pvalue))
    for sec, sel in [("misclassified_sessions_best_fusion", ~T.correct_best_fusion), ("wrong_in_all_modalities", T.n_modalities_wrong == len(mods))]:
        for _, r in T[sel].iterrows():
            err.append(dict(section=sec, session_id=r.session_id, fruit_id=r.fruit_id, level=r.label_capture, pred=r.pred_best_fusion, pmax=r.pmax_best_fusion, Maturity_Stage=r.Maturity_Stage, actual=r.actual, Class=r.Class, is_endpoint=r.is_endpoint_session, n_modalities_wrong=r.n_modalities_wrong, **{f"pred_{m}": r[f"pred__{m}"] for m in mods}, DM=r.Dry_Matter_AVE, TSS=r.TSS_AVE, Firmness=r.Firmness_AVE))
    pd.DataFrame(err).to_csv(f"{out}/error_analysis.csv", index=False)
    # session table (all sessions; fusion probabilities for the common set)
    ST = T_all.copy()
    for k in {BF, (all3, "mean"), (all3, "stacking")} | ({BE} if probs_early else set()):
        tag = f"{k[0]}|{k[1]}".replace("+", "_"); tmp = pd.DataFrame(probs[k], columns=[f"p_{c}__fusion[{tag}]" for c in CLASSES], index=T.session_id)
        ST = ST.merge(tmp, left_on="session_id", right_index=True, how="left")
    ST = ST.merge(T[["session_id", "pred_best_fusion", "correct_best_fusion", "pred_all_mean", "correct_all_mean", "n_modalities_wrong"] + [f"pred__{m}" for m in mods]], on="session_id", how="left")
    ST["in_main_analysis"] = ST.n_modalities == len(mods)
    ST.to_parquet(f"{out}/session_table.parquet", index=False); ST.to_csv(f"{out}/session_table.csv", index=False)
    log.update(n_sessions_main=len(T), best_single=best_single.subset, best_fusion="|".join(BF), best_fusion_f1=float(best_late.macro_f1_mean), best_fusion_acc=float(best_late.accuracy_mean))
    json.dump(log, open(f"{out}/run_log.json", "w"), indent=1, default=str)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); a = ap.parse_args()
    print(json.dumps(run_all(json.load(open(a.config))), indent=1, default=str))
