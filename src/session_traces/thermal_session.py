# thermal track - literal cells executed in the interactive analysis session (chronological, successful cells only,
# exploration/figure cells omitted).  Kept for transparency: this is the code that produced the shipped
# intermediate/oof files.  It is NOT a standalone script (it was executed cell by cell in one kernel, with files
# being edited between cells); the consolidated final-state driver is src/thermal/thermal_models.py.
# Absolute paths were not present in these cells; the fold table was read from the working directory.

# ==========================================================================================
# cell 2: Inspecting thermal fold table and sample CSV format  [python]
# ==========================================================================================

import os, sys; sys.path.append(os.getcwd())
import durian_common as dc, numpy as np, pandas as pd, io, zipfile
df_all = dc.load_folds("durian_mm_capture_folds.csv", "thermal")
print(df_all.shape, df_all.file.str.endswith(".csv").sum(), df_all.file.str.endswith(".jpg").sum())
df = df_all[df_all.file.str.endswith(".csv") & ~df_all.zip_member.str.startswith("__MACOSX")].reset_index(drop=True)
print(df.shape, df.fruit_id.nunique(), df.session_id.nunique())
print(df.label_capture.value_counts().to_dict())
print(df.groupby("fold_fruit").fruit_id.nunique().to_dict(), df.groupby("fold_fruit").size().to_dict())
print(df.Maturity_Stage.value_counts().to_dict(), df.date_captured.nunique(), df.device.unique()[:5])
zf = dc.open_zip("thermal")
names = zf.namelist()
print(len(names), sum(n.startswith("__MACOSX") for n in names))
missing = [m for m in df.zip_member if m not in set(names)]
print("missing members:", len(missing))
b = dc.read_member(zf, df.zip_member[0])
print(len(b)); print(b[:300])

# ==========================================================================================
# cell 21: Checking final segmentation stats and building feature table  [python]
# ==========================================================================================

importlib.reload(tf)
F = pd.read_parquet("thermal_feat_raw.parquet"); imgs = np.load("thermal_imgs128.npy")
assert (F.zip_member.values == df.zip_member.values).all()
print("unreadable:", (F.read_error!="").sum(), "fallback:", int(F.seg_fallback.sum()))
print(pd.crosstab(df.label_capture, F.seg_fallback).to_dict())
print("quality q:", F.seg_quality.quantile([0,.05,.25,.5,.75,1]).round(2).tolist())
print("area_frac q:", F.area_frac.quantile([0,.05,.5,.95,1]).round(3).tolist())
# random QC gallery of 24 for visual check
pick2 = np.random.RandomState(7).choice(len(df), 24, replace=False)
fig, axes = plt.subplots(3, 8, figsize=(20, 6))
for ax, i in zip(axes.ravel(), pick2):
    M = tf.load_matrix(df.zip_member[i]); mask, info = tf.segment(M)
    ax.imshow(np.clip(M, np.percentile(M,1), np.percentile(M,99)), cmap="inferno"); ax.contour(mask, colors="cyan", linewidths=0.6)
    ax.set_title(f"{df.label_capture[i]} {df.orientation[i]} q={info['seg_quality']:.1f} fb={info['seg_fallback']}", fontsize=7); ax.axis("off")
fig.tight_layout(); fig.savefig("figs/seg_qc_random.png", dpi=60)

# assemble feature table
meta_cols = ["modality","file","zip_member","fruit_id","session_id","orientation","date_captured","label_capture","y","actual","Maturity_Stage","Class","fold_fruit","fold_random"]
feat = pd.concat([df[meta_cols], F.drop(columns=["zip_member","read_error","shape_h","shape_w"])], axis=1)
feat["ambient_c"] = feat.bg_median
EXCLUDE = set(meta_cols) | {"ambient_c","seg_fallback","otsu_thr_c"}
feature_cols = [c for c in feat.columns if c not in EXCLUDE]
print(len(feature_cols)); print(feature_cols)
feat.to_parquet("features_thermal.parquet", index=False)
print(feat[feature_cols].isna().sum().sum(), np.isinf(feat[feature_cols].values).sum())

# ==========================================================================================
# cell 22: Training LR, RF, LightGBM, SVM with fruit-grouped CV  [python]
# ==========================================================================================

import json, warnings; warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
import lightgbm as lgb
SEED = 42
X = feat[feature_cols].values.astype(np.float64); y = feat.y.values; groups = feat.fruit_id.values

def make_model(name):
    inner = GroupKFold(n_splits=3)
    if name == "logreg":
        pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))])
        return GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10]}, cv=inner, scoring="f1_macro", n_jobs=4)
    if name == "rf":
        return Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=4))])
    if name == "lgbm":
        return Pipeline([("sc", StandardScaler()), ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced", random_state=SEED, n_jobs=4, verbose=-1))])
    if name == "svm_rbf":
        pipe = Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))])
        return GridSearchCV(pipe, {"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.003, 0.01]}, cv=inner, scoring="f1_macro", n_jobs=4)
    raise ValueError(name)

def run_cv(name, X, split="fold_fruit", cols=None):
    oof = np.zeros((len(X), 3)); chosen = []
    for k, tr, te in dc.iter_folds(feat, split):
        m = make_model(name)
        fit_kw = {"groups": groups[tr]} if isinstance(m, GridSearchCV) else {}
        m.fit(X[tr], y[tr], **fit_kw)
        if isinstance(m, GridSearchCV): chosen.append(m.best_params_)
        oof[te] = m.predict_proba(X[te])
    return oof, chosen

results = {}
for name in ["logreg", "rf", "lgbm", "svm_rbf"]:
    t = time.time()
    oof, chosen = run_cv(name, X)
    p1, p2 = dc.save_predictions(feat, oof, "fold_fruit", "thermal", name)
    ev = dc.evaluate_oof(feat, oof, "fold_fruit"); results[name] = ev
    s = ev["summary"]
    print(f"{name:8s} cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} f1 {s['capture']['macro_f1_mean']:.3f} | ses acc {s['session']['accuracy_mean']:.3f}±{s['session']['accuracy_sd']:.3f} f1 {s['session']['macro_f1_mean']:.3f}±{s['session']['macro_f1_sd']:.3f} | {time.time()-t:.0f}s {chosen[:2]}")

# ==========================================================================================
# cell 23: Training LR, RF, LightGBM, SVM with threading backend  [python]
# ==========================================================================================

from joblib import parallel_backend
results = {}
with parallel_backend("threading", n_jobs=4):
    for name in ["logreg", "rf", "lgbm", "svm_rbf"]:
        t = time.time()
        oof, chosen = run_cv(name, X)
        p1, p2 = dc.save_predictions(feat, oof, "fold_fruit", "thermal", name)
        ev = dc.evaluate_oof(feat, oof, "fold_fruit"); results[name] = ev
        s = ev["summary"]
        print(f"{name:8s} cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} f1 {s['capture']['macro_f1_mean']:.3f} | ses acc {s['session']['accuracy_mean']:.3f}±{s['session']['accuracy_sd']:.3f} f1 {s['session']['macro_f1_mean']:.3f}±{s['session']['macro_f1_sd']:.3f} | {time.time()-t:.0f}s {chosen[:2]}")

# ==========================================================================================
# cell 24: Training LR, RF, LightGBM, SVM with threading backend  [python]
# ==========================================================================================

feat["device"] = df.device.values
results = {}
with parallel_backend("threading", n_jobs=4):
    for name in ["logreg", "rf", "lgbm", "svm_rbf"]:
        t = time.time()
        oof, chosen = run_cv(name, X)
        p1, p2 = dc.save_predictions(feat, oof, "fold_fruit", "thermal", name)
        ev = dc.evaluate_oof(feat, oof, "fold_fruit"); results[name] = ev
        s = ev["summary"]
        print(f"{name:8s} cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} f1 {s['capture']['macro_f1_mean']:.3f} | ses acc {s['session']['accuracy_mean']:.3f}±{s['session']['accuracy_sd']:.3f} f1 {s['session']['macro_f1_mean']:.3f}±{s['session']['macro_f1_sd']:.3f} | {time.time()-t:.0f}s {chosen[:2]}")

# ==========================================================================================
# cell 25: Fitting ambient-only confound model and relative-feature ablation  [python]
# ==========================================================================================

# pooled OOF session metrics for tie-break
def pooled_session(ev_name):
    oof = pd.read_csv(f"results/oof_thermal__{ev_name}__fold_fruit.csv")
    ys, Ps, _ = dc.aggregate_to_session(feat, oof[[f"p_{c}" for c in dc.CLASSES]].values)
    return dc.metrics_from_probs(ys, Ps)
for n in results: 
    m = pooled_session(n); print(n, "pooled session acc", round(m["accuracy"],3), "f1", round(m["macro_f1"],3))

# (1) ambient-only: background median temperature -> label (quantifies batch confound)
abs_cols = ["fruit_mean","fruit_median","fruit_min","fruit_max","fruit_p5","fruit_p25","fruit_p75","fruit_p95","bg_mean","bg_median","bg_p5","bg_p95","frame_min","frame_max","frame_mean"]
rel_cols = [c for c in feature_cols if c not in abs_cols]
ablations = {"ambient_only": ["bg_median"], "ambient_bgstats": ["bg_median","bg_mean","bg_std","bg_p5","bg_p95"], "relative_only_lgbm": rel_cols}
abl_res = {}
with parallel_backend("threading", n_jobs=4):
    for an, cols in ablations.items():
        Xa = feat[cols].values.astype(float)
        mname = "lgbm" if an.endswith("lgbm") else "logreg"
        oof, _ = run_cv(mname, Xa)
        dc.save_predictions(feat, oof, "fold_fruit", "thermal", an)
        ev = dc.evaluate_oof(feat, oof, "fold_fruit"); abl_res[an] = ev; s = ev["summary"]
        print(f"{an:20s} ({mname}, {len(cols)} feats) cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} f1 {s['capture']['macro_f1_mean']:.3f} | ses acc {s['session']['accuracy_mean']:.3f}±{s['session']['accuracy_sd']:.3f} f1 {s['session']['macro_f1_mean']:.3f}±{s['session']['macro_f1_sd']:.3f}")
# majority-class baseline at session level
ses = feat.groupby("session_id").y.first(); print("session class counts", ses.value_counts().to_dict(), "majority acc", round(ses.value_counts(normalize=True).max(),3))
# ambient by class / batch table
print(feat.groupby(["Maturity_Stage","label_capture"]).ambient_c.agg(["mean","std","size"]).round(2))

# ==========================================================================================
# cell 28: Running fold_random leakage and leave-one-batch-out experiments  [python]
# ==========================================================================================

# Provisional best among feature models by session macro-F1 (mean per fold): svm_rbf vs lgbm tied; run both leakage/LOBO for the top-2 so final choice (after CNN) is covered.
leak = {}
with parallel_backend("threading", n_jobs=2):
    for name in ["svm_rbf", "lgbm"]:
        oof_r, _ = run_cv(name, X, split="fold_random")
        dc.save_predictions(feat, oof_r, "fold_random", "thermal", name)
        ev_r = dc.evaluate_oof(feat, oof_r, "fold_random"); leak[name] = ev_r
        print(name, "fold_random cap acc %.3f±%.3f f1 %.3f | ses acc %.3f f1 %.3f" % (ev_r["summary"]["capture"]["accuracy_mean"], ev_r["summary"]["capture"]["accuracy_sd"], ev_r["summary"]["capture"]["macro_f1_mean"], ev_r["summary"]["session"]["accuracy_mean"], ev_r["summary"]["session"]["macro_f1_mean"]))
    # LOBO
    lobo_rows = []
    for name in ["svm_rbf", "lgbm"]:
        for b, tr, te in dc.leave_one_batch_out(feat):
            m = make_model(name); fit_kw = {"groups": groups[tr]} if isinstance(m, GridSearchCV) else {}
            m.fit(X[tr], y[tr], **fit_kw); P = m.predict_proba(X[te])
            cm = dc.metrics_from_probs(y[te], P); ys, Ps, _ = dc.aggregate_to_session(feat.iloc[te], P); sm = dc.metrics_from_probs(ys, Ps)
            lobo_rows.append(dict(model=name, held_out_batch=b, n_captures=cm["n"], n_sessions=sm["n"], capture_accuracy=cm["accuracy"], capture_macro_f1=cm["macro_f1"],
                                  session_accuracy=sm["accuracy"], session_macro_f1=sm["macro_f1"], session_recall_unripe=sm["recall_per_class"][0], session_recall_ripe=sm["recall_per_class"][1], session_recall_overripe=sm["recall_per_class"][2], session_confusion=json.dumps(sm["confusion"])))
lobo = pd.DataFrame(lobo_rows)
print(lobo.drop(columns=["session_confusion"]).round(3).to_string())

# ==========================================================================================
# cell 31: Refitting feature models without quantization artefact feature  [python]
# ==========================================================================================

feature_cols = [c for c in feature_cols if c != "n_unique_vals"]
rel_cols = [c for c in rel_cols if c != "n_unique_vals"]
X = feat[feature_cols].values.astype(np.float64)
results = {}; chosen_params = {}
with parallel_backend("threading", n_jobs=4):
    # acquisition-artefact-only model
    oof, _ = run_cv("lgbm", feat[["n_unique_vals"]].values.astype(float))
    dc.save_predictions(feat, oof, "fold_fruit", "thermal", "quantization_only"); ev = dc.evaluate_oof(feat, oof, "fold_fruit"); abl_res["quantization_only_lgbm"] = ev
    print("quantization_only  ses acc %.3f f1 %.3f" % (ev["summary"]["session"]["accuracy_mean"], ev["summary"]["session"]["macro_f1_mean"]))
    for name in ["logreg", "rf", "lgbm", "svm_rbf"]:
        oof, chosen = run_cv(name, X); chosen_params[name] = chosen
        dc.save_predictions(feat, oof, "fold_fruit", "thermal", name)
        ev = dc.evaluate_oof(feat, oof, "fold_fruit"); results[name] = ev; s = ev["summary"]
        print(f"{name:8s} cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} f1 {s['capture']['macro_f1_mean']:.3f}±{s['capture']['macro_f1_sd']:.3f} | ses acc {s['session']['accuracy_mean']:.3f}±{s['session']['accuracy_sd']:.3f} f1 {s['session']['macro_f1_mean']:.3f}±{s['session']['macro_f1_sd']:.3f}")
    oof, _ = run_cv("lgbm", feat[rel_cols].values.astype(float)); dc.save_predictions(feat, oof, "fold_fruit", "thermal", "relative_only_lgbm")
    ev = dc.evaluate_oof(feat, oof, "fold_fruit"); abl_res["relative_only_lgbm"] = ev
    print("relative_only_lgbm ses acc %.3f f1 %.3f" % (ev["summary"]["session"]["accuracy_mean"], ev["summary"]["session"]["macro_f1_mean"]))

# ==========================================================================================
# cell 34: Dropping dead fruit_warmer column and refitting feature models  [python]
# ==========================================================================================

assert feat.fruit_warmer.nunique() == 1
feat = feat.drop(columns=["fruit_warmer"]); feat.to_parquet("features_thermal.parquet", index=False)
feature_cols = [c for c in feature_cols if c != "fruit_warmer"]; rel_cols = [c for c in rel_cols if c != "fruit_warmer"]
X = feat[feature_cols].values.astype(np.float64); print(len(feature_cols))
results = {}; chosen_params = {}
with parallel_backend("threading", n_jobs=4):
    for name in ["logreg", "rf", "lgbm", "svm_rbf"]:
        oof, chosen = run_cv(name, X); chosen_params[name] = chosen
        dc.save_predictions(feat, oof, "fold_fruit", "thermal", name)
        ev = dc.evaluate_oof(feat, oof, "fold_fruit"); results[name] = ev; s = ev["summary"]
        print(f"{name:8s} cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} f1 {s['capture']['macro_f1_mean']:.3f}±{s['capture']['macro_f1_sd']:.3f} | ses acc {s['session']['accuracy_mean']:.3f}±{s['session']['accuracy_sd']:.3f} f1 {s['session']['macro_f1_mean']:.3f}±{s['session']['macro_f1_sd']:.3f}")
    oof, _ = run_cv("lgbm", feat[rel_cols].values.astype(float)); dc.save_predictions(feat, oof, "fold_fruit", "thermal", "relative_only_lgbm")
    ev = dc.evaluate_oof(feat, oof, "fold_fruit"); abl_res["relative_only_lgbm"] = ev
    print("relative_only_lgbm ses acc %.3f f1 %.3f" % (ev["summary"]["session"]["accuracy_mean"], ev["summary"]["session"]["macro_f1_mean"]))

# ==========================================================================================
# cell 36: Re-running fold_random and LOBO for final SVM feature set  [python]
# ==========================================================================================

best = "svm_rbf"; leak = {}
def lobo_run(name, Xa, label):
    rows = []
    for b, tr, te in dc.leave_one_batch_out(feat):
        m = make_model(name); fit_kw = {"groups": groups[tr]} if isinstance(m, GridSearchCV) else {}
        m.fit(Xa[tr], y[tr], **fit_kw); P = m.predict_proba(Xa[te])
        cm = dc.metrics_from_probs(y[te], P); ys_, Ps_, _ = dc.aggregate_to_session(feat.iloc[te], P); sm = dc.metrics_from_probs(ys_, Ps_)
        rows.append(dict(model=label, held_out_batch=b, n_train_captures=len(tr), n_captures=cm["n"], n_sessions=sm["n"], capture_accuracy=cm["accuracy"], capture_macro_f1=cm["macro_f1"],
                         session_accuracy=sm["accuracy"], session_macro_f1=sm["macro_f1"], session_recall_unripe=sm["recall_per_class"][0], session_recall_ripe=sm["recall_per_class"][1], session_recall_overripe=sm["recall_per_class"][2], session_confusion=json.dumps(sm["confusion"])))
    return rows
with parallel_backend("threading", n_jobs=4):
    for name in [best, "lgbm"]:
        oof_r, _ = run_cv(name, X, split="fold_random"); dc.save_predictions(feat, oof_r, "fold_random", "thermal", name)
        leak[name] = dc.evaluate_oof(feat, oof_r, "fold_random"); s = leak[name]["summary"]
        print(f"{name} fold_random cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} f1 {s['capture']['macro_f1_mean']:.3f} | ses acc {s['session']['accuracy_mean']:.3f}±{s['session']['accuracy_sd']:.3f} f1 {s['session']['macro_f1_mean']:.3f}")
    lobo = pd.DataFrame(lobo_run(best, X, best) + lobo_run("lgbm", X, "lgbm") + lobo_run("lgbm", feat[rel_cols].values.astype(float), "relative_only_lgbm"))
lobo.to_csv(f"results/lobo_thermal__{best}.csv", index=False)
print(lobo.drop(columns=["session_confusion","n_train_captures"]).round(3).to_string())
# session-level per-batch label distribution for context
print(feat.groupby("session_id").first().groupby("Maturity_Stage").label_capture.value_counts().unstack())

# ==========================================================================================
# cell 38: Compiling results summary table and per-fold metrics  [python]
# ==========================================================================================

def row(name, ev, split):
    s = ev["summary"]; return dict(model=name, split=split, capture_acc=f"{s['capture']['accuracy_mean']:.3f} ± {s['capture']['accuracy_sd']:.3f}", capture_macro_f1=f"{s['capture']['macro_f1_mean']:.3f} ± {s['capture']['macro_f1_sd']:.3f}",
                                   session_acc=f"{s['session']['accuracy_mean']:.3f} ± {s['session']['accuracy_sd']:.3f}", session_macro_f1=f"{s['session']['macro_f1_mean']:.3f} ± {s['session']['macro_f1_sd']:.3f}")
rows = [row(n, results[n], "fold_fruit") for n in results] + [row(n, abl_res[n], "fold_fruit") for n in ["ambient_only","ambient_bgstats","quantization_only_lgbm","relative_only_lgbm"]] + [row(n, leak[n], "fold_random") for n in leak]
summ = pd.DataFrame(rows); summ.to_csv("results/summary_thermal_models.csv", index=False)
print(summ.to_string(index=False))
# per-fold table for best model
pf = pd.DataFrame(results[best]["per_fold"])[["fold","level","accuracy","macro_f1","n"]]; print(pf.pivot(index="fold", columns="level", values=["accuracy","macro_f1","n"]).round(3))
print("svm params per fold:", chosen_params[best]); print("logreg C:", [c["clf__C"] for c in chosen_params["logreg"]])
print("pooled session confusion (rows true U/R/O):", cm_ses.tolist())
print("session recall fold_fruit:", dc.metrics_from_probs(ys, Ps)["recall_per_class"], " fold_random:", dc.metrics_from_probs(ys_r, Ps_r)["recall_per_class"])
