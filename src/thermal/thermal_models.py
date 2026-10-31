#!/usr/bin/env python
"""Thermal track: feature-table assembly, classical models, confound ablations, leakage and LOBO.

Consolidated from the interactive analysis session (final state; see src/session_traces/thermal_session.py
for the literal executed cells).  Run from work/thermal/ after run_extract.py has produced
thermal_feat_raw.parquet and thermal_imgs128.npy:

    python ../../src/thermal/run_extract.py
    python ../../src/thermal/thermal_models.py
    python ../../src/thermal/run_cnn.py fold_fruit        # optional thermal CNN baseline

Outputs (in ./results): oof_thermal__<model>__fold_fruit.csv (+ metrics json) for logreg / rf / lgbm / svm_rbf,
ablations ambient_only / ambient_bgstats / quantization_only / relative_only_lgbm, fold_random OOF for svm_rbf
and lgbm, lobo_thermal__svm_rbf.csv, summary_thermal_models.csv.  Seed 42 throughout.
"""
import os, sys, json, time, warnings
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
import numpy as np, pandas as pd
import durian_common as dc
from joblib import parallel_backend
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
import lightgbm as lgb

warnings.filterwarnings("ignore")
SEED = 42
N_JOBS = int(os.environ.get("DURIAN_N_JOBS", "4"))
os.makedirs("results", exist_ok=True)

# ----------------------------------------------------------------------------- 1. feature table
df_all = dc.load_folds(dc.FOLDS_CSV, "thermal")
df = df_all[df_all.file.str.endswith(".csv") & ~df_all.zip_member.str.startswith("__MACOSX")].reset_index(drop=True)
F = pd.read_parquet("thermal_feat_raw.parquet")
assert (F.zip_member.values == df.zip_member.values).all(), "thermal_feat_raw.parquet is not aligned with the fold table"
print("captures", len(df), "| unreadable:", (F.read_error != "").sum(), "| segmentation fallback:", int(F.seg_fallback.sum()))

meta_cols = ["modality", "file", "zip_member", "fruit_id", "session_id", "orientation", "date_captured", "label_capture", "y",
             "actual", "Maturity_Stage", "Class", "fold_fruit", "fold_random"]
feat = pd.concat([df[meta_cols], F.drop(columns=["zip_member", "read_error", "shape_h", "shape_w"])], axis=1)
feat["ambient_c"] = feat.bg_median
feat["device"] = df.device.values
if "fruit_warmer" in feat.columns and feat.fruit_warmer.nunique() == 1:   # constant column, dropped in the final run
    feat = feat.drop(columns=["fruit_warmer"])
feat.to_parquet("features_thermal.parquet", index=False)

# Final feature set: all segmentation-derived features except the acquisition artefact n_unique_vals
# (export quantization; kept only for the quantization_only ablation), the ambient proxy and QC flags.
EXCLUDE = set(meta_cols) | {"device", "ambient_c", "seg_fallback", "otsu_thr_c", "n_unique_vals"}
feature_cols = [c for c in feat.columns if c not in EXCLUDE]
abs_cols = ["fruit_mean", "fruit_median", "fruit_min", "fruit_max", "fruit_p5", "fruit_p25", "fruit_p75", "fruit_p95",
            "bg_mean", "bg_median", "bg_p5", "bg_p95", "frame_min", "frame_max", "frame_mean"]
rel_cols = [c for c in feature_cols if c not in abs_cols]
print(len(feature_cols), "features;", len(rel_cols), "relative (non-absolute-temperature) features")
X = feat[feature_cols].values.astype(np.float64); y = feat.y.values; groups = feat.fruit_id.values


# ----------------------------------------------------------------------------- 2. models
def make_model(name):
    inner = GroupKFold(n_splits=3)
    if name == "logreg":
        pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))])
        return GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1, 10]}, cv=inner, scoring="f1_macro", n_jobs=N_JOBS)
    if name == "rf":
        return Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                                                                 random_state=SEED, n_jobs=N_JOBS))])
    if name == "lgbm":
        return Pipeline([("sc", StandardScaler()), ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                                                                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                                                             class_weight="balanced", random_state=SEED, n_jobs=N_JOBS, verbose=-1))])
    if name == "svm_rbf":
        pipe = Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))])
        return GridSearchCV(pipe, {"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.003, 0.01]}, cv=inner, scoring="f1_macro", n_jobs=N_JOBS)
    raise ValueError(name)


def run_cv(name, Xm, split="fold_fruit"):
    oof = np.zeros((len(Xm), 3)); chosen = []
    for k, tr, te in dc.iter_folds(feat, split):
        m = make_model(name)
        fit_kw = {"groups": groups[tr]} if isinstance(m, GridSearchCV) else {}
        m.fit(Xm[tr], y[tr], **fit_kw)
        if isinstance(m, GridSearchCV): chosen.append(m.best_params_)
        oof[te] = m.predict_proba(Xm[te])
    return oof, chosen


def report(tag, ev):
    s = ev["summary"]
    print(f"{tag:22s} cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} f1 {s['capture']['macro_f1_mean']:.3f} | "
          f"ses acc {s['session']['accuracy_mean']:.3f}±{s['session']['accuracy_sd']:.3f} f1 {s['session']['macro_f1_mean']:.3f}±{s['session']['macro_f1_sd']:.3f}", flush=True)


def lobo_run(name, Xa, label):
    rows = []
    for b, tr, te in dc.leave_one_batch_out(feat):
        m = make_model(name); fit_kw = {"groups": groups[tr]} if isinstance(m, GridSearchCV) else {}
        m.fit(Xa[tr], y[tr], **fit_kw); P = m.predict_proba(Xa[te])
        cm = dc.metrics_from_probs(y[te], P); ys_, Ps_, _ = dc.aggregate_to_session(feat.iloc[te], P); sm = dc.metrics_from_probs(ys_, Ps_)
        rows.append(dict(model=label, held_out_batch=b, n_train_captures=len(tr), n_captures=cm["n"], n_sessions=sm["n"],
                         capture_accuracy=cm["accuracy"], capture_macro_f1=cm["macro_f1"], session_accuracy=sm["accuracy"], session_macro_f1=sm["macro_f1"],
                         session_recall_unripe=sm["recall_per_class"][0], session_recall_ripe=sm["recall_per_class"][1],
                         session_recall_overripe=sm["recall_per_class"][2], session_confusion=json.dumps(sm["confusion"])))
    return rows


results, abl_res, leak, chosen_params = {}, {}, {}, {}
with parallel_backend("threading", n_jobs=N_JOBS):
    # 2a. main models, fruit-grouped folds
    for name in ["logreg", "rf", "lgbm", "svm_rbf"]:
        t = time.time(); oof, chosen = run_cv(name, X); chosen_params[name] = chosen
        dc.save_predictions(feat, oof, "fold_fruit", "thermal", name)
        results[name] = dc.evaluate_oof(feat, oof, "fold_fruit"); report(f"{name} ({time.time()-t:.0f}s)", results[name])
    # 2b. confound ablations
    ablations = {"ambient_only": (["bg_median"], "logreg"), "ambient_bgstats": (["bg_median", "bg_mean", "bg_std", "bg_p5", "bg_p95"], "logreg"),
                 "quantization_only": (["n_unique_vals"], "lgbm"), "relative_only_lgbm": (rel_cols, "lgbm")}
    for an, (cols, mname) in ablations.items():
        oof, _ = run_cv(mname, feat[cols].values.astype(float))
        dc.save_predictions(feat, oof, "fold_fruit", "thermal", an)
        abl_res[an] = dc.evaluate_oof(feat, oof, "fold_fruit"); report(f"{an} ({mname},{len(cols)})", abl_res[an])
    # 2c. leakage control: capture-level random folds (best model + GBM)
    best = "svm_rbf"
    for name in [best, "lgbm"]:
        oof_r, _ = run_cv(name, X, split="fold_random"); dc.save_predictions(feat, oof_r, "fold_random", "thermal", name)
        leak[name] = dc.evaluate_oof(feat, oof_r, "fold_random"); report(f"{name} fold_random", leak[name])
    # 2d. leave-one-harvest-batch-out
    lobo = pd.DataFrame(lobo_run(best, X, best) + lobo_run("lgbm", X, "lgbm") + lobo_run("lgbm", feat[rel_cols].values.astype(float), "relative_only_lgbm"))
lobo.to_csv(f"results/lobo_thermal__{best}.csv", index=False)
print(lobo.drop(columns=["session_confusion", "n_train_captures"]).round(3).to_string())

# ----------------------------------------------------------------------------- 3. summary table
def row(name, ev, split):
    s = ev["summary"]
    return dict(model=name, split=split,
                capture_acc=f"{s['capture']['accuracy_mean']:.3f} ± {s['capture']['accuracy_sd']:.3f}", capture_macro_f1=f"{s['capture']['macro_f1_mean']:.3f} ± {s['capture']['macro_f1_sd']:.3f}",
                session_acc=f"{s['session']['accuracy_mean']:.3f} ± {s['session']['accuracy_sd']:.3f}", session_macro_f1=f"{s['session']['macro_f1_mean']:.3f} ± {s['session']['macro_f1_sd']:.3f}")
rows = [row(n, results[n], "fold_fruit") for n in results] + [row(n, abl_res[n], "fold_fruit") for n in abl_res] + [row(n, leak[n], "fold_random") for n in leak]
summ = pd.DataFrame(rows); summ.to_csv("results/summary_thermal_models.csv", index=False)
print(summ.to_string(index=False))
print("svm params per fold:", chosen_params[best])
json.dump({k: v for k, v in chosen_params.items()}, open("results/thermal_chosen_hyperparams.json", "w"), indent=1, default=str)
