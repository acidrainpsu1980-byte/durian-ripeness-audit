# Runs inside the kernel via exec(open('run_all.py').read()); expects df, dc, mf, FULL/RAW8 definitions below.
import re, warnings; warnings.filterwarnings("ignore")
from joblib import parallel_backend
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ms_models.py")).read())

feat = pd.read_parquet("features_multispectral.parquet")
assert (feat.file.values == df.file.values).all()
exclude_exact = {"otsu_thr", "seg_band_idx", "seg_ok", "read_ok", "err", "cleaned_flag"}
meta_cols = set(df.columns) | {"zip_member"}
FULL = [c for c in feat.columns if c not in meta_cols and c not in exclude_exact and not c.startswith("bgmean_")]
RAW8 = mf.FEATURE_SETS["raw8"]
FS = {"full": feat[FULL].values.astype(float), "raw8": feat[RAW8].values.astype(float)}
models = make_models()

# ---- 1. all models, fold_fruit
results = {}; t0 = time.time()
with parallel_backend("threading", n_jobs=4):
    for name, (pipe, grid, fs) in models.items():
        oof, chosen = run_cv(df, FS[fs], pipe, grid, "fold_fruit")
        dc.save_predictions(df, oof, "fold_fruit", "multispectral", name)
        results[name] = dict(oof=oof, chosen=chosen, ev=dc.evaluate_oof(df, oof, "fold_fruit"), fs=fs)
        print(name, round(time.time() - t0), "s", flush=True)

def fmt(m, s): return f"{m:.3f} ± {s:.3f}"
rows_ = []
for name, r in results.items():
    s = r["ev"]["summary"]; pc = r["ev"]["pooled_capture"]
    ys, Ps, _ = dc.aggregate_to_session(df, r["oof"]); ps = dc.metrics_from_probs(ys, Ps)
    rows_.append(dict(model=name, features=r["fs"], n_features=len(FULL) if r["fs"] == "full" else 8,
                      hyperparams_per_fold=json.dumps([{k: v for k, v in c.items() if k != "fold"} for c in r["chosen"]]),
                      cap_acc=fmt(s["capture"]["accuracy_mean"], s["capture"]["accuracy_sd"]),
                      cap_f1=fmt(s["capture"]["macro_f1_mean"], s["capture"]["macro_f1_sd"]),
                      ses_acc=fmt(s["session"]["accuracy_mean"], s["session"]["accuracy_sd"]),
                      ses_f1=fmt(s["session"]["macro_f1_mean"], s["session"]["macro_f1_sd"]),
                      pooled_cap_acc=round(pc["accuracy"], 3), pooled_ses_acc=round(ps["accuracy"], 3),
                      pooled_ses_f1=round(ps["macro_f1"], 3), pooled_ses_recall_U_R_O=json.dumps(ps["recall_per_class"]),
                      _ses_f1_mean=s["session"]["macro_f1_mean"]))
summary_tbl = pd.DataFrame(rows_).sort_values("_ses_f1_mean", ascending=False)
best = summary_tbl.model.iloc[0]
summary_tbl.drop(columns="_ses_f1_mean").to_csv("results/summary_multispectral_fold_fruit.csv", index=False)
print(summary_tbl.drop(columns=["_ses_f1_mean", "hyperparams_per_fold"]).to_string(index=False))
print("BEST (session macro-F1):", best)

# ---- 2. leakage experiment: best model on fold_random (+ runner-up for context)
leak_rows = []
with parallel_backend("threading", n_jobs=4):
    for name in [best, summary_tbl.model.iloc[1]]:
        pipe, grid, fs = models[name]
        oof_r, chosen_r = run_cv(df, FS[fs], pipe, grid, "fold_random")
        dc.save_predictions(df, oof_r, "fold_random", "multispectral", name)
        for split, ev in [("fold_fruit", results[name]["ev"]), ("fold_random", dc.evaluate_oof(df, oof_r, "fold_random"))]:
            s = ev["summary"]
            leak_rows.append(dict(model=name, split=split,
                                  cap_acc=fmt(s["capture"]["accuracy_mean"], s["capture"]["accuracy_sd"]),
                                  cap_f1=fmt(s["capture"]["macro_f1_mean"], s["capture"]["macro_f1_sd"]),
                                  ses_acc=fmt(s["session"]["accuracy_mean"], s["session"]["accuracy_sd"]),
                                  ses_f1=fmt(s["session"]["macro_f1_mean"], s["session"]["macro_f1_sd"]),
                                  cap_acc_mean=s["capture"]["accuracy_mean"], ses_f1_mean=s["session"]["macro_f1_mean"]))
        if name == best:
            oof_random_best = oof_r
leak = pd.DataFrame(leak_rows); leak.to_csv("results/leakage_multispectral_fruit_vs_random.csv", index=False)
print(leak.to_string(index=False))

# ---- 3. leave-one-harvest-batch-out with best
pipe, grid, fs = models[best]
with parallel_backend("threading", n_jobs=4):
    lobo, oof_lobo = run_lobo(df, FS[fs], pipe, grid)
lobo.to_csv(f"results/lobo_multispectral__{best}.csv", index=False)
print(lobo.drop(columns=["params", "test_class_counts"]).round(3).to_string(index=False))

# ---- 4. importance
X_full, y = FS["full"], df.y.values
lr_C = results["logreg"]["chosen"][0].get("clf__C", 0.1)
lr = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
               ("clf", LogisticRegression(C=lr_C, max_iter=5000, class_weight="balanced", random_state=SEED))]).fit(X_full, y)
with parallel_backend("threading", n_jobs=4):
    rf = clone(models["rf"][0]).fit(X_full, y)
Fv, pv = f_classif(np.nan_to_num(X_full), y)
imp = pd.DataFrame({"feature": FULL, "logreg_abs_coef_sum": np.abs(lr["clf"].coef_).sum(0),
                    "rf_importance": rf["clf"].feature_importances_, "anova_F": Fv, "anova_p": pv})
imp["rank_mean"] = imp[["logreg_abs_coef_sum", "rf_importance", "anova_F"]].rank(ascending=False).mean(1)
imp = imp.sort_values("rank_mean"); imp.to_csv("results/feature_importance_multispectral.csv", index=False)
print(imp.head(12).round(4).to_string(index=False))

def vip(pls):
    T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, h = W.shape; ss = np.diag(T.T @ T @ Q.T @ Q).reshape(h, -1)
    return np.sqrt(p * (W ** 2 @ ss).ravel() / ss.sum())
nc_mode = int(pd.Series([c["clf__n_components"] for c in results["plsda_raw8"]["chosen"]]).mode()[0])
pls = PLSRegression(nc_mode, scale=False).fit(StandardScaler().fit_transform(FS["raw8"]), np.eye(3)[y])
vipdf = pd.DataFrame({"band": RAW8, "VIP": vip(pls)}).sort_values("VIP", ascending=False)
vipdf.to_csv("results/plsda_raw8_vip_multispectral.csv", index=False); print(vipdf.round(3).to_string(index=False))

# ---- 5. PLSR regression of DM / TSS at fruit level, endpoint session only
ft = pd.read_csv(dc.FRUIT_TABLE_CSV)
ep_date = df.groupby("fruit_id").date_captured.max()
ep = df[df.date_captured.values == df.fruit_id.map(ep_date).values]
agg = feat.loc[ep.index].groupby("fruit_id")[FULL].mean()
agg["fold_fruit"] = ep.groupby("fruit_id").fold_fruit.first()
agg["label_endpoint"] = ep.groupby("fruit_id").label_capture.first()
agg["Maturity_Stage"] = ep.groupby("fruit_id").Maturity_Stage.first()
agg = agg.join(ft.set_index("fruit_id")[["Dry_Matter_AVE", "TSS_AVE", "excluded"]], how="left")
reg_rows = []; reg_pred = agg[["fold_fruit", "label_endpoint", "Maturity_Stage", "Dry_Matter_AVE", "TSS_AVE"]].copy()
for target in ["Dry_Matter_AVE", "TSS_AVE"]:
    for fsname, cols in [("raw8", RAW8), ("refl8", mf.FEATURE_SETS["refl8"]), ("full", FULL)]:
        d = agg.dropna(subset=[target]); Xa = d[cols].values; ya = d[target].values; folds = d.fold_fruit.values
        pred = np.zeros(len(d)); ncs = []
        for k in range(5):
            tr, te = folds != k, folds == k
            gs = GridSearchCV(Pipeline([("sc", StandardScaler()), ("pls", PLSRegression(scale=False))]),
                              {"pls__n_components": list(range(1, 9))}, cv=KFold(3, shuffle=True, random_state=SEED),
                              scoring="neg_root_mean_squared_error").fit(Xa[tr], ya[tr])
            pred[te] = gs.predict(Xa[te]).ravel(); ncs.append(gs.best_params_["pls__n_components"])
        pf = np.array([(r2_score(ya[folds == k], pred[folds == k]), np.sqrt(mean_squared_error(ya[folds == k], pred[folds == k]))) for k in range(5)])
        reg_rows.append(dict(target=target, features=fsname, n_fruits=len(d), n_components_per_fold=str(ncs),
                             R2_mean=round(pf[:, 0].mean(), 3), R2_sd=round(pf[:, 0].std(ddof=1), 3),
                             RMSE_mean=round(pf[:, 1].mean(), 3), RMSE_sd=round(pf[:, 1].std(ddof=1), 3),
                             pooled_R2=round(r2_score(ya, pred), 3), pooled_RMSE=round(np.sqrt(mean_squared_error(ya, pred)), 3),
                             target_sd=round(ya.std(), 3)))
        reg_pred.loc[d.index, f"pred_{target}_{fsname}"] = pred
reg = pd.DataFrame(reg_rows); reg.to_csv("results/plsr_regression_multispectral.csv", index=False)
reg_pred.reset_index().to_csv("results/plsr_fruit_predictions_multispectral.csv", index=False)
print(reg.to_string(index=False))
print("endpoint captures", len(ep), "fruits", agg.shape[0], "DM available", agg.Dry_Matter_AVE.notna().sum(), "TSS available", agg.TSS_AVE.notna().sum())
