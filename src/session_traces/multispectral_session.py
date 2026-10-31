# multispectral track - literal cells executed in the interactive analysis session (chronological, successful cells only,
# exploration/figure cells omitted).  Kept for transparency: this is the code that produced the shipped
# intermediate/oof files.  It is NOT a standalone script (it was executed cell by cell in one kernel, with files
# being edited between cells); the consolidated final-state driver is src/multispectral/ms_run_models.py.
# Absolute paths were not present in these cells; the fold table was read from the working directory.

# ==========================================================================================
# cell 1: Inspecting multispectral TIFF structure and fold table  [python]
# ==========================================================================================

import sys, os, io, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import durian_common as dc
import tifffile

df = dc.load_folds("durian_mm_capture_folds.csv", "multispectral")
print(df.shape, df.fruit_id.nunique(), df.session_id.nunique())
print(df.label_capture.value_counts().to_dict())
print(df.orientation.value_counts(dropna=False).to_dict())
print(df.file.str.extract(r'\.(\w+)$')[0].value_counts().to_dict())
print(df.fold_fruit.value_counts().sort_index().to_dict())
print(df.groupby('fold_fruit').fruit_id.nunique().to_dict())

zf = dc.open_zip("multispectral")
names = zf.namelist()
print(len(names), names[:3])
print(sum(n.startswith("__MACOSX") for n in names))
for i in [0, 500, 1500]:
    m = df.zip_member.iloc[i]
    b = dc.read_member(zf, m)
    with tifffile.TiffFile(io.BytesIO(b)) as tf:
        print(m, len(b), "pages", len(tf.pages), "series", len(tf.series))
        for p in tf.pages[:2]:
            print("  ", p.shape, p.dtype, p.photometric, p.compression, p.samplesperpixel, p.bitspersample)
        tags = tf.pages[0].tags
        for t in tags.values():
            if t.name not in ("StripOffsets","StripByteCounts","TileOffsets","TileByteCounts"):
                v = str(t.value); print("   ", t.name, v[:120])
        arr = tf.asarray()
        print("  asarray", arr.shape, arr.dtype, arr.min(), arr.max())

# ==========================================================================================
# cell 19: Loading features and running five classifiers with fruit-grouped CV  [python]
# ==========================================================================================

feat = pd.read_parquet("features_multispectral.parquet")
assert (feat.file.values == df.file.values).all()
print(feat.shape, feat.seg_ok.sum(), feat.cleaned_flag.value_counts(dropna=False).to_dict())
print(feat.fruit_area_px.describe().round(0).to_dict())
print(feat.groupby("label_capture")[[f"refl_{b}" for b in mf.BAND_NAMES]].mean().round(3))
print(feat.groupby("label_capture")[["slope_per100nm","fruit_area_px","ND_1_8"]].mean().round(4))

exclude_prefix = ("bgmean_",)
exclude_exact = {"otsu_thr","seg_band_idx","seg_ok","read_ok","err","cleaned_flag"}
meta_cols = set(df.columns) | {"zip_member"}
FULL = [c for c in feat.columns if c not in meta_cols and c not in exclude_exact and not c.startswith(exclude_prefix)]
RAW8 = mf.FEATURE_SETS["raw8"]
print(len(FULL), "full features;", "nan frac", float(feat[FULL].isna().mean().mean()))
FS = {"full": feat[FULL].values.astype(float), "raw8": feat[RAW8].values.astype(float)}

exec(open("ms_models.py").read())
models = make_models()
results = {}; t0 = time.time()
for name, (pipe, grid, fs) in models.items():
    oof, chosen = run_cv(df, FS[fs], pipe, grid, "fold_fruit")
    p1, p2 = dc.save_predictions(df, oof, "fold_fruit", "multispectral", name)
    ev = dc.evaluate_oof(df, oof, "fold_fruit")
    results[name] = dict(oof=oof, chosen=chosen, ev=ev, fs=fs)
    s = ev["summary"]
    print(f"{name:12s} cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} "
          f"ses acc {s['session']['accuracy_mean']:.3f}±{s['session']['accuracy_sd']:.3f} "
          f"ses F1 {s['session']['macro_f1_mean']:.3f}±{s['session']['macro_f1_sd']:.3f}  {chosen[0]}  t={time.time()-t0:.0f}s")

# ==========================================================================================
# cell 20: Running five classifiers with threaded joblib backend  [python]
# ==========================================================================================

from joblib import parallel_backend
results = {}; t0 = time.time()
with parallel_backend("threading", n_jobs=4):
    for name, (pipe, grid, fs) in models.items():
        oof, chosen = run_cv(df, FS[fs], pipe, grid, "fold_fruit")
        p1, p2 = dc.save_predictions(df, oof, "fold_fruit", "multispectral", name)
        ev = dc.evaluate_oof(df, oof, "fold_fruit")
        results[name] = dict(oof=oof, chosen=chosen, ev=ev, fs=fs)
        s = ev["summary"]
        print(f"{name:12s} cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} "
              f"ses acc {s['session']['accuracy_mean']:.3f}±{s['session']['accuracy_sd']:.3f} "
              f"ses F1 {s['session']['macro_f1_mean']:.3f}±{s['session']['macro_f1_sd']:.3f}  {[c for c in chosen]}  t={time.time()-t0:.0f}s")

# ==========================================================================================
# cell 21: Summarising fold-fruit results across models  [python]
# ==========================================================================================

import warnings; warnings.filterwarnings("ignore")
rows_ = []
for name, r in results.items():
    s = r["ev"]["summary"]; pc = r["ev"]["pooled_capture"]
    ys, Ps, _ = dc.aggregate_to_session(df, r["oof"]); ps = dc.metrics_from_probs(ys, Ps)
    rows_.append(dict(model=name, features=r["fs"],
        cap_acc=f"{s['capture']['accuracy_mean']:.3f} ± {s['capture']['accuracy_sd']:.3f}",
        cap_f1=f"{s['capture']['macro_f1_mean']:.3f} ± {s['capture']['macro_f1_sd']:.3f}",
        ses_acc=f"{s['session']['accuracy_mean']:.3f} ± {s['session']['accuracy_sd']:.3f}",
        ses_f1=f"{s['session']['macro_f1_mean']:.3f} ± {s['session']['macro_f1_sd']:.3f}",
        pooled_cap_acc=round(pc["accuracy"],3), pooled_ses_acc=round(ps["accuracy"],3), pooled_ses_f1=round(ps["macro_f1"],3),
        pooled_ses_recall=ps["recall_per_class"], ses_f1_mean=s['session']['macro_f1_mean']))
summary_tbl = pd.DataFrame(rows_).sort_values("ses_f1_mean", ascending=False)
print(summary_tbl.drop(columns="ses_f1_mean").to_string(index=False))
summary_tbl.drop(columns="ses_f1_mean").to_csv("results/summary_multispectral_fold_fruit.csv", index=False)

# ==========================================================================================
# cell 22: Running leakage (fold_random) and leave-one-batch-out with logistic regression  [python]
# ==========================================================================================

best = "logreg"
pipe, grid, fs = models[best]
print("majority baseline capture", round(df.label_capture.value_counts(normalize=True).max(),3),
      "session", round(df.groupby("session_id").label_capture.first().value_counts(normalize=True).max(),3),
      "n sessions", df.session_id.nunique())
with parallel_backend("threading", n_jobs=4):
    oof_r, chosen_r = run_cv(df, FS[fs], pipe, grid, "fold_random")
    dc.save_predictions(df, oof_r, "fold_random", "multispectral", best)
    ev_r = dc.evaluate_oof(df, oof_r, "fold_random")
    s = ev_r["summary"]
    print(f"fold_random {best}: cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} cap F1 {s['capture']['macro_f1_mean']:.3f}±{s['capture']['macro_f1_sd']:.3f} "
          f"ses acc {s['session']['accuracy_mean']:.3f}±{s['session']['accuracy_sd']:.3f} ses F1 {s['session']['macro_f1_mean']:.3f}±{s['session']['macro_f1_sd']:.3f}", chosen_r)
    lobo, oof_lobo = run_lobo(df, FS[fs], pipe, grid)
lobo.to_csv(f"results/lobo_multispectral__{best}.csv", index=False)
print(lobo.drop(columns=["params"]).to_string(index=False))
# also svm_rbf for fold_random (runner-up) for context
with parallel_backend("threading", n_jobs=4):
    oof_r2, _ = run_cv(df, FS["full"], *models["svm_rbf"][:2], "fold_random")
dc.save_predictions(df, oof_r2, "fold_random", "multispectral", "svm_rbf")
s = dc.evaluate_oof(df, oof_r2, "fold_random")["summary"]
print(f"fold_random svm_rbf: cap acc {s['capture']['accuracy_mean']:.3f}±{s['capture']['accuracy_sd']:.3f} ses F1 {s['session']['macro_f1_mean']:.3f}±{s['session']['macro_f1_sd']:.3f}")

# ==========================================================================================
# cell 31: Running full modelling pipeline on final features  [python]
# ==========================================================================================

import glob
for p in glob.glob("results/*"): os.remove(p)
exec(open("run_all.py").read())
