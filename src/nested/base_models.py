"""Selected base models of the thermal, multispectral and RGB tracks, re-implemented exactly as in
thermal_models.py, ms_models.py/run_all.py and rgb_models.py, so that they can be refitted on any
subset of outer folds for nested stacking.

Each `fit_predict(train_idx, test_idx)` runs the track's full procedure on the training captures only
(scaling, imputation and the inner fruit-grouped GridSearch for hyper-parameters), then predicts
the test captures.
"""
import os
import numpy as np, pandas as pd
from joblib import parallel_backend
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV

SEED = 42
N_JOBS = 4
R = os.environ.get("DURIAN_INTERMEDIATE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "intermediate"))


def load_thermal():
    feat = pd.read_parquet(f"{R}/features/features_thermal.parquet")
    meta = ["modality", "file", "zip_member", "fruit_id", "session_id", "orientation", "date_captured", "label_capture", "y",
            "actual", "Maturity_Stage", "Class", "fold_fruit", "fold_random"]
    excl = set(meta) | {"device", "ambient_c", "seg_fallback", "otsu_thr_c", "n_unique_vals"}
    cols = [c for c in feat.columns if c not in excl]
    X = feat[cols].values.astype(np.float64)

    def fit_predict(tr, te):
        pipe = Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))])
        gs = GridSearchCV(pipe, {"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.003, 0.01]}, cv=GroupKFold(n_splits=3),
                          scoring="f1_macro", n_jobs=N_JOBS)
        gs.fit(X[tr], feat.y.values[tr], groups=feat.fruit_id.values[tr])
        return gs.predict_proba(X[te])
    return feat, cols, fit_predict


def load_multispectral():
    feat = pd.read_parquet(f"{R}/features/features_multispectral.parquet")
    folds = pd.read_csv(f"{R}/oof/oof_multispectral__logreg__fold_fruit.csv")
    meta = set(folds.columns) | {"zip_member", "modality", "orientation", "device", "actual", "fold_random", "date_captured"}
    excl = {"otsu_thr", "seg_band_idx", "seg_ok", "read_ok", "err", "cleaned_flag"}
    cols = [c for c in feat.columns if c not in meta and c not in excl and not c.startswith("bgmean_")]
    X = feat[cols].values.astype(float)

    def fit_predict(tr, te):
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))])
        ytr, gtr = feat.y.values[tr], feat.fruit_id.values[tr]
        gs = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 1.0, 10.0]}, scoring="f1_macro",
                          cv=GroupKFold(n_splits=3).split(X[tr], ytr, gtr), n_jobs=N_JOBS, refit=True)
        gs.fit(X[tr], ytr)
        return gs.best_estimator_.predict_proba(X[te])
    return feat, cols, fit_predict


def load_rgb():
    man = pd.read_csv(f"{R}/rgb/rgb512_manifest.csv")
    E = np.load(f"{R}/rgb/emb_efficientnetb0.npy").mean(1)        # feature-level TTA over the 5 views

    def fit_predict(tr, te):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))
        gs = GridSearchCV(pipe, {"logisticregression__C": [0.001, 0.01, 0.1]}, cv=GroupKFold(3), scoring="f1_macro", n_jobs=N_JOBS)
        gs.fit(E[tr], man.y.values[tr], groups=man.fruit_id.values[tr])
        return gs.predict_proba(E[te])
    return man, ["emb_efficientnetb0_mean"], fit_predict


LOADERS = {"thermal": load_thermal, "multispectral": load_multispectral, "rgb": load_rgb}
