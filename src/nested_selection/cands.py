"""Candidate model families per sensor, fixed before the nested re-analysis (v1.4, reviewer comment 4); the candidates and their
preprocessing were developed on this dataset, so nested estimates are conditional on this candidate set.

Each family is re-implemented exactly as in the original track scripts (src/thermal/thermal_models.py,
src/multispectral/ms_models.py, src/sound/train_classical.py, src/rgb/rgb_models.py), including its inner
fruit-grouped hyper-parameter search, and trained on the v1.3 common label (intermediate_common/).
Excluded a priori and disclosed: the thermal CNN (CPU-bounded, 15 epochs) and the RGB fine-tunes.
The knock CNN is not refitted here; its outer and fold-pair predictions come from cnn_common.py.
"""
import os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
from joblib import parallel_backend

SEED = 42
NJ = int(os.environ.get("CAND_NJ", "2"))
R = os.environ["DURIAN_INTERMEDIATE"]
EMB_MOBILENET = os.environ.get("EMB_MOBILENET")


class PLSDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2]); Y = np.eye(3)[np.asarray(y, int)]
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False).fit(X, Y); return self
    def decision_function(self, X):
        return self.pls_.predict(X)
    def predict(self, X):
        return self.decision_function(X).argmax(1)
    def predict_proba(self, X):
        s = np.clip(self.decision_function(X), 1e-3, None); return s / s.sum(1, keepdims=True)


def _gs(pipe, grid, n_splits=3):
    return GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=n_splits), scoring="f1_macro", n_jobs=NJ)


def load(mod):
    if mod == "thermal":
        f = pd.read_parquet(f"{R}/features/features_thermal.parquet")
        meta = ["modality", "file", "zip_member", "fruit_id", "session_id", "orientation", "date_captured", "label_capture", "y",
                "actual", "Maturity_Stage", "Class", "fold_fruit", "fold_random"]
        excl = set(meta) | {"device", "ambient_c", "seg_fallback", "otsu_thr_c", "n_unique_vals"}
        X = {"full": f[[c for c in f.columns if c not in excl]].values.astype(np.float64)}
    elif mod == "multispectral":
        f = pd.read_parquet(f"{R}/features/features_multispectral.parquet")
        o = pd.read_csv(f"{R}/oof/oof_multispectral__logreg__fold_fruit.csv")
        meta = set(o.columns) | {"zip_member", "modality", "orientation", "device", "actual", "fold_random", "date_captured"}
        excl = {"otsu_thr", "seg_band_idx", "seg_ok", "read_ok", "err", "cleaned_flag"}
        cols = [c for c in f.columns if c not in meta and c not in excl and not c.startswith("bgmean_")]
        raw8 = [c for c in cols if c.startswith("mean_B")]
        assert len(raw8) == 8, raw8
        X = {"full": f[cols].values.astype(float), "raw8": f[raw8].values.astype(float)}
    elif mod == "sound":
        f = pd.read_parquet(f"{R}/features/features_sound_recordings.parquet")
        meta = ['file', 'fruit_id', 'session_id', 'device', 'date_captured', 'label_capture', 'y', 'Maturity_Stage', 'Class',
                'fold_fruit', 'fold_random', 'n_rows_in_fold_table', 'n_sessions_in_fold_table']
        X = {"full": f[[c for c in f.columns if c not in meta]].values.astype(float)}
        # session assignment from the verified recording manifest (Table S7), not the stale date field
        man = pd.read_csv(os.environ.get("SOUND_MANIFEST", os.path.join(R, "..", "data", "sound_recording_manifest.csv"))).set_index("file")
        f = f.copy(); f["session_id"] = f.file.map(man.session_id); assert f.session_id.notna().all()
    elif mod == "rgb":
        man = pd.read_csv(f"{R}/rgb/rgb512_manifest.csv")
        h = pd.read_parquet(f"{R}/features/features_rgb.parquet")
        assert (h.file.values == man.file.values).all() and (h.y.values == man.y.values).all()
        meta = ["modality", "file", "fruit_id", "session_id", "orientation", "date_captured", "label_capture", "y", "Maturity_Stage", "Class",
                "fold_fruit", "fold_random", "camera", "seg_ok", "mask_frac"]
        X = {"hc": h[[c for c in h.columns if c not in meta]].values.astype(np.float32),
             "effnet": np.load(f"{R}/rgb/emb_efficientnetb0.npy").mean(1)}
        if EMB_MOBILENET:
            X["mobilenet"] = np.load(EMB_MOBILENET).mean(1)
        f = h
    else:
        raise ValueError(mod)
    m = f[["file", "fruit_id", "session_id", "y", "fold_fruit"]].reset_index(drop=True).copy()
    m["fruit_id"] = m.fruit_id.astype(float).astype(int)
    return m, X


def _thermal(name):
    if name == "logreg":
        return "full", _gs(Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))]), {"clf__C": [0.01, 0.1, 1, 10]})
    if name == "rf":
        return "full", Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=NJ))])
    if name == "lgbm":
        return "full", Pipeline([("sc", StandardScaler()), ("clf", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced", random_state=SEED, n_jobs=NJ, verbose=-1))])
    if name == "svm_rbf":
        return "full", _gs(Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))]),
                           {"clf__C": [0.3, 1, 3, 10], "clf__gamma": ["scale", 0.003, 0.01]})


def _ms(name):
    pre = lambda: [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    if name in ("plsda_raw8", "plsda_full"):
        return name.split("_")[1], _gs(Pipeline(pre() + [("clf", PLSDA())]), {"clf__n_components": list(range(2, 9))})
    if name == "logreg":
        return "full", _gs(Pipeline(pre() + [("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))]), {"clf__C": [0.01, 0.1, 1.0, 10.0]})
    if name == "rf":
        return "full", Pipeline(pre() + [("clf", RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=NJ, random_state=SEED))])
    if name == "lightgbm":
        return "full", Pipeline(pre() + [("clf", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                                                     colsample_bytree=0.7, reg_lambda=1.0, class_weight="balanced", n_jobs=NJ, random_state=SEED, verbose=-1))])
    if name == "svm_rbf":
        return "full", _gs(Pipeline(pre() + [("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))]),
                           {"clf__C": [0.3, 1, 3, 10, 30], "clf__gamma": ["scale", 0.003, 0.01, 0.03]})


def _sound(name):
    if name == "logreg":
        return "full", make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced", random_state=SEED))
    if name == "knn":
        return "full", _gs(make_pipeline(StandardScaler(), KNeighborsClassifier(weights="distance")), {"kneighborsclassifier__n_neighbors": [3, 5, 7, 11, 15, 21]}, 4)
    if name == "rf":
        return "full", make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=NJ, random_state=SEED))
    if name == "svm":
        return "full", _gs(make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED)), {"svc__C": [0.5, 1, 2, 5, 10], "svc__gamma": ["scale"]}, 4)
    if name == "lgbm":
        return "full", make_pipeline(StandardScaler(), lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=5, min_child_samples=10, subsample=0.8,
                                                                          subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, class_weight="balanced", n_jobs=NJ, random_state=SEED, verbose=-1))


def _rgb(name):
    if name == "hc_logreg":
        return "hc", _gs(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)), {"logisticregression__C": [0.01, 0.1, 1.0]})
    if name == "hc_rf":
        return "hc", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=NJ)
    if name == "hc_lgbm":
        return "hc", lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, class_weight="balanced", random_state=SEED, n_jobs=NJ, verbose=-1)
    if name in ("mobilenetv2_probe", "efficientnetb0_probe"):
        return ("mobilenet" if name.startswith("mobilenet") else "effnet"), _gs(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)), {"logisticregression__C": [0.001, 0.01, 0.1]})


CANDIDATES = {
    "thermal": ["logreg", "rf", "lgbm", "svm_rbf"],
    "multispectral": ["plsda_raw8", "plsda_full", "logreg", "rf", "lightgbm", "svm_rbf"],
    "sound": ["logreg", "knn", "rf", "svm", "lgbm", "cnn"],
    "rgb": ["hc_logreg", "hc_rf", "hc_lgbm", "mobilenetv2_probe", "efficientnetb0_probe"],
}
SELECTED_V13 = {"thermal": "svm_rbf", "multispectral": "logreg", "sound": "cnn", "rgb": "efficientnetb0_probe"}
_FACT = {"thermal": _thermal, "multispectral": _ms, "sound": _sound, "rgb": _rgb}


def fit_predict(mod, cand, meta, X, tr, te):
    xs, est = _FACT[mod](cand)
    Xm = X[xs]; y = meta.y.values; g = meta.fruit_id.values
    with parallel_backend("threading", n_jobs=NJ):
        if isinstance(est, GridSearchCV):
            est.fit(Xm[tr], y[tr], groups=g[tr])
        else:
            est.fit(Xm[tr], y[tr])
        return est.predict_proba(Xm[te])
