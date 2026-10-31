"""Classification models for the multispectral track. Executed inside the kernel via exec()."""
import numpy as np, pandas as pd, json, os, time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score
import lightgbm as lgb
import durian_common as dc

SEED = 42
N_JOBS = 4


class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS regression on one-hot targets; class = argmax of predicted scores.
    predict_proba = clipped, row-normalised predicted scores (for fusion only)."""
    def __init__(self, n_components=2):
        self.n_components = n_components
    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2])
        Y = np.eye(3)[np.asarray(y, int)]
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False).fit(X, Y)
        return self
    def decision_function(self, X):
        return self.pls_.predict(X)
    def predict(self, X):
        return self.decision_function(X).argmax(1)
    def predict_proba(self, X):
        s = np.clip(self.decision_function(X), 1e-3, None)
        return s / s.sum(1, keepdims=True)


def make_models():
    pre = lambda: [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    models = {
        "plsda_raw8": (Pipeline(pre() + [("clf", PLSDA())]), {"clf__n_components": list(range(2, 9))}, "raw8"),
        "plsda_full": (Pipeline(pre() + [("clf", PLSDA())]), {"clf__n_components": list(range(2, 9))}, "full"),
        "logreg": (Pipeline(pre() + [("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))]),
                   {"clf__C": [0.01, 0.1, 1.0, 10.0]}, "full"),
        "rf": (Pipeline(pre() + [("clf", RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample",
                                                                 min_samples_leaf=2, n_jobs=N_JOBS, random_state=SEED))]),
               None, "full"),
        "lightgbm": (Pipeline(pre() + [("clf", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                                                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                                                                  colsample_bytree=0.7, reg_lambda=1.0, class_weight="balanced",
                                                                  n_jobs=N_JOBS, random_state=SEED, verbose=-1))]),
                     None, "full"),
        "svm_rbf": (Pipeline(pre() + [("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))]),
                    {"clf__C": [0.3, 1, 3, 10, 30], "clf__gamma": ["scale", 0.003, 0.01, 0.03]}, "full"),
    }
    return models


def fit_predict(pipe, grid, Xtr, ytr, gtr, Xte):
    """Hyper-parameters chosen by 3-fold fruit-grouped inner CV on the TRAINING folds only."""
    if grid:
        inner = GroupKFold(n_splits=3)
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=inner.split(Xtr, ytr, gtr), n_jobs=N_JOBS, refit=True)
        gs.fit(Xtr, ytr)
        est, best = gs.best_estimator_, gs.best_params_
    else:
        est, best = clone(pipe).fit(Xtr, ytr), {}
    return est.predict_proba(Xte), best, est


def run_cv(df, X, pipe, grid, split):
    oof = np.zeros((len(df), 3)); chosen = []
    for k, tr, te in dc.iter_folds(df, split):
        P, best, _ = fit_predict(pipe, grid, X[tr], df.y.values[tr], df.fruit_id.values[tr], X[te])
        oof[te] = P; chosen.append({"fold": k, **best})
    return oof, chosen


def run_lobo(df, X, pipe, grid):
    rows = []; oof = np.full((len(df), 3), np.nan)
    for b, tr, te in dc.leave_one_batch_out(df):
        P, best, _ = fit_predict(pipe, grid, X[tr], df.y.values[tr], df.fruit_id.values[tr], X[te])
        oof[te] = P
        sub = df.iloc[te]
        cap = dc.metrics_from_probs(sub.y.values, P)
        ys, Ps, _ = dc.aggregate_to_session(sub, P)
        ses = dc.metrics_from_probs(ys, Ps)
        rows.append(dict(held_out_batch=b, n_captures=cap["n"], n_sessions=ses["n"],
                         capture_acc=cap["accuracy"], capture_macro_f1=cap["macro_f1"],
                         session_acc=ses["accuracy"], session_macro_f1=ses["macro_f1"],
                         session_recall_unripe=ses["recall_per_class"][0], session_recall_ripe=ses["recall_per_class"][1],
                         session_recall_overripe=ses["recall_per_class"][2],
                         test_class_counts=json.dumps(sub.label_capture.value_counts().to_dict()), params=json.dumps(best)))
    return pd.DataFrame(rows), oof
