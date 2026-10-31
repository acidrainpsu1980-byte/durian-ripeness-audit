"""Classical models on per-recording features. Expects globals: df, X (DataFrame of features), dc.
Defines run_model(name, split, df_sub=None, X_sub=None) -> oof_proba."""
import numpy as np, pandas as pd, json, os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
import lightgbm as lgb

SEED = 42
NJ = 4


def make_model(name, y_train=None, groups=None, Xtr=None):
    """Fixed defaults, except kNN-k and SVM-C chosen by inner fruit-grouped CV on the training fold."""
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced", random_state=SEED))
    if name == "knn":
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(weights="distance"))
        gs = GridSearchCV(pipe, {"kneighborsclassifier__n_neighbors": [3, 5, 7, 11, 15, 21]},
                          cv=GroupKFold(4), scoring="f1_macro", n_jobs=NJ)
        gs.fit(Xtr, y_train, groups=groups)
        return gs.best_estimator_
    if name == "rf":
        return make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                                                      n_jobs=NJ, random_state=SEED))
    if name == "svm":
        pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED))
        gs = GridSearchCV(pipe, {"svc__C": [0.5, 1, 2, 5, 10], "svc__gamma": ["scale"]},
                          cv=GroupKFold(4), scoring="f1_macro", n_jobs=NJ)
        gs.fit(Xtr, y_train, groups=groups)
        return gs.best_estimator_
    if name == "lgbm":
        return make_pipeline(StandardScaler(), lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, max_depth=5,
                                                                  min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                                                  reg_lambda=1.0, class_weight="balanced", n_jobs=NJ, random_state=SEED, verbose=-1))
    raise ValueError(name)


def fit_predict(name, Xtr, ytr, gtr, Xte):
    m = make_model(name, ytr, gtr, Xtr)
    if name in ("knn", "svm"):
        pass  # already fitted by GridSearchCV refit
    else:
        m.fit(Xtr, ytr)
    return m, m.predict_proba(Xte)


def run_oof(name, d, Xm, split):
    import durian_common as dc
    oof = np.zeros((len(d), 3))
    for k, tr, te in dc.iter_folds(d, split):
        _, p = fit_predict(name, Xm[tr], d.y.values[tr], d.fruit_id.values[tr], Xm[te])
        oof[te] = p
    return oof
