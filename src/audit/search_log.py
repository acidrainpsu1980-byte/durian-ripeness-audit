"""v1.8: full inner hyper-parameter search log of every grid-searched candidate on the primary partition (p0).

For each grid-searched candidate (cands.py) and each outer fold k of p0, the GridSearchCV is refitted on the outer training set
exactly as in run_fits.py, and every grid point's inner fruit-grouped CV scores are written out, with the chosen point flagged.
The outer-test predictions of the refitted search are compared with the shipped intermediate_v14/fits/p0 arrays (column
outer_pred_max_abs_diff) to show the logged search is the one behind the published predictions.
usage (repo root):  DURIAN_INTERMEDIATE=intermediate_common EMB_MOBILENET=intermediate_common/rgb/emb_mobilenetv2.npy \
                    OMP_NUM_THREADS=4 python src/audit/search_log.py          (~4 min)
writes: audit/hyperparameter_search_p0.csv
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src", "nested_selection"))
os.environ.setdefault("DURIAN_INTERMEDIATE", "intermediate_common")
import cands
from joblib import parallel_backend
from sklearn.model_selection import GridSearchCV
AOUT = os.environ.get("DURIAN_AUDIT_OUT", "audit")
part = pd.read_csv("src/nested_selection/partitions.csv").set_index("fruit_id")["p0"]
rows = []
for mod, cl in cands.CANDIDATES.items():
    meta = X = None
    for c in cl:
        if c == "cnn" or not isinstance(cands._FACT[mod](c)[1], GridSearchCV):
            continue
        if meta is None:
            meta, X = cands.load(mod)
        fold = meta.fruit_id.map(part).values.astype(int)
        ref = np.load(f"intermediate_v14/fits/p0/{mod}__{c}.npz", allow_pickle=True)["outer"]
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            xs, gs = cands._FACT[mod](c); Xm = X[xs]
            with parallel_backend("threading", n_jobs=cands.NJ):
                gs.fit(Xm[tr], meta.y.values[tr], groups=meta.fruit_id.values[tr])
            d = float(np.abs(gs.predict_proba(Xm[te]) - ref[te]).max())
            cv = gs.cv_results_
            for i, p in enumerate(cv["params"]):
                rows.append(dict(sensor=mod, candidate=c, outer_fold=k, params=json.dumps({q.split("__")[-1]: v for q, v in p.items()}),
                                 inner_splits=gs.cv.n_splits, scoring=gs.scoring,
                                 **{f"split{j}_score": cv[f"split{j}_test_score"][i] for j in range(gs.cv.n_splits)},
                                 mean_inner_score=cv["mean_test_score"][i], rank=int(cv["rank_test_score"][i]),
                                 chosen=(i == gs.best_index_), outer_pred_max_abs_diff=d))
        print(mod, c, "logged", flush=True)
L = pd.DataFrame(rows); L.to_csv(f"{AOUT}/hyperparameter_search_p0.csv", index=False)
print(len(L), "grid-point rows;", L[L.chosen].groupby(["sensor", "candidate"]).size().to_dict(), "| max outer diff", L.outer_pred_max_abs_diff.max())
