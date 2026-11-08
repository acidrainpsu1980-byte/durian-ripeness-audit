"""POST HOC (not pre-registered): uncertainty of the within-batch ranking diagnostic of manuscript Table 7.
For every arm of src/foundation/posthoc_shift.py and every held-out batch: AUC of P(Overripe) between Ripe and Overripe endpoint
sessions with a fruit-cluster percentile bootstrap 95% CI (2,000 resamples of the batch's fruits; resamples containing one class only
are redrawn; seed 20261103). Same predictions and session population as posthoc_shift.py.
Paired differences (same resamples) for embedding arms versus the task-specific knock CNNs.
usage (repo root): python src/foundation/posthoc_auc_ci.py   -> results/round5/posthoc_auc_ci.csv, posthoc_auc_diff.csv
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("R5_EMB", "work/r5_embeddings")
import round5 as r
R5 = r.R5; R4 = r.R4; V21 = r.V21
d, T, A, pop, lab, fb, ff = r.context()
y = T.loc[pop, "y"].values.astype(int); bat = T.loc[pop, "Maturity_Stage"].values; fruit = T.loc[pop, "fruit_id"].values.astype(int)
PS = pd.read_parquet(f"{R5}/probe_sessions.parquet"); prob = {}
for (m, sp), g in PS.groupby(["model", "split"]):
    if sp == "lobo": prob[m] = g.set_index("session_id").loc[pop, "p_Overripe"].values
L1 = pd.read_parquet(f"{V21}/lobo_cnn_sessions.parquet"); g = L1[L1.variant == "M0"].set_index("session_id").loc[pop]
prob["CDR_RO"] = (g.p_Overripe / (g.p_Ripe + g.p_Overripe)).values
for n, paths in [("K1", [f"{R4}/cnn/current/paired_lobo_{b}.npy" for b in r.B]), ("A1ft", [f"{R5}/ft/beats_ft_lobo_{b}.npy" for b in r.B])]:
    S = r.r4.knock_sessions(paths).loc[pop]; prob[n] = (S.p_Overripe / (S.p_Ripe + S.p_Overripe)).values
prob["F1"] = np.mean([prob[k] for k in ["A1", "V-thermal-DINO", "V-ms-DINO", "V-rgb-DINO"]], 0)
ARMS = ["CDR_RO", "K1", "A1", "A1ft", "A1c", "A5", "F1", "F3", "V-thermal-DINO", "V-ms-DINO", "V-rgb-DINO"]
NB = 2000; rows = []; drows = []
PAIRS = [("F3", "CDR_RO"), ("F1", "CDR_RO"), ("A1", "CDR_RO"), ("A1", "K1"), ("F3", "K1")]
for b in r.B:
    k = np.where(bat == b)[0]; fr = np.unique(fruit[k]); idx = {f: k[fruit[k] == f] for f in fr}
    rng = np.random.default_rng(20261103); draws = []
    while len(draws) < NB:
        s = np.concatenate([idx[f] for f in rng.choice(fr, len(fr), replace=True)])
        if len(np.unique(y[s])) == 2: draws.append(s)
    for m in ARMS:
        p = prob[m]; yy = (y == 2).astype(int)
        bs = np.array([roc_auc_score(yy[s], p[s]) for s in draws])
        rows.append(dict(model=m, batch=b, n=len(k), n_fruits=len(fr), auc_overripe=roc_auc_score(yy[k], p[k]),
                         auc_ci_lo=float(np.percentile(bs, 2.5)), auc_ci_hi=float(np.percentile(bs, 97.5)), n_boot=NB))
    yy = (y == 2).astype(int)
    for m1, m0 in PAIRS:
        bs = np.array([roc_auc_score(yy[s], prob[m1][s]) - roc_auc_score(yy[s], prob[m0][s]) for s in draws])
        drows.append(dict(arm=m1, reference=m0, batch=b, d_auc=roc_auc_score(yy[k], prob[m1][k]) - roc_auc_score(yy[k], prob[m0][k]),
                          ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)), n_boot=NB))
pd.DataFrame(rows).to_csv(f"{R5}/posthoc_auc_ci.csv", index=False)
pd.DataFrame(drows).to_csv(f"{R5}/posthoc_auc_diff.csv", index=False)
print(pd.DataFrame(drows).round(3).to_string(index=False))
