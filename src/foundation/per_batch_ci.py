"""POST HOC (after the round-6 evaluation; not pre-registered): per-batch uncertainty for main-text Table 6.
Rebuilds the leave-one-batch-out endpoint predictions of the round-6 evaluation (round5.py evaluate6, same functions and
inputs) and reports, per held-out batch and pooled, endpoint macro-F1 (Ripe/Overripe) with a fruit-cluster percentile
bootstrap 95% CI (2,000 resamples of the batch's fruits, redrawn if one class is absent; seed 20261105), and the paired
difference of the anchored fusion (R6) against the capture-day rule (CDR_RO) on the same resamples.
usage (repo root): python src/foundation/per_batch_ci.py  -> results/round6/per_batch_ci.csv
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import round5 as r
ev = r.ev
d, T, A, pop, lab, fb, ff = r.context()
y = T.loc[pop, "y"].values.astype(int); fruit = T.loc[pop, "fruit_id"].values.astype(int); bat = T.loc[pop, "Maturity_Stage"].values
P6 = pd.read_parquet(f"{r.R6}/probe_sessions_with_intake.parquet"); P6 = pd.concat([P6, r._f1_rows(P6)], ignore_index=True)
E = P6[(P6.role == "endpoint") & (P6.split == "lobo")]; prob = {}
for m in r.R6_ARMS + ["F1"]:
    prob[m] = E[E.model == m].set_index("session_id").loc[pop, ["p_Ripe", "p_Overripe"]].values
preds = {m: 1 + prob[m].argmax(1) for m in prob}
preds.update(r._references_lobo(pop, y))
preds["R6"], _ = r._anchored(P6, "F3", pop, bat)
S6 = pd.read_csv(f"{r.R6}/scores.csv") if os.path.exists(f"{r.R6}/scores.csv") else None
ARMS = [a for a in ["CDR_RO", "K1", "A1", "A5", "F1", "F3", "R6"] if a in preds]
f1 = lambda yy, pp: ev.endpoint_scores(yy, pp)["macro_f1_RO"]
rows = []
for b in list(r.B) + ["pooled"]:
    k = np.arange(len(y)) if b == "pooled" else np.where(bat == b)[0]
    fr = np.unique(fruit[k]); idx = {f: k[fruit[k] == f] for f in fr}
    rng = np.random.default_rng(20261105); draws = []
    while len(draws) < 2000:
        s = np.concatenate([idx[f] for f in rng.choice(fr, len(fr), replace=True)])
        if len(np.unique(y[s])) == 2: draws.append(s)
    for m in ARMS:
        bs = np.array([f1(y[s], preds[m][s]) for s in draws])
        rows.append(dict(batch=b, arm=m, n=len(k), n_ripe=int((y[k] == 1).sum()), n_overripe=int((y[k] == 2).sum()), n_fruits=len(fr),
                         macro_f1_RO=f1(y[k], preds[m][k]), ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5))))
    bs = np.array([f1(y[s], preds["R6"][s]) - f1(y[s], preds["CDR_RO"][s]) for s in draws])
    rows.append(dict(batch=b, arm="R6 - CDR_RO", n=len(k), n_ripe=int((y[k] == 1).sum()), n_overripe=int((y[k] == 2).sum()), n_fruits=len(fr),
                     macro_f1_RO=f1(y[k], preds["R6"][k]) - f1(y[k], preds["CDR_RO"][k]), ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5))))
D = pd.DataFrame(rows); D.to_csv(f"{r.R6}/per_batch_ci.csv", index=False)
print(D.round(3).to_string(index=False))
