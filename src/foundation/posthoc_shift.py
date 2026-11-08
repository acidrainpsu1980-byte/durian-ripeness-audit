"""POST HOC (not pre-registered; computed after the round-5 evaluation, before round 6 was planned): ranking versus thresholding within
each held-out batch, manuscript Table 7 and results/round5/POSTHOC_shift_diagnostic.md.
Per arm and held-out batch: AUC of P(Overripe) between Ripe and Overripe endpoint sessions, macro-F1 at the fixed 0.5 boundary, macro-F1
at the best threshold chosen with the batch's labels (an upper bound, not a method), mean P(Overripe); and pooled macro-F1 of label-free
score-shift rules (median-centering, training-prior quantile, prior-shift EM) applied within the held-out batch.
usage (repo root): python src/foundation/posthoc_shift.py   -> results/round5/posthoc_shift_diagnostic.csv, posthoc_shift_rules.csv
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("R5_EMB", "work/r5_embeddings")
import round5 as r, evaluate as ev
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
mf1 = lambda yy, pp: f1_score(yy, pp, labels=[1, 2], average="macro", zero_division=0)
rows = []
for m in ARMS:
    p = prob[m]
    for b in r.B:
        k = bat == b; pb = p[k]; yb = y[k]
        rows.append(dict(model=m, batch=b, n=int(k.sum()), n_ripe=int((yb == 1).sum()), n_overripe=int((yb == 2).sum()),
                         auc_overripe=roc_auc_score((yb == 2).astype(int), pb), macro_f1_at_0p5=mf1(yb, np.where(pb >= 0.5, 2, 1)),
                         macro_f1_oracle_threshold=max(mf1(yb, np.where(pb >= t, 2, 1)) for t in np.unique(pb)), mean_p_overripe=float(pb.mean())))
pd.DataFrame(rows).to_csv(f"{R5}/posthoc_shift_diagnostic.csv", index=False)
q = {b: float((y[bat != b] == 2).mean()) for b in r.B}


def rule(p, kind):
    out = np.empty(len(p), int)
    for b in r.B:
        k = bat == b; pb = p[k]
        if kind == "em":
            ps = np.array([1 - q[b], q[b]]); P = np.c_[1 - pb, pb]; w = ps.copy()
            for _ in range(100):
                Q = P * (w / ps); Q /= Q.sum(1, keepdims=True); w2 = Q.mean(0)
                if abs(w2 - w).max() < 1e-8: break
                w = w2
            Q = P * (w / ps); Q /= Q.sum(1, keepdims=True); out[k] = np.where(Q[:, 1] >= 0.5, 2, 1); continue
        thr = {"argmax": 0.5, "median": np.median(pb), "trainprior_quantile": np.quantile(pb, 1 - q[b])}[kind]
        out[k] = np.where(pb >= thr, 2, 1)
    return out


ref = rule(prob["CDR_RO"], "argmax"); rows = []
for m in ["CDR_RO", "A1", "A1ft", "A5", "F1", "F3", "V-ms-DINO"]:
    for kind in ["argmax", "median", "trainprior_quantile", "em"]:
        p = rule(prob[m], kind); c = ev.paired(y, ref, p, fruit)
        rows.append(dict(model=m, rule=kind, pooled_macro_f1=mf1(y, p), **{f"macro_f1_{b}": mf1(y[bat == b], p[bat == b]) for b in r.B},
                         overmature_ripe_recall=float(np.mean(p[(bat == "Overmature") & (y == 1)] == 1)), d_f1_pp_vs_CDR_RO=c["d_f1_pp"], ci_lo=c["ci_f1_lo_pp"], ci_hi=c["ci_f1_hi_pp"]))
pd.DataFrame(rows).to_csv(f"{R5}/posthoc_shift_rules.csv", index=False)
print("written: posthoc_shift_diagnostic.csv, posthoc_shift_rules.csv (POST HOC)")
