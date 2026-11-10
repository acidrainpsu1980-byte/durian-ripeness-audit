"""POST HOC (not pre-registered; computed after the round-7 evaluation): oracle-threshold headroom bound.
Each arm gets, in each held-out batch, the decision threshold that maximizes that batch's macro-F1 (Ripe vs Overripe), chosen with the
batch's own labels. This is an upper bound on what any label-free threshold rule could reach with the arm's ranking; it is not a method.
Thresholds: the batch's unique scores, Overripe if score >= t; the smallest maximizing threshold is used (as src/foundation/posthoc_shift.py).
Paired contrasts use evaluate.paired (fruit-cluster bootstrap, 2,000 resamples; permutation, 20,000; seed 42) with the oracle predictions held
fixed, so the intervals ignore the selection of the thresholds and are optimistic.
usage (repo root): python src/foundation/posthoc_oracle_headroom.py [OUT_DIR]
  -> OUT_DIR/posthoc_oracle_headroom.csv, posthoc_oracle_headroom_contrasts.csv (OUT_DIR default results/round7)
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("R5_EMB", "work/r5_embeddings")
import round5 as r
ev = r.ev; B = r.B
OUT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ROUND7_DIR", "results/round7"); os.makedirs(OUT, exist_ok=True)
R7 = os.environ.get("ROUND7_IN", "results/round7")
d, T, A, pop, lab, fb, ff = r.context()
y = T.loc[pop, "y"].values.astype(int); bat = T.loc[pop, "Maturity_Stage"].values; fruit = T.loc[pop, "fruit_id"].values.astype(int)
mf1 = lambda yy, pp: f1_score(yy, pp, labels=[1, 2], average="macro", zero_division=0)

# scores (higher = more Overripe) and the predictions as used, for the same 104 endpoint sessions
S, U = {}, {}
PS = pd.read_parquet(f"{r.R5}/probe_sessions.parquet")
for (m, sp), g in PS.groupby(["model", "split"]):
    if sp == "lobo":
        q = g.set_index("session_id").loc[pop, ["p_Ripe", "p_Overripe"]].values; S[m] = q[:, 1]; U[m] = 1 + q.argmax(1)
S["F1"] = np.mean([S[k] for k in ["A1", "V-thermal-DINO", "V-ms-DINO", "V-rgb-DINO"]], 0); U["F1"] = np.where(S["F1"] > 0.5, 2, 1)
L1 = pd.read_parquet(f"{r.V21}/lobo_cnn_sessions.parquet"); g = L1[L1.variant == "M0"].set_index("session_id").loc[pop]
S["CDR_RO"] = (g.p_Overripe / (g.p_Ripe + g.p_Overripe)).values; U["CDR_RO"] = 1 + g[["p_Ripe", "p_Overripe"]].values.argmax(1)
for n, paths in [("K1", [f"{r.R4}/cnn/current/paired_lobo_{b}.npy" for b in B]), ("A1ft", [f"{r.R5}/ft/beats_ft_lobo_{b}.npy" for b in B])]:
    K = r.r4.knock_sessions(paths).loc[pop]; S[n] = (K.p_Overripe / (K.p_Ripe + K.p_Overripe)).values; U[n] = 1 + K[["p_Ripe", "p_Overripe"]].values.argmax(1)
P6 = pd.read_parquet(f"{r.R6}/probe_sessions_with_intake.parquet"); U["R6"], _ = r._anchored(P6, "F3", pop, bat)   # R6 ranks as F3 within batch
RS = pd.read_parquet(f"{R7}/rta_sessions.parquet")
for m in ["R7", "R7-nc", "R7-ce", "R7-A", "R7-noRGB"]:
    E = RS[(RS.model == m) & (RS.split == "lobo") & (RS.role == "endpoint")].set_index("session_id").loc[pop]; S[m] = E.score.values; U[m] = E.pred.values.astype(int)
S["R6"] = S["F3"]
S6 = pd.read_csv(f"{r.R6}/scores.csv").query("split == 'lobo' and batch == 'pooled'").set_index("model").macro_f1_RO
S7 = pd.read_csv(f"{R7}/scores.csv").query("split == 'lobo' and batch == 'pooled'").set_index("model").macro_f1_RO
for m, ref in [("CDR_RO", S6), ("K1", S6), ("F3", S6), ("R6", S6), ("R7", S7), ("R7-ce", S7)]:
    assert abs(mf1(y, U[m]) - ref[m]) < 1e-9, m                                      # predictions as used reproduce the shipped scores

ARMS = ["CDR_RO", "K1", "A1", "A1ft", "A5", "F1", "F3", "R6", "R7", "R7-nc", "R7-ce", "R7-A", "R7-noRGB"]
O, rows = {}, []
for m in ARMS:
    p = np.zeros(len(y), int); row = dict(model=m, macro_f1_as_used=mf1(y, U[m]))
    for b in B:
        k = bat == b; sb = S[m][k]; ts = np.unique(sb); f = [mf1(y[k], np.where(sb >= t, 2, 1)) for t in ts]
        t = ts[int(np.argmax(f))]; p[k] = np.where(sb >= t, 2, 1)
        row[f"oracle_{b}"] = float(max(f)); row[f"as_used_{b}"] = mf1(y[k], U[m][k])
    O[m] = p; row["macro_f1_oracle_pooled"] = mf1(y, p); rows.append(row)
pd.DataFrame(rows).to_csv(f"{OUT}/posthoc_oracle_headroom.csv", index=False)
con = []
for cand, ref, kind in [("F3", "CDR_RO", "oracle vs as used"), ("F3", "CDR_RO", "oracle vs oracle"), ("F3", "R6", "oracle vs as used"),
                        ("R7", "CDR_RO", "oracle vs as used"), ("R7", "CDR_RO", "oracle vs oracle"), ("F1", "CDR_RO", "oracle vs oracle"),
                        ("A5", "CDR_RO", "oracle vs oracle"), ("K1", "CDR_RO", "oracle vs oracle")]:
    pr = U[ref] if kind == "oracle vs as used" else O[ref]
    c = ev.paired(y, pr, O[cand], fruit); c.update(candidate=f"{cand} (oracle)", reference=f"{ref} ({kind.split(' vs ')[1]})",
                                                   status="post hoc; thresholds held fixed across resamples (optimistic)"); con.append(c)
pd.DataFrame(con).to_csv(f"{OUT}/posthoc_oracle_headroom_contrasts.csv", index=False)
print("written: posthoc_oracle_headroom.csv, posthoc_oracle_headroom_contrasts.csv (POST HOC)")
