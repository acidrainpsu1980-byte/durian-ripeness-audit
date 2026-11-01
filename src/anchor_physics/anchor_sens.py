"""Pre-registered analysis (docs/ANALYSIS_PLAN_anchor_physics.md §6), anchor sensitivity (exploratory): M1 with 10, 20, 40 % and
all anchor fruits of the held-out batch (5 random draws each).

Only the held-out batch's standardization depends on the draw (source batches always use all their anchors), so each LOBO model is
fitted once per held-out batch exactly as lobo_classical.lobo M1 and re-applied with the subset statistics. Draws are fruit-level,
shared by every modality (OUT/anchor_draws.csv; n = max(2, ceil(f * N)) anchor fruits, numpy default_rng(42)); fraction 1.0 is M1
itself and is checked against OUT/lobo_classical_sessions.parquet.
usage (repo root): python src/anchor_physics/anchor_sens.py OUT draws|classical
"""
import os, sys, math
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import lobo_classical as lc

PC = lc.PC; CL = lc.CL; FRACS = [0.1, 0.2, 0.4]; NDRAW = 5; BATCHES = ["Immature", "Mature", "Overmature"]


def draws(out):
    A = pd.read_csv(f"{out}/anchor_sessions.csv"); A = A[A.anchor]
    fb = pd.read_csv("results/session_table.csv", low_memory=False).drop_duplicates("fruit_id").set_index("fruit_id").Maturity_Stage
    A["batch"] = A.fruit_id.map(fb); assert A.batch.notna().all()
    rng = np.random.default_rng(42); rows = []
    for b in BATCHES:
        fr = np.sort(A[A.batch == b].fruit_id.unique())
        for f in FRACS:
            n = max(2, math.ceil(f * len(fr)))
            for k in range(NDRAW):
                for x in rng.choice(fr, n, replace=False): rows.append(dict(batch=b, fraction=f, draw=k, fruit_id=int(x), n_anchor_fruits_batch=len(fr)))
    D = pd.DataFrame(rows); D.to_csv(f"{out}/anchor_draws.csv", index=False); print(D.groupby(["batch", "fraction", "draw"]).size().unstack().to_string())


def classical(out):
    A = pd.read_csv(f"{out}/anchor_sessions.csv"); anchor_sessions = set(A.session_id[A.anchor])
    D = pd.read_csv(f"{out}/anchor_draws.csv"); ref = pd.read_parquet(f"{out}/lobo_classical_sessions.parquet")
    rows = []
    for mod in ["thermal", "rgb", "physics"]:
        d, cols, fam = lc.load(mod)
        X = d[cols].astype(float).values; X = np.where(np.isfinite(X), X, np.nan)
        y = d.label_capture.map({c: i for i, c in enumerate(CL)}).values; batch = d.Maturity_Stage.values
        anc = d.session_id.isin(anchor_sessions).values
        Xs = lc.standardize(X, batch, anc)                                  # M1 for every batch (full anchors)
        for b in BATCHES:
            te = batch == b; tr = ~te
            med = np.nanmedian(Xs[tr], axis=0); Xtr = np.where(np.isnan(Xs[tr]), med, Xs[tr])
            est, grid = lc.estimator(fam)
            gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="f1_macro", n_jobs=1).fit(Xtr, y[tr], groups=d.fruit_id.values[tr])
            cfgs = [(1.0, 0, None)] + [(f, k, set(D[(D.batch == b) & (D.fraction == f) & (D.draw == k)].fruit_id)) for f in FRACS for k in range(NDRAW)]
            for f, k, fr in cfgs:
                m = anc if fr is None else (anc & d.fruit_id.isin(fr).values)
                Z = lc.standardize(X[te], batch[te], m[te]); Z = np.where(np.isnan(Z), med, Z)
                t = pd.DataFrame(gs.predict_proba(Z), columns=PC); t["session_id"] = d.session_id.values[te]
                s = t.groupby("session_id")[PC].mean().reset_index(); s["model"] = mod; s["fraction"] = f; s["draw"] = k; s["batch"] = b; rows.append(s)
            print(mod, b, flush=True)
    R = pd.concat(rows, ignore_index=True)
    # control: fraction 1.0 equals the M1 LOBO predictions of lobo_classical.py
    for mod in ["thermal", "rgb", "physics"]:
        a = R[(R.model == mod) & (R.fraction == 1.0)].set_index("session_id")[PC].sort_index()
        r = ref[(ref.model == mod) & (ref.variant == "M1")].set_index("session_id")[PC].loc[a.index]
        dm = float(np.abs(a.values - r.values).max()); print("control", mod, "fraction 1.0 vs M1 max|diff|", dm); assert dm < 1e-9, mod
    R.to_parquet(f"{out}/anchor_sens_classical.parquet", index=False)


if __name__ == "__main__":
    OUT = sys.argv[1]; lc.OUT = OUT
    {"draws": draws, "classical": classical}[sys.argv[2]](OUT)
