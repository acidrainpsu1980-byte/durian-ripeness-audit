"""Round 2 (docs/ANALYSIS_PLAN_round2.md, tag prereg-round2): protocol-aware adaptation, nested leave-one-batch-out selection.

  python src/anchor_physics/round2.py classical      single-batch (inner) fits of thermal, RGB probe and peak+centroid+MFCC LR
  python src/anchor_physics/round2.py evaluate       candidates C1-C5, nested selection, contrasts R2-1/R2-2, controls, audit
Inputs: results/v21 (anchors, round-1 M0 LOBO session probabilities, CDR-RO reference), results/round2/cnn_sessions.parquet
(src/anchor_physics/round2_cnn.py). Outputs in results/round2/.
"""
import os, sys, json, re
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import lobo_classical as lc, baselines as bl, evaluate as ev
V21 = os.environ.get("ROUND_V21", "results/v21"); R2 = os.environ.get("ROUND2_DIR", "results/round2"); lc.OUT = V21   # v2.0: path overrides for run_all.sh
PC = ev.PC; B = ev.BATCHES; CANDS = ["C1", "C2", "C3", "C4", "C5"]


def classical():
    """Inner fits: each model trained on one harvest batch (same family, grid, inner fruit-grouped 3-fold CV as round 1) and applied
    to each other batch. Session probability = mean over captures."""
    rows = []
    for mod, loader in [("thermal", lc.load), ("rgb", lc.load), ("phys_pcm", bl.load)]:
        d, cols, fam = loader(mod)
        X = d[cols].astype(float).values; X = np.where(np.isfinite(X), X, np.nan)
        y = d.label_capture.map({c: i for i, c in enumerate(lc.CL)}).values; batch = d.Maturity_Stage.values
        for tr_b in B:
            tr = batch == tr_b; med = np.nanmedian(X[tr], axis=0); Xtr = np.where(np.isnan(X[tr]), med, X[tr])
            est, grid = lc.estimator(fam)
            gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="f1_macro", n_jobs=1).fit(Xtr, y[tr], groups=d.fruit_id.values[tr])
            assert list(gs.classes_) == [0, 1, 2], (mod, tr_b, gs.classes_)
            for te_b in [b for b in B if b != tr_b]:
                te = batch == te_b; Xte = np.where(np.isnan(X[te]), med, X[te])
                t = pd.DataFrame(gs.predict_proba(Xte), columns=PC); t["session_id"] = d.session_id.values[te]
                s = t.groupby("session_id")[PC].mean().reset_index(); s["model"] = mod; s["train"] = tr_b; s["test"] = te_b; rows.append(s)
            print(mod, tr_b, flush=True)
    os.makedirs(R2, exist_ok=True); pd.concat(rows, ignore_index=True).to_parquet(f"{R2}/inner_classical_sessions.parquet", index=False)


def gate(P, anchor):
    p = 1 + P[:, 1:].argmax(1); p[anchor] = 0; return p


def ro(P):
    q = P[:, 1:]; return q / q.sum(1, keepdims=True)


def candidates(cnn_tri, cnn_bin, thermal, rgb, pcm, anchor):
    """3-class probability arrays (sessions x 3) -> dict candidate -> predicted class."""
    q4 = (ro(cnn_tri) + ro(pcm)) / 2
    return {"C1": cnn_tri.argmax(1), "C2": gate(cnn_tri, anchor), "C3": gate(cnn_bin, anchor),
            "C4": gate(np.concatenate([np.zeros((len(q4), 1)), q4], 1), anchor), "C5": gate((thermal + cnn_tri + rgb) / 3, anchor)}


def evaluate():
    T = pd.read_csv("results/session_table.csv", low_memory=False); T = T[T.n_modalities == 4].sort_values("session_id").set_index("session_id")
    A = pd.read_csv(f"{V21}/anchor_sessions.csv").set_index("session_id")
    T["anchor"] = A.loc[T.index, "anchor"].astype(bool); assert T.anchor.sum() == 162
    CN = pd.read_parquet(f"{R2}/cnn_sessions.parquet"); IC = pd.read_parquet(f"{R2}/inner_classical_sessions.parquet")
    OC = pd.concat([pd.read_parquet(f"{V21}/lobo_classical_sessions.parquet"), pd.read_parquet(f"{V21}/lobo_baselines_sessions.parquet")])
    OC = OC[OC.variant == "M0"]

    def probs(train, test):
        sid = T.index[T.Maturity_Stage == test]
        g = lambda base: CN[(CN.base == base) & (CN.train == train) & (CN.test == test)].set_index("session_id").loc[sid, PC].values
        if train == "lobo":
            c = lambda m: OC[OC.model == m].set_index("session_id").loc[sid, PC].values
        else:
            c = lambda m: IC[(IC.model == m) & (IC.train == train) & (IC.test == test)].set_index("session_id").loc[sid, PC].values
        return sid, candidates(g("tri"), g("bin"), c("thermal"), c("rgb"), c("phys_pcm"), T.loc[sid, "anchor"].values)

    # outer predictions of every candidate (all 268 sessions)
    outer = {k: pd.Series(0, index=T.index) for k in CANDS}
    for b in B:
        sid, pr = probs("lobo", b)
        for k in CANDS: outer[k].loc[sid] = pr[k]
    # nested selection
    sel_rows, inner_rows = [], []; proc = pd.Series(-1, index=T.index)
    for b in B:
        src = [x for x in B if x != b]; ys, ps = [], {k: [] for k in CANDS}
        for tr, te in [(src[0], src[1]), (src[1], src[0])]:
            sid, pr = probs(tr, te); ep = ~T.loc[sid, "anchor"].values; ys.append(T.loc[sid, "y"].values[ep].astype(int))
            for k in CANDS: ps[k].append(pr[k][ep])
            for k in CANDS:
                inner_rows.append(dict(held_out=b, inner_train=tr, inner_test=te, candidate=k, **ev.endpoint_scores(ys[-1], pr[k][ep])))
        yy = np.concatenate(ys); sc = {k: ev.endpoint_scores(yy, np.concatenate(ps[k]))["macro_f1_RO"] for k in CANDS}
        best = max(CANDS, key=lambda k: (sc[k], -CANDS.index(k)))
        for k in CANDS: inner_rows.append(dict(held_out=b, inner_train="pooled", inner_test="pooled", candidate=k, n=len(yy), macro_f1_RO=sc[k]))
        sel_rows.append(dict(held_out=b, selected=best, **{f"inner_{k}": sc[k] for k in CANDS}))
        m = T.Maturity_Stage == b; proc[m] = outer[best][m]
    assert (proc >= 0).all()
    pd.DataFrame(sel_rows).to_csv(f"{R2}/selection.csv", index=False); pd.DataFrame(inner_rows).to_csv(f"{R2}/inner_scores.csv", index=False)

    # references (round 1): CDR-RO = gated M0 CNN; M5 = C1
    L1 = pd.read_parquet(f"{V21}/lobo_cnn_sessions.parquet"); M0 = L1[L1.variant == "M0"].set_index("session_id").loc[T.index, PC].values
    cdr = gate(M0, T.anchor.values)
    S1 = pd.read_csv(f"{V21}/scores.csv"); r1 = S1[(S1.model == "sound_cnn") & (S1.variant == "CDR_RO") & (S1.scope == "endpoint") & (S1.batch == "pooled")].macro_f1_RO.iloc[0]
    ep = ~T.anchor.values; y = T.y.values.astype(int); fruit = T.fruit_id.values.astype(int); bat = T.Maturity_Stage.values
    assert abs(ev.endpoint_scores(y[ep], cdr[ep])["macro_f1_RO"] - r1) < 1e-12                     # control: CDR-RO reproduced
    L5 = L1[L1.variant == "M5"].set_index("session_id").loc[T.index, PC].values
    assert (outer["C1"].values == L5.argmax(1)).all()                                               # control: C1 outer = M5
    m5 = outer["C1"].values
    preds = {"procedure": proc.values, "CDR_RO": cdr, "M5": m5, **{k: outer[k].values for k in CANDS}}

    rows = []
    for name, p in preds.items():
        rows.append(dict(rule=name, batch="pooled", **ev.endpoint_scores(y[ep], p[ep])))
        for b in B:
            k = ep & (bat == b); rows.append(dict(rule=name, batch=b, **ev.endpoint_scores(y[k], p[k])))
        rows.append(dict(rule=name, batch="all_268", **ev.all_scores(y, p)))
    pd.DataFrame(rows).to_csv(f"{R2}/scores.csv", index=False)

    prim = []
    for cname, ref in [("R2-1", "CDR_RO"), ("R2-2", "M5")]:
        r = ev.paired(y[ep], preds[ref][ep], proc.values[ep], fruit[ep]); r.update(contrast=cname, candidate="procedure", reference=ref); prim.append(r)
    PR = pd.DataFrame(prim); PR["p_holm"] = ev.holm(PR.p_perm.values)
    r21 = PR[PR.contrast == "R2-1"].iloc[0]
    PR["beneficial_vs_CDR_RO"] = bool((r21.d_f1_pp >= 3) & (r21.ci_f1_lo_pp > 0) & (r21.p_holm < 0.05))
    PR.to_csv(f"{R2}/primary_contrasts.csv", index=False)
    sec = []
    for k in CANDS:
        for ref in ["CDR_RO", "M5"]:
            if k == "C1" and ref == "M5": continue
            r = ev.paired(y[ep], preds[ref][ep], preds[k][ep], fruit[ep]); r.update(candidate=k, reference=ref); sec.append(r)
    pd.DataFrame(sec).to_csv(f"{R2}/secondary_contrasts.csv", index=False)

    # leakage audit (code-level): AdaBN references and gating never read labels; selection never reads the outer batch
    src = open(os.path.join(HERE, "round2_cnn.py")).read(); blk = src[src.index("rows = []; ctrl"):src.index("S = pd.concat")]
    au = {"adabn_reference_uses_labels": bool(re.search(r"\byk\b|label|\.y\b", blk))}
    me = open(__file__).read(); g = me[me.index("def gate"):me.index("def evaluate")]
    au["gating_uses_labels"] = bool(re.search(r"\by\b|label", g))
    s2 = me[me.index("    # nested selection"):me.index("    assert (proc >= 0).all()")]
    au["selection_reads_outer_batch_scores"] = bool(re.search(r'probs\("lobo"', s2.split("m = T.Maturity_Stage == b")[0]))
    assert not any(au.values()), au
    json.dump(au, open(f"{R2}/leakage_audit.json", "w"), indent=1)
    print("written: selection, inner_scores, scores, primary_contrasts, secondary_contrasts, leakage_audit")


if __name__ == "__main__":
    {"classical": classical, "evaluate": evaluate}[sys.argv[1]]()
