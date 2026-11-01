"""Round 3, final (docs/ANALYSIS_PLAN_round3.md, tag prereg-round3): label-free Ripe/Overripe prior adaptation on the target batch's
endpoint sessions, nested leave-one-batch-out selection over {C2, C4, D1, D2, D3, D4}. No model is trained; inputs are the round-2
session probabilities (results/round2) and the round-1 M0 LOBO probabilities (results/v21).
usage (repo root): python src/anchor_physics/round3.py [selftest|evaluate]
"""
import os, sys, json, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import evaluate as ev, round2 as r2
V21, R2, R3 = "results/v21", "results/round2", "results/round3"
PC = ev.PC; B = ev.BATCHES; CANDS = ["C2", "C4", "D1", "D2", "D3", "D4"]


def saerens2(Q, prior_src=np.array([0.5, 0.5]), n_iter=100, tol=1e-8):
    """EM re-estimation of the Ripe/Overripe prior (Saerens et al. 2002), as round 1; returns adjusted Q and the estimated prior."""
    w = prior_src.copy()
    for _ in range(n_iter):
        A = Q * (w / prior_src); A /= A.sum(1, keepdims=True); w_new = A.mean(0)
        if np.abs(w_new - w).max() < tol: w = w_new; break
        w = w_new
    A = Q * (w / prior_src); return A / A.sum(1, keepdims=True), w


def quantile(Q, pi_o, sid):
    """Overripe for the top round(pi_o * n) sessions by log P(O) - log P(R); ties broken by session_id (ascending first)."""
    s = np.log(Q[:, 1] + 1e-12) - np.log(Q[:, 0] + 1e-12); n = len(s); k = int(round(pi_o * n))
    o = np.lexsort((np.asarray(sid, str), -s)); out = np.zeros(n, int); out[o[:k]] = 1; return out


def rules(q2, q4, pi_o, ep_sid):
    """q2, q4: (n_endpoints x 2) renormalized Ripe/Overripe probabilities -> dict rule -> 1 (Ripe) / 2 (Overripe); plus EM priors."""
    e1, w1 = saerens2(q4); e3, w3 = saerens2(q2)
    out = {"C2": 1 + q2.argmax(1), "C4": 1 + q4.argmax(1), "D1": 1 + e1.argmax(1), "D2": 1 + quantile(q4, pi_o, ep_sid),
           "D3": 1 + e3.argmax(1), "D4": 1 + quantile(q2, pi_o, ep_sid)}
    return out, {"D1_prior_O": float(w1[1]), "D3_prior_O": float(w3[1])}


def selftest():
    rng = np.random.default_rng(0)
    for n in [5, 31, 41]:
        for pi in [0.0, 0.3, 0.46, 0.5, 1.0]:
            Q = rng.dirichlet([1, 1], n); q = quantile(Q, pi, [f"s{i:03d}" for i in range(n)]); assert q.sum() == int(round(pi * n)), (n, pi)
    Q = np.full((4, 2), 0.5); assert list(quantile(Q, 0.5, ["d", "c", "b", "a"])) == [0, 0, 1, 1]          # ties -> session_id order
    A, w = saerens2(np.array([[0.9, 0.1]] * 3 + [[0.2, 0.8]])); assert abs(A.sum(1) - 1).max() < 1e-12 and w.sum() > 0.999
    print("selftest ok")


def evaluate():
    T = pd.read_csv("results/session_table.csv", low_memory=False); T = T[T.n_modalities == 4].sort_values("session_id").set_index("session_id")
    A = pd.read_csv(f"{V21}/anchor_sessions.csv").set_index("session_id"); T["anchor"] = A.loc[T.index, "anchor"].astype(bool)
    CN = pd.read_parquet(f"{R2}/cnn_sessions.parquet"); IC = pd.read_parquet(f"{R2}/inner_classical_sessions.parquet")
    OC = pd.concat([pd.read_parquet(f"{V21}/lobo_classical_sessions.parquet"), pd.read_parquet(f"{V21}/lobo_baselines_sessions.parquet")]); OC = OC[OC.variant == "M0"]
    EPT = T[~T.anchor]

    def pi_o(batches):                                   # Overripe share among the training batches' endpoint sessions (training labels)
        e = EPT[EPT.Maturity_Stage.isin(batches)]; assert set(e.y.astype(int)) <= {1, 2}; return float((e.y.astype(int) == 2).mean())

    def preds(train, test, train_batches):
        sid = T.index[T.Maturity_Stage == test]; anc = T.loc[sid, "anchor"].values
        tri = CN[(CN.base == "tri") & (CN.train == train) & (CN.test == test)].set_index("session_id").loc[sid, PC].values
        if train == "lobo": pcm = OC[OC.model == "phys_pcm"].set_index("session_id").loc[sid, PC].values
        else: pcm = IC[(IC.model == "phys_pcm") & (IC.train == train) & (IC.test == test)].set_index("session_id").loc[sid, PC].values
        q2 = r2.ro(tri); q4 = (r2.ro(tri) + r2.ro(pcm)) / 2
        R, pri = rules(q2[~anc], q4[~anc], pi_o(train_batches), sid[~anc])
        full = {}
        for k, v in R.items(): p = np.zeros(len(sid), int); p[~anc] = v; full[k] = p
        # control: C2 and C4 equal the round-2 candidate predictions computed from the same inputs
        th = np.full_like(tri, 1 / 3); c = r2.candidates(tri, tri, th, th, pcm, anc)
        assert (c["C2"] == full["C2"]).all() and (c["C4"] == full["C4"]).all()
        return sid, full, pri

    outer = {k: pd.Series(-1, index=T.index) for k in CANDS}; priors = []
    for b in B:
        src = [x for x in B if x != b]; sid, pr, pri = preds("lobo", b, src)
        for k in CANDS: outer[k].loc[sid] = pr[k]
        priors.append(dict(held_out=b, split="outer", pi_o_source=pi_o(src), **pri))
    sel_rows, inner_rows = [], []; proc = pd.Series(-1, index=T.index)
    for b in B:
        src = [x for x in B if x != b]; ys, ps = [], {k: [] for k in CANDS}
        for tr, te in [(src[0], src[1]), (src[1], src[0])]:
            sid, pr, pri = preds(tr, te, [tr]); ep = ~T.loc[sid, "anchor"].values; ys.append(T.loc[sid, "y"].values[ep].astype(int))
            priors.append(dict(held_out=b, split=f"inner {tr}>{te}", pi_o_source=pi_o([tr]), **pri))
            for k in CANDS:
                ps[k].append(pr[k][ep]); inner_rows.append(dict(held_out=b, inner_train=tr, inner_test=te, candidate=k, **ev.endpoint_scores(ys[-1], pr[k][ep])))
        yy = np.concatenate(ys); sc = {k: ev.endpoint_scores(yy, np.concatenate(ps[k]))["macro_f1_RO"] for k in CANDS}
        best = max(CANDS, key=lambda k: (sc[k], -CANDS.index(k)))
        sel_rows.append(dict(held_out=b, selected=best, **{f"inner_{k}": sc[k] for k in CANDS}))
        m = T.Maturity_Stage == b; proc[m] = outer[best][m]
    assert (proc >= 0).all() and all((v >= 0).all() for v in outer.values())
    os.makedirs(R3, exist_ok=True)
    pd.DataFrame(sel_rows).to_csv(f"{R3}/selection.csv", index=False); pd.DataFrame(inner_rows).to_csv(f"{R3}/inner_scores.csv", index=False)
    pd.DataFrame(priors).to_csv(f"{R3}/priors.csv", index=False)

    ep = ~T.anchor.values; y = T.y.values.astype(int); fruit = T.fruit_id.values.astype(int); bat = T.Maturity_Stage.values
    L1 = pd.read_parquet(f"{V21}/lobo_cnn_sessions.parquet"); M0 = L1[L1.variant == "M0"].set_index("session_id").loc[T.index, PC].values
    cdr = r2.gate(M0, T.anchor.values)
    sel2 = pd.read_csv(f"{R2}/selection.csv").set_index("held_out").selected; p2 = pd.Series(-1, index=T.index)
    for b in B: m = T.Maturity_Stage == b; p2[m] = outer[sel2[b]][m] if sel2[b] in outer else -9
    assert (p2 >= 0).all(), "round-2 selection outside {C2, C4}"
    S2 = pd.read_csv(f"{R2}/scores.csv"); ref2 = S2[(S2.rule == "procedure") & (S2.batch == "pooled")].macro_f1_RO.iloc[0]
    assert abs(ev.endpoint_scores(y[ep], p2.values[ep])["macro_f1_RO"] - ref2) < 1e-12                 # control: round-2 procedure
    preds_all = {"procedure": proc.values, "CDR_RO": cdr, "round2_procedure": p2.values, **{k: outer[k].values for k in CANDS}}
    rows = []
    for name, p in preds_all.items():
        rows.append(dict(rule=name, batch="pooled", **ev.endpoint_scores(y[ep], p[ep])))
        for b in B:
            k = ep & (bat == b); rows.append(dict(rule=name, batch=b, **ev.endpoint_scores(y[k], p[k])))
    pd.DataFrame(rows).to_csv(f"{R3}/scores.csv", index=False)
    prim = []
    for cname, ref in [("R3-1", "CDR_RO"), ("R3-2", "round2_procedure")]:
        r = ev.paired(y[ep], preds_all[ref][ep], proc.values[ep], fruit[ep]); r.update(contrast=cname, candidate="procedure", reference=ref); prim.append(r)
    PR = pd.DataFrame(prim); PR["p_holm"] = ev.holm(PR.p_perm.values); r1 = PR.iloc[0]
    PR["beneficial_vs_CDR_RO"] = bool((r1.d_f1_pp >= 3) & (r1.ci_f1_lo_pp > 0) & (r1.p_holm < 0.05)); PR.to_csv(f"{R3}/primary_contrasts.csv", index=False)
    sec = []
    for k in CANDS:
        for ref in ["CDR_RO", "round2_procedure"]:
            r = ev.paired(y[ep], preds_all[ref][ep], preds_all[k][ep], fruit[ep]); r.update(candidate=k, reference=ref); sec.append(r)
    pd.DataFrame(sec).to_csv(f"{R3}/secondary_contrasts.csv", index=False)
    me = open(__file__).read(); blk = me[me.index("def saerens2"):me.index("def selftest")]
    s2 = me[me.index("    sel_rows, inner_rows"):me.index("    assert (proc >= 0).all()")]
    au = {"rules_use_labels": bool(re.search(r"\.y\b|label", blk)),
          "selection_reads_outer_batch": bool(re.search(r'preds\("lobo"', s2.split("m = T.Maturity_Stage == b")[0]))}
    assert not any(au.values()), au; json.dump(au, open(f"{R3}/leakage_audit.json", "w"), indent=1)
    print("written: selection, inner_scores, priors, scores, primary_contrasts, secondary_contrasts, leakage_audit")


if __name__ == "__main__":
    {"selftest": selftest, "evaluate": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "evaluate"]()
