"""Pre-registered analysis (docs/ANALYSIS_PLAN_anchor_physics.md §4-§6): scores, primary contrasts with Holm adjustment and the
decision rule, secondary (exploratory) contrasts, within-batch check, anchor sensitivity and leakage audit.

Inputs (OUT = results/v21): lobo_classical_sessions, lobo_cnn_sessions, lobo_baselines_sessions, within_batch_<model>,
anchor_sens_classical, anchor_sens_cnn (parquet); cnn/M1_outer (knock-CNN M1 on the primary partition); anchor_sessions.csv.
Derived models: tsr_mean = mean of thermal, sound_cnn, rgb (same variant); hybrid = mean of physics and sound_cnn (no weights).
CDR = capture-day rule (anchors Unripe, endpoints by M0; pre-registered); CDR_RO = anchors Unripe, endpoints argmax over Ripe/Overripe
of M0 (exploratory addition, declared in DEVIATIONS.md before scoring).
Primary score: pooled LOBO endpoint sessions (106), macro-F1 over Ripe and Overripe (an endpoint predicted Unripe is an error) and the
false-Unripe rate. Uncertainty: fruit-cluster bootstrap (2,000, seed 42), fruit-level permutation (20,000, seed 42), clustered McNemar.
usage (repo root): python src/anchor_physics/evaluate.py OUT
"""
import os, sys, json, re
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "nested_selection")); sys.path.insert(0, os.path.join(HERE, "..", "label_sensitivity"))
import stats4
PC = ["p_Unripe", "p_Ripe", "p_Overripe"]; BATCHES = ["Immature", "Mature", "Overmature"]
N_BOOT, N_PERM, SEED = 2000, 20000, 42
PRIMARY = [("PA1", ("sound_cnn", "M1"), ("sound_cnn", "M0")), ("PA2", ("tsr_mean", "M1"), ("tsr_mean", "M0")),
           ("PC1", ("physics", "M1"), ("sound_cnn", "M0"))]
# within-batch comparator for each primary contrast: (candidate, reference), primary partition p0 (a knock CNN is involved in all three)
PRIMARY_WB = {"PA1": (("sound_cnn", "M1"), ("sound_cnn", "M0")), "PA2": (("tsr_mean", "M1"), ("tsr_mean", "M0")),
              "PC1": (("physics", "M1"), ("sound_cnn", "M0"))}


# ------------------------------------------------------------------ metrics from per-fruit counts (vectorized bootstrap / permutation)
def ro_counts(y, p, fruit, fr_index):
    """per fruit: tp, n_pred, n_true for Ripe (1) and Overripe (2), correct, n -> (F x 8)"""
    C = np.zeros((len(fr_index), 8)); fi = np.searchsorted(fr_index, fruit)
    for j, c in enumerate([1, 2]):
        np.add.at(C[:, 3 * j + 0], fi, ((p == c) & (y == c)).astype(float))
        np.add.at(C[:, 3 * j + 1], fi, (p == c).astype(float)); np.add.at(C[:, 3 * j + 2], fi, (y == c).astype(float))
    np.add.at(C[:, 6], fi, (p == y).astype(float)); np.add.at(C[:, 7], fi, (p == 0).astype(float))
    return C


def ro_f1(S):                                       # S (..., 8) summed counts -> macro-F1 over Ripe/Overripe
    f = []
    for j in range(2):
        den = S[..., 3 * j + 1] + S[..., 3 * j + 2]; f.append(np.where(den > 0, 2 * S[..., 3 * j] / np.maximum(den, 1e-12), 0.0))
    return (f[0] + f[1]) / 2


def endpoint_scores(y, p):
    return dict(n=len(y), macro_f1_RO=f1_score(y, p, labels=[1, 2], average="macro", zero_division=0), accuracy=accuracy_score(y, p),
                false_unripe=float(np.mean(p == 0)), recall_R=float(np.mean(p[y == 1] == 1)) if (y == 1).any() else np.nan,
                recall_O=float(np.mean(p[y == 2] == 2)) if (y == 2).any() else np.nan)


def all_scores(y, p):
    r = recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    return dict(n=len(y), macro_f1=f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0), accuracy=accuracy_score(y, p),
                recall_U=r[0], recall_R=r[1], recall_O=r[2])


def paired(y, p_ref, p_cand, fruit):
    fr = np.unique(fruit); Cr = ro_counts(y, p_ref, fruit, fr); Cc = ro_counts(y, p_cand, fruit, fr); n = len(y)
    d_f1 = float(ro_f1(Cc.sum(0)) - ro_f1(Cr.sum(0))); d_acc = float((Cc[:, 6].sum() - Cr[:, 6].sum()) / n)
    d_fu = float((Cc[:, 7].sum() - Cr[:, 7].sum()) / n)
    rng = np.random.default_rng(SEED); W = np.stack([np.bincount(rng.integers(0, len(fr), len(fr)), minlength=len(fr)) for _ in range(N_BOOT)]).astype(float)
    Sr, Sc = W @ Cr, W @ Cc; nb = W @ np.bincount(np.searchsorted(fr, fruit), minlength=len(fr)).astype(float)
    bf = ro_f1(Sc) - ro_f1(Sr); ba = (Sc[:, 6] - Sr[:, 6]) / nb; bu = (Sc[:, 7] - Sr[:, 7]) / nb
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    rng = np.random.default_rng(SEED); Z = rng.integers(0, 2, size=(N_PERM, len(fr))).astype(float)
    A = (1 - Z) @ Cr + Z @ Cc; B = Z @ Cr + (1 - Z) @ Cc; Tp = ro_f1(B) - ro_f1(A)
    p_perm = float((np.sum(np.abs(Tp) >= abs(d_f1) - 1e-12) + 1) / (N_PERM + 1))
    mc = stats4.clustered_mcnemar(p_ref == y, p_cand == y, fruit)
    return dict(d_f1_pp=100 * d_f1, ci_f1_lo_pp=100 * q(bf)[0], ci_f1_hi_pp=100 * q(bf)[1], p_perm=p_perm,
                d_acc_pp=100 * d_acc, ci_acc_lo_pp=100 * q(ba)[0], ci_acc_hi_pp=100 * q(ba)[1], p_mcnemar_clustered=mc["p"],
                d_false_unripe_pp=100 * d_fu, ci_fu_lo_pp=100 * q(bu)[0], ci_fu_hi_pp=100 * q(bu)[1], sessions_changed=int((p_ref != p_cand).sum()))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


# ------------------------------------------------------------------ data
def load_probs(out, sid):
    R = pd.concat([pd.read_parquet(f"{out}/{f}.parquet") for f in ["lobo_classical_sessions", "lobo_cnn_sessions", "lobo_baselines_sessions"]], ignore_index=True)
    P = {}
    for (m, v), g in R.groupby(["model", "variant"]):
        g = g.set_index("session_id")[PC]; assert set(sid) <= set(g.index), (m, v, len(set(sid) - set(g.index))); P[(m, v)] = g.loc[sid].values
    for v in ["M0", "M1", "M2", "M3", "M4"]: P[("tsr_mean", v)] = (P[("thermal", v)] + P[("sound_cnn", v)] + P[("rgb", v)]) / 3
    for v in ["M0", "M1", "M2"]: P[("hybrid", v)] = (P[("physics", v)] + P[("sound_cnn", v)]) / 2
    return P


def predictions(P, anchor):
    pr = {k: v.argmax(1) for k, v in P.items()}
    for m in {k[0] for k in P}:
        if (m, "M0") not in P: continue
        p = P[(m, "M0")].argmax(1).copy(); p[anchor] = 0; pr[(m, "CDR")] = p
        q = 1 + P[(m, "M0")][:, 1:].argmax(1); q[anchor] = 0; pr[(m, "CDR_RO")] = q
    return pr


def cnn_sessions(cnn_dir, sid):
    import retrain_without_questionable as rq
    return rq._sound_sessions(cnn_dir).loc[sid].values


def leakage_audit(out):
    import lobo_classical as lc; lc.OUT = out
    import baselines as bl
    forbidden = re.compile(r"(end.?of.?storage|storage|weight.?loss|loss|days|dry.?matter|tss|firm|label|class|actual|endpoint|^y$)", re.I)
    res = {}
    for m in ["thermal", "rgb", "sound_lr", "physics"]:
        _, cols, _ = lc.load(m); bad = [c for c in cols if forbidden.search(c)]; res[f"features_{m}"] = dict(n=len(cols), forbidden=bad); assert not bad, (m, bad)
    for m in ["phys_weight", "phys_fdom", "phys_pcm"]:
        _, cols, _ = bl.load(m); bad = [c for c in cols if forbidden.search(c)]; res[f"features_{m}"] = dict(n=len(cols), cols=cols, forbidden=bad); assert not bad
    _, cols, _ = lc.load("physics"); assert cols == ["f_dom", "f_first", "centroid", "decay20", "inv_q", "stiffness", "device_iphone"]
    S = pd.read_csv("audit/sessions.csv")[["session_id", "fruit_id", "date", "days_from_recorded_harvest"]]; S["date"] = pd.to_datetime(S.date)
    a = (S.date == S.groupby("fruit_id").date.transform("min")) & (S.days_from_recorded_harvest <= 4)
    A = pd.read_csv(f"{out}/anchor_sessions.csv").set_index("session_id")
    res["anchor_rule_rederived_label_free"] = bool((A.loc[S.session_id, "anchor"].values == a.values).all()); assert res["anchor_rule_rederived_label_free"]
    src = open("src/common_label/cnn_common.py").read(); blk = src[src.index('_norm = os.environ.get("CNN_NORM"'):src.index("dev_k = knock.device")]
    res["cnn_norm_block_uses_labels"] = bool(re.search(r"\byk\b|label|\.y\b", blk)); assert not res["cnn_norm_block_uses_labels"]
    ca = open(os.path.join(HERE, "cnn_adapt.py")).read(); blk = ca[ca.index("def adabn"):ca.index("def to_sessions")]
    res["adabn_uses_labels"] = bool(re.search(r"\byk\b|label", blk)); assert not res["adabn_uses_labels"]
    lcs = open(os.path.join(HERE, "lobo_classical.py")).read(); blk = lcs[lcs.index("def standardize"):lcs.index("def estimator")]
    res["standardize_uses_labels"] = bool(re.search(r"\by\b|label", blk)); assert not res["standardize_uses_labels"]
    return res


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    OUT = sys.argv[1]
    T = pd.read_csv("results/session_table.csv", low_memory=False); T = T[T.n_modalities == 4].sort_values("session_id")
    sid = T.session_id.values; y = T.y.values.astype(int); fruit = T.fruit_id.values.astype(int); batch = T.Maturity_Stage.values
    A = pd.read_csv(f"{OUT}/anchor_sessions.csv").set_index("session_id"); anchor = A.loc[sid, "anchor"].values.astype(bool)
    assert len(sid) == 268 and anchor.sum() == 162 and (~anchor).sum() == 106
    ep = ~anchor
    P = load_probs(OUT, sid); pr = predictions(P, anchor)

    # scores
    rows = []
    for (m, v), p in sorted(pr.items()):
        rows.append(dict(model=m, variant=v, scope="endpoint", batch="pooled", **endpoint_scores(y[ep], p[ep])))
        for b in BATCHES:
            k = ep & (batch == b); rows.append(dict(model=m, variant=v, scope="endpoint", batch=b, **endpoint_scores(y[k], p[k])))
        q = p.copy(); q[anchor] = 0
        for scope, pp in [("all_model_scored", p), ("all_capture_day_anchors", q)]:
            rows.append(dict(model=m, variant=v, scope=scope, batch="pooled", **all_scores(y, pp)))
            for b in BATCHES:
                k = batch == b; r = all_scores(y[k], pp[k]); r["majority_rate"] = float(np.bincount(y[k]).max() / k.sum())
                rows.append(dict(model=m, variant=v, scope=scope, batch=b, **r))
    SC = pd.DataFrame(rows); SC.to_csv(f"{OUT}/scores.csv", index=False)

    # within-batch check (session macro-F1, 3 classes, all 268 sessions)
    WB = {}
    for m in ["physics", "thermal", "rgb"]:
        w = pd.read_parquet(f"{OUT}/within_batch_{m}.parquet")
        for (v, part), g in w.groupby(["variant", "partition"]): WB[(m, v, part)] = g.set_index("session_id").loc[sid, PC].values
    WB[("sound_cnn", "M0", "p0")] = cnn_sessions("intermediate_v14/cnn_cpu_p0", sid)
    WB[("sound_cnn", "M1", "p0")] = cnn_sessions(f"{OUT}/cnn/M1_outer", sid)
    for v in ["M0", "M1"]: WB[("tsr_mean", v, "p0")] = (WB[("thermal", v, "p0")] + WB[("sound_cnn", v, "p0")] + WB[("rgb", v, "p0")]) / 3
    wrows = [dict(model=m, variant=v, partition=part, macro_f1=f1_score(y, Q.argmax(1), average="macro"), accuracy=accuracy_score(y, Q.argmax(1)))
             for (m, v, part), Q in WB.items()]
    WS = pd.DataFrame(wrows); WS.to_csv(f"{OUT}/within_batch_scores.csv", index=False)
    wb = lambda m, v, part="p0": float(WS[(WS.model == m) & (WS.variant == v) & (WS.partition == part)].macro_f1.iloc[0])
    wb10 = lambda m, v: float(WS[(WS.model == m) & (WS.variant == v) & (WS.partition.str.match(r"p\d$"))].macro_f1.mean())

    # primary contrasts
    prow = []
    for name, cand, ref in PRIMARY:
        r = paired(y[ep], pr[ref][ep], pr[cand][ep], fruit[ep])
        (wc, wr) = PRIMARY_WB[name]; r.update(contrast=name, candidate="/".join(cand), reference="/".join(ref),
                                              wb_cand_p0=wb(*wc), wb_ref_p0=wb(*wr), wb_drop_pp=100 * (wb(*wr) - wb(*wc)))
        prow.append(r)
    PR = pd.DataFrame(prow); PR["p_holm"] = holm(PR.p_perm.values)
    PR["beneficial"] = (PR.d_f1_pp >= 5) & (PR.ci_f1_lo_pp > 0) & (PR.p_holm < 0.05) & (PR.wb_drop_pp <= 2)
    PR.to_csv(f"{OUT}/primary_contrasts.csv", index=False)

    # secondary (exploratory, unadjusted)
    sec = [(("sound_cnn", v), ("sound_cnn", "M0")) for v in ["M2", "M3", "M4", "M5", "M6", "CDR_RO"]] + \
          [(("sound_cnn", "M1"), ("sound_cnn", "M2")), (("sound_cnn", "M5"), ("sound_cnn", "M6")), (("sound_cnn", "M1"), ("sound_cnn", "CDR_RO"))] + \
          [(("tsr_mean", v), ("tsr_mean", "M0")) for v in ["M2", "M3", "M4", "CDR_RO"]] + [(("tsr_mean", "M1"), ("tsr_mean", "CDR_RO")), (("tsr_mean", "M1"), ("tsr_mean", "M2"))] + \
          [(("physics", v), ("sound_cnn", "M0")) for v in ["M0", "M2"]] + [(("physics", "M1"), ("physics", "M0")), (("physics", "M1"), ("physics", "M2"))] + \
          [(("hybrid", v), ("sound_cnn", "M0")) for v in ["M0", "M1", "M2"]] + \
          [((b, v), ("sound_cnn", "M0")) for b in ["phys_weight", "phys_fdom", "phys_pcm"] for v in ["M0", "M1"]] + \
          [(("physics", "M1"), ("phys_pcm", "M1")), (("physics", "M1"), ("phys_weight", "M1"))] + \
          [((m, v), (m, "M0")) for m in ["thermal", "rgb", "sound_lr"] for v in ["M1", "M2", "M3", "M4", "CDR_RO"]]
    srow = []
    for cand, ref in sec:
        r = paired(y[ep], pr[ref][ep], pr[cand][ep], fruit[ep]); r.update(candidate="/".join(cand), reference="/".join(ref)); srow.append(r)
    SE = pd.DataFrame(srow); SE.to_csv(f"{OUT}/secondary_contrasts.csv", index=False)

    # within-batch summary (ten partitions where available)
    wsum = [dict(model=m, variant=v, p0=wb(m, v), ten_partition_mean=wb10(m, v) if m in ("physics", "thermal", "rgb") else np.nan)
            for m, v in sorted({(a, b) for a, b, _ in WB})]
    pd.DataFrame(wsum).to_csv(f"{OUT}/within_batch_summary.csv", index=False)

    # anchor sensitivity
    AC = pd.concat([pd.read_parquet(f"{OUT}/anchor_sens_classical.parquet"), pd.read_parquet(f"{OUT}/anchor_sens_cnn.parquet").assign(model="sound_cnn")], ignore_index=True)
    arows = []
    for f in sorted(AC.fraction.unique()):
        for k in sorted(AC[AC.fraction == f].draw.unique()):
            g = AC[(AC.fraction == f) & (AC.draw == k)]; Q = {}
            for m in ["thermal", "rgb", "physics", "sound_cnn"]:
                s = g[g.model == m].set_index("session_id")[PC]; Q[m] = s.loc[sid].values
            Q["tsr_mean"] = (Q["thermal"] + Q["sound_cnn"] + Q["rgb"]) / 3
            for m, Qm in Q.items():
                arows.append(dict(model=m, fraction=f, draw=k, **endpoint_scores(y[ep], Qm.argmax(1)[ep])))
    AS = pd.DataFrame(arows); AS.to_csv(f"{OUT}/anchor_sensitivity.csv", index=False)
    for m in ["thermal", "rgb", "physics", "sound_cnn", "tsr_mean"]:       # control: fraction 1.0 equals M1
        a = AS[(AS.model == m) & (AS.fraction == 1.0)].macro_f1_RO.iloc[0]
        b = SC[(SC.model == m) & (SC.variant == "M1") & (SC.scope == "endpoint") & (SC.batch == "pooled")].macro_f1_RO.iloc[0]
        assert abs(a - b) < 1e-12, (m, a, b)

    json.dump(leakage_audit(OUT), open(f"{OUT}/leakage_audit.json", "w"), indent=1)
    print("written: scores, primary_contrasts, secondary_contrasts, within_batch_scores/summary, anchor_sensitivity, leakage_audit")
