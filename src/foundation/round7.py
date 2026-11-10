"""Round 7 (docs/ANALYSIS_PLAN_round7.md, tag prereg-round7): Rank-then-Anchor (RtA).
Intake-centred round-5 embedding blocks with a within-batch pairwise ranking head and a threshold in intake-centred coordinates.

  python src/foundation/round7.py fit7        build the four F3 session blocks with the round-5 code path (needs R5_EMB; GPU host, CPU only),
                                              assert the uncentred round-5-headed F3 reproduces results/round6/probe_sessions_with_intake.parquet,
                                              fit R7, R7-nc, R7-ce, R7-A, R7-noRGB (LOBO; R7 and R7-ce also p0); batch decodability before/after centring
  python src/foundation/round7.py evaluate7   references, scores, primary contrast R7 vs CDR-RO, secondaries, ranking/offset diagnostics, audit
  python src/foundation/round7.py smoke       structural smoke test on random session blocks of the real shapes (scores meaningless, not printed),
                                              including the negative test that a cross-batch pair must fail the pair assertion
Outputs: results/round7/ (override ROUND7_DIR). fit7 prints only arm names, row and pair counts, the selected C, run times and the
identity-check maximum difference.
"""
import os, sys, json, re, time, tempfile
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import round5 as r5
ev = r5.ev; lc = r5.lc; B = r5.B; R5 = r5.R5; R6 = r5.R6
R7 = os.environ.get("ROUND7_DIR", "results/round7")
CS = [0.001, 0.01, 0.1]                                   # round-5 grid; ties -> smaller C
BLOCKS = ["A1", "thermal", "ms", "rgb"]                   # the four F3 blocks, in the F3 order
ARMS = {"R7": (BLOCKS, True, "rank"), "R7-nc": (BLOCKS, False, "rank"), "R7-ce": (BLOCKS, True, "ce"),
        "R7-A": (["A1"], True, "rank"), "R7-noRGB": (["A1", "thermal", "ms"], True, "rank")}
P0_ARMS = ["R7", "R7-ce"]                                  # p0 cost check only (plan Section 4)
DIMS = {"A1": 768, "thermal": 768, "ms": 8 * 768, "rgb": 768}   # smoke-test shapes (BEATs maskedmean; DINOv2-base pooler; MS band8)
EPS = 1e-12


# ---------------------------------------------------------------- round 7 fit
def session_blocks():
    """the four F3 session blocks, built exactly as round5.fit6 builds them"""
    d, T, A, pop, lab, fb, ff = r5.context(); SESS = {}
    Z = r5.audio_recordings("A1", d); SESS["A1"] = pd.DataFrame(Z).groupby(d.session_id.values).mean()
    for mod in ["thermal", "ms", "rgb"]:
        X, u = r5.image_embeddings(mod, "DINO", d, T, A, lab); SESS[mod] = pd.DataFrame(X).groupby(u.session_id.values).mean()
    S = sorted(set.intersection(*[set(v.index) for v in SESS.values()]))
    return _units(S, d, A, lab, fb), {k: SESS[k].loc[S].values for k in BLOCKS}, ff


def smoke_blocks():
    """random session blocks of the real shapes for the sessions with four modalities (structural test only)"""
    d, T, A, pop, lab, fb, ff = r5.context(); S = sorted(T.index); rng = np.random.default_rng(0)
    return _units(S, d, A, lab, fb), {k: rng.normal(size=(len(S), DIMS[k])) for k in BLOCKS}, ff


def _units(S, d, A, lab, fb):
    su = pd.DataFrame({"session_id": S}); su["fruit_id"] = su.session_id.str.split("_").str[0].astype(int)
    su["batch"] = su.fruit_id.map(fb); su["label"] = su.session_id.map(lab); su["intake"] = su.session_id.map(A.anchor).fillna(False).astype(bool)
    su["has_base"] = su.fruit_id.isin(set(d.fruit_id[d.intake])); return su


def centre(blocks, su):
    """Step 1: subtract, per batch and block, the mean session embedding of that batch's intake sessions (label-free)"""
    bt = su.batch.values; it = su.intake.values; out = {}; info = {"n_intake": {}, "max_abs_intake_mean_after": 0.0}
    for k, X in blocks.items():
        Xc = np.empty_like(X)
        for g in B:
            m = bt == g; mi = m & it; mu = X[mi].mean(0); Xc[m] = X[m] - mu
            info["n_intake"][g] = int(mi.sum())
            info["max_abs_intake_mean_after"] = max(info["max_abs_intake_mean_after"], float(np.abs(Xc[mi].mean(0)).max()))
        out[k] = Xc
    assert info["max_abs_intake_mean_after"] < 1e-9, info
    return out, info


def scaled(blocks, names, tr):
    """Step 2 (as F3): standardise each block on the training rows, scale by 1/sqrt(dim), concatenate"""
    Xs = []
    for k in names:
        Bk = blocks[k]; mu = Bk[tr].mean(0); sd = Bk[tr].std(0) + 1e-8; Xs.append((Bk - mu) / sd / np.sqrt(Bk.shape[1]))
    return np.hstack(Xs)


def make_pairs(idx, y, bt):
    """all ordered (Overripe i, Ripe j) pairs within the same batch among the rows idx"""
    I, J, G = [], [], []
    for g in B:
        o = idx[(y[idx] == 2) & (bt[idx] == g)]; r = idx[(y[idx] == 1) & (bt[idx] == g)]
        if len(o) == 0 or len(r) == 0: continue
        ii, jj = np.meshgrid(o, r, indexing="ij"); I.append(ii.ravel()); J.append(jj.ravel()); G += [g] * ii.size
    return np.concatenate(I), np.concatenate(J), np.array(G)


def check_pairs(I, J, bt):
    assert (bt[I] == bt[J]).all(), "a difference vector joins two batches"


def fit_rank(U, I, J, G, C):
    """Step 3: L2 pairwise logistic loss without intercept, each training batch weighted equally (weights 1/P_g)"""
    D = U[I] - U[J]; P = pd.Series(G).value_counts(); w = 1.0 / P.loc[G].values
    X = np.vstack([D, -D]); yy = np.r_[np.ones(len(D)), np.zeros(len(D))]; sw = np.r_[w, w]
    m = LogisticRegression(fit_intercept=False, C=C, l1_ratio=0, max_iter=5000).fit(X, yy, sample_weight=sw)   # l1_ratio=0: pure L2
    return m.coef_.ravel()


def rank_head(U, tr_idx, y, bt, fruit):
    """inner fruit-grouped 3-fold CV over CS (criterion: mean within-batch AUC over training batches), Step-4 threshold from the inner
    out-of-fold scores of the selected C, final fit on all training pairs. Returns w, t, C, inner info, pair counts."""
    tb = [g for g in B if (bt[tr_idx] == g).any()]; best = None
    for C in CS:
        oof = np.full(len(y), np.nan)
        for itr, ite in GroupKFold(3).split(tr_idx, groups=fruit[tr_idx]):
            I, J, G = make_pairs(tr_idx[itr], y, bt); check_pairs(I, J, bt)
            oof[tr_idx[ite]] = U[tr_idx[ite]] @ fit_rank(U, I, J, G, C)
        crit = np.mean([roc_auc_score(y[tr_idx][bt[tr_idx] == g] == 2, oof[tr_idx][bt[tr_idx] == g]) for g in tb])
        if best is None or crit > best[1]: best = (C, crit, oof.copy())
    C, _, oof = best
    tg = [0.5 * (oof[tr_idx][(bt[tr_idx] == g) & (y[tr_idx] == 1)].mean() + oof[tr_idx][(bt[tr_idx] == g) & (y[tr_idx] == 2)].mean()) for g in tb]
    I, J, G = make_pairs(tr_idx, y, bt); check_pairs(I, J, bt)
    return fit_rank(U, I, J, G, C), float(np.mean(tg)), C, pd.Series(G).value_counts().to_dict()


def fit_arms(su, blocks, ff, out_dir, identity):
    t0 = time.time(); bt = su.batch.values; fruit = su.fruit_id.values; it = su.intake.values
    ep = (~su.intake & su.has_base).values; y = su.label.map({"Ripe": 1, "Overripe": 2}); trainable = ep & y.notna().values
    y = y.fillna(0).astype(int).values
    cen, cinfo = centre(blocks, su); rows = []; info = {"pairs": {}, "selected_C": {}, "n_train": {}}
    for arm, (names, centred, head) in ARMS.items():
        BL = cen if centred else blocks
        for sp, arg, trm, tem in r5.splits(su, ff):
            if sp == "p0" and arm not in P0_ARMS: continue
            tr = trainable & trm; te = (ep & tem) | it                 # endpoint units of the test split + every intake session (output only)
            assert not (tr & it).any(), "intake session used as a training row"
            if sp == "lobo": assert not (tr & (bt == arg)).any(), "held-out batch row in training"
            U = scaled(BL, names, tr); key = f"{arm}|{sp}|{arg}"
            if head == "rank":
                w, t, C, pc = rank_head(U, np.flatnonzero(tr), y, bt, fruit); f = U[te] @ w; info["pairs"][key] = pc
            else:                                                          # R7-ce: round-5 embedding_logreg head on the centred blocks
                est, grid = lc.estimator("embedding_logreg")
                gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="f1_macro", n_jobs=1).fit(U[tr], y[tr], groups=fruit[tr])
                q = np.clip(gs.predict_proba(U[te]), EPS, 1 - EPS); assert list(gs.classes_) == [1, 2]
                f = np.log(q[:, 1]) - np.log(q[:, 0]); t = 0.0; C = gs.best_params_["clf__C"]
            info["selected_C"][key] = C; info["n_train"][key] = int(tr.sum())
            rows.append(pd.DataFrame({"session_id": su.session_id.values[te], "model": arm, "split": sp, "held_out": arg,
                                      "role": r5._roles(su, te, tem), "batch": bt[te], "score": f, "threshold": t,
                                      "pred": np.where(f > t, 2, 1), "C": C}))
        print(arm, "done", f"{time.time()-t0:.0f}s", flush=True)
    RS = pd.concat(rows, ignore_index=True); os.makedirs(out_dir, exist_ok=True); RS.to_parquet(f"{out_dir}/rta_sessions.parquet", index=False)
    # batch decodability of the concatenated blocks on intake sessions, before and after Step 1 (label-free outcome)
    bd = [dict(r5.batch_decode(np.hstack([blocks[k] for k in BLOCKS])[it], bt[it], fruit[it], "F3 blocks, uncentred"), step="before"),
          dict(r5.batch_decode(np.hstack([cen[k] for k in BLOCKS])[it], bt[it], fruit[it], "F3 blocks, intake-centred"), step="after")]
    pd.DataFrame(bd).to_csv(f"{out_dir}/batch_decodability_centred.csv", index=False)
    n = RS.groupby(["model", "split", "role"]).size().rename("n").reset_index()
    json.dump({"identity": identity, "centring": cinfo, "rows": int(len(RS)), "counts": n.to_dict(orient="records"),
               "pairs_within_batch_only": True, "intake_never_training_rows": True, "held_out_rows_never_training_rows": True, **info},
              open(f"{out_dir}/fit7_identity_check.json", "w"), indent=1, default=str)
    print("rows", len(RS), "| intake per batch", cinfo["n_intake"], "| selected C", sorted(set(info["selected_C"].values())))
    print("pair counts (first LOBO split of R7)", info["pairs"].get(f"R7|lobo|{B[0]}"), f"| {time.time()-t0:.0f}s")


def identity_f3(su, blocks, ff):
    """control: the uncentred, round-5-headed F3 must reproduce the round-6 F3 rows (endpoint and intake, LOBO and p0)"""
    P = pd.concat(r5.probe_blocks([blocks[k] for k in BLOCKS], su, ff, "F3", intake_rows=True), ignore_index=True)
    ref = pd.read_parquet(f"{R6}/probe_sessions_with_intake.parquet"); ref = ref[ref.model == "F3"]
    k = ["session_id", "split", "role", "held_out"]; P = P.set_index(k).sort_index(); ref = ref.set_index(k).sort_index()
    assert P.index.equals(ref.index), (len(P), len(ref))
    dmax = float(np.abs(P[["p_Ripe", "p_Overripe"]].values - ref[["p_Ripe", "p_Overripe"]].values).max())
    assert dmax < 1e-9, dmax
    return {"F3_vs_round6_max_abs_diff": dmax, "rows_compared": int(len(P))}


def fit7():
    t0 = time.time(); su, blocks, ff = session_blocks(); print("blocks built", len(su), "sessions", f"{time.time()-t0:.0f}s", flush=True)
    ident = identity_f3(su, blocks, ff); print("identity F3 vs round 6 max|diff|", f"{ident['F3_vs_round6_max_abs_diff']:.1e}", flush=True)
    fit_arms(su, blocks, ff, R7, ident)


# ---------------------------------------------------------------- round 7 evaluation
def _auc_ci(yb, sb, fb_, seed=42, nb=2000):
    fr = np.unique(fb_); idx = {f: np.flatnonzero(fb_ == f) for f in fr}; rng = np.random.default_rng(seed); bs = []
    while len(bs) < nb:
        s = np.concatenate([idx[f] for f in rng.choice(fr, len(fr), replace=True)])
        if len(np.unique(yb[s])) == 2: bs.append(roc_auc_score(yb[s] == 2, sb[s]))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _mf1(yy, pp):
    return f1_score(yy, pp, labels=[1, 2], average="macro", zero_division=0)


def evaluate7():
    d, T, A, pop, lab, fb, ff = r5.context()
    y = T.loc[pop, "y"].values.astype(int); assert set(y) <= {1, 2}
    fruit = T.loc[pop, "fruit_id"].values.astype(int); bat = T.loc[pop, "Maturity_Stage"].values
    RS = pd.read_parquet(f"{R7}/rta_sessions.parquet"); J = json.load(open(f"{R7}/fit7_identity_check.json"))
    if J["identity"] != "skipped (smoke test)": assert J["identity"]["F3_vs_round6_max_abs_diff"] < 1e-9
    preds, score, preds0 = {}, {}, {}
    for arm in ARMS:
        E = RS[(RS.model == arm) & (RS.split == "lobo") & (RS.role == "endpoint")].set_index("session_id"); assert set(pop) <= set(E.index), arm
        preds[arm] = E.loc[pop, "pred"].values.astype(int); score[arm] = E.loc[pop, "score"].values
    for arm in P0_ARMS:
        E = RS[(RS.model == arm) & (RS.split == "p0") & (RS.role == "endpoint")].set_index("session_id"); assert set(pop) <= set(E.index), arm
        preds0[arm] = E.loc[pop, "pred"].values.astype(int)
    # references (gated, same 104 sessions), recomputed and asserted equal to the shipped values
    preds.update(r5._references_lobo(pop, y))
    L1 = pd.read_parquet(f"{r5.V21}/lobo_cnn_sessions.parquet"); g = L1[L1.variant == "M0"].set_index("session_id").loc[pop]
    score["CDR_RO"] = np.log(g.p_Overripe.values + EPS) - np.log(g.p_Ripe.values + EPS)
    S = r5.r4.knock_sessions([f"{r5.R4}/cnn/current/paired_lobo_{b}.npy" for b in B]).loc[pop]
    score["K1"] = np.log(S.p_Overripe.values + EPS) - np.log(S.p_Ripe.values + EPS)
    P6 = pd.read_parquet(f"{R6}/probe_sessions_with_intake.parquet")
    F3 = P6[(P6.model == "F3") & (P6.split == "lobo") & (P6.role == "endpoint")].set_index("session_id").loc[pop, ["p_Ripe", "p_Overripe"]].values
    preds["F3"] = 1 + F3.argmax(1); score["F3"] = r5._logit(F3)
    preds["R6"], sh = r5._anchored(P6, "F3", pop, bat); score["R6"] = score["F3"] + np.array([sh[b]["shift_logit"] for b in bat])
    S6 = pd.read_csv(f"{R6}/scores.csv").query("split == 'lobo' and batch == 'pooled'").set_index("model").macro_f1_RO
    for m, v in [("CDR_RO", 0.661), ("K1", 0.567), ("F3", 0.692), ("R6", 0.759)]:
        s = ev.endpoint_scores(y, preds[m])["macro_f1_RO"]; assert abs(s - S6[m]) < 1e-9 and round(s, 3) == v, (m, s)
    P5 = pd.read_parquet(f"{R5}/probe_sessions.parquet"); g = P5[(P5.model == "F3") & (P5.split == "p0")].set_index("session_id")
    preds0["F3"] = 1 + g.loc[pop, ["p_Ripe", "p_Overripe"]].values.argmax(1)
    S5 = pd.read_csv(f"{R5}/scores.csv").query("split == 'p0' and batch == 'pooled'").set_index("model").macro_f1_RO
    assert abs(ev.endpoint_scores(y, preds0["F3"])["macro_f1_RO"] - S5["F3"]) < 1e-9
    rows = []
    for sp, P in [("lobo", preds), ("p0", preds0)]:
        for m, p in P.items():
            rows.append(dict(split=sp, model=m, batch="pooled", **ev.endpoint_scores(y, p)))
            for b in B:
                k = bat == b; rows.append(dict(split=sp, model=m, batch=b, **ev.endpoint_scores(y[k], p[k])))
    SC = pd.DataFrame(rows); os.makedirs(R7, exist_ok=True); SC.to_csv(f"{R7}/scores.csv", index=False)
    # primary: R7 vs CDR-RO (single contrast; Holm is the identity)
    r = ev.paired(y, preds["CDR_RO"], preds["R7"], fruit); r.update(contrast="R7", candidate="R7", reference="CDR_RO")
    PR = pd.DataFrame([r]); PR["p_holm"] = ev.holm(PR.p_perm.values)
    PR["beneficial"] = (PR.d_f1_pp >= 3) & (PR.ci_f1_lo_pp > 0) & (PR.p_holm < 0.05); PR.to_csv(f"{R7}/primary_contrast.csv", index=False)
    pairs = [("lobo", "R7", "F3"), ("lobo", "R7", "R6"), ("lobo", "R7", "K1"), ("lobo", "R7", "R7-nc"), ("lobo", "R7", "R7-ce"),
             ("lobo", "R7-nc", "CDR_RO"), ("lobo", "R7-ce", "CDR_RO"), ("lobo", "R7-A", "CDR_RO"), ("lobo", "R7-A", "R7"),
             ("lobo", "R7-noRGB", "CDR_RO"), ("lobo", "R7-noRGB", "R7"), ("p0", "R7", "F3"), ("p0", "R7-ce", "F3"), ("p0", "R7", "R7-ce")]
    sec = []
    for sp, cand, ref in pairs:
        P = preds if sp == "lobo" else preds0
        r = ev.paired(y, P[ref], P[cand], fruit); r.update(split=sp, candidate=cand, reference=ref, status="exploratory_unadjusted"); sec.append(r)
    SE = pd.DataFrame(sec); SE.to_csv(f"{R7}/secondary_contrasts.csv", index=False)
    # ranking and offset per held-out batch
    diag = []; truef = {b: float((y[bat == b] == 2).mean()) for b in B}
    for m in ["R7", "R7-nc", "R7-ce", "R7-A", "R7-noRGB", "F3", "R6", "CDR_RO", "K1"]:
        for b in B:
            k = bat == b; yb = y[k]; sb = score[m][k]; pb = preds[m][k]; lo, hi = _auc_ci(yb, sb, fruit[k])
            fixed = _mf1(yb, pb); orac = max(_mf1(yb, np.where(sb >= t, 2, 1)) for t in np.unique(sb))
            diag.append(dict(model=m, batch=b, n=int(k.sum()), auc_overripe=roc_auc_score(yb == 2, sb), auc_ci_lo=lo, auc_ci_hi=hi,
                             frac_pred_overripe=float((pb == 2).mean()), true_frac_overripe=truef[b], macro_f1_fixed=fixed,
                             macro_f1_oracle_threshold=orac, threshold_loss_pp=100 * (orac - fixed)))
    DG = pd.DataFrame(diag)
    mean3 = DG.groupby("model").auc_overripe.mean()
    DG = pd.concat([DG, pd.DataFrame({"model": mean3.index, "batch": "mean of 3 batches", "auc_overripe": mean3.values})], ignore_index=True)
    DG.to_csv(f"{R7}/ranking_offset_diagnostics.csv", index=False)
    cond = lambda c, ref: bool((lambda s: (s.d_f1_pp >= 3) & (s.ci_f1_lo_pp > 0))(SE[(SE.split == "lobo") & (SE.candidate == c) & (SE.reference == ref)].iloc[0]))
    tl = DG[(DG.model == "R7") & DG.batch.isin(B)].threshold_loss_pp
    wording = {"threshold_transfers_without_post_hoc_shift": bool((tl <= 5).all()),
               "within_batch_ranking_training_does_not_cost_ranking": bool(mean3["R7"] >= mean3["F3"] - 0.03),
               "intake_centring_is_the_active_component": cond("R7", "R7-nc"), "ranking_head_is_the_active_component": cond("R7", "R7-ce")}
    wording["components_not_separable_at_n104"] = not (wording["intake_centring_is_the_active_component"] or wording["ranking_head_is_the_active_component"])
    me = open(__file__).read(); sec7 = me[me.index("# ---------------------------------------------------------------- round 7 fit"):me.index("# ---------------------------------------------------------------- round 7 evaluation")]
    au = {"round7_fit_reads_storage_visit_or_weight": bool(re.search(r"storage|Storage|weight_loss|Weight|end_of|days_from|elapsed|interval|Day_", sec7)),
          "identity": J["identity"], "centring": J["centring"], "pairs_within_batch_only": J["pairs_within_batch_only"],
          "intake_never_training_rows": J["intake_never_training_rows"], "held_out_rows_never_training_rows": J["held_out_rows_never_training_rows"],
          "held_out_information": "only the intake-session embeddings of the held-out batch enter its centring mean (Step 1)"}
    assert not au["round7_fit_reads_storage_visit_or_weight"]
    json.dump({"audit": au, "wording_conditions": wording, "population": len(pop), "primary_beneficial": bool(PR.beneficial.iloc[0]),
               "selected_C": J["selected_C"]}, open(f"{R7}/audit.json", "w"), indent=1, default=str)
    print("written: scores, primary_contrast, secondary_contrasts, ranking_offset_diagnostics, audit;", len(SC), "score rows")


# ---------------------------------------------------------------- smoke test (structure only; random embeddings)
def smoke():
    global R7
    su, blocks, ff = smoke_blocks(); bt = su.batch.values
    I, J, G = make_pairs(np.flatnonzero(su.label.isin(["Ripe", "Overripe"]).values), su.label.map({"Ripe": 1, "Overripe": 2}).fillna(0).astype(int).values, bt)
    check_pairs(I, J, bt)
    x = np.flatnonzero(bt == B[0])[0]; z = np.flatnonzero(bt == B[1])[0]
    try:
        check_pairs(np.r_[I, x], np.r_[J, z], bt); raise RuntimeError("negative test failed: cross-batch pair not detected")
    except AssertionError:
        print("negative test passed: injected cross-batch pair rejected")
    R7 = tempfile.mkdtemp(prefix="round7_smoke_", dir=os.environ.get("ROUND7_SMOKE_DIR", tempfile.gettempdir()))
    fit_arms(su, blocks, ff, R7, "skipped (smoke test)"); evaluate7()
    print("smoke outputs (meaningless scores, not inspected):", R7)


if __name__ == "__main__":
    {"fit7": fit7, "evaluate7": evaluate7, "smoke": smoke}[sys.argv[1]]()
