"""Round 5 (docs/ANALYSIS_PLAN_round5.md, tag prereg-round5): pretrained audio and vision foundation encoders.

  python src/foundation/round5.py fit        frozen probes A1-A6, A1c/A1p/A1s/A1b/A1w, vision V-T/V-M/V-R, fusion F3 (LOBO + p0)
  python src/foundation/round5.py diagnose   batch decodability of each frozen embedding on intake units (label-free outcome)
  python src/foundation/round5.py evaluate   F1/F2/F4/F5 (probability means), A1ft (beats_ft.py outputs), references; Q1/Q2 + Holm
Round 6 (docs/ANALYSIS_PLAN_round6.md, tag prereg-round6; outputs results/round6/, override ROUND6_DIR):
  python src/foundation/round5.py fit6       refit F3, A1, A5, V-{thermal,ms,rgb}-DINO with the same probe; also predict intake sessions;
                                             assert endpoint probabilities identical to results/round5/probe_sessions.parquet
  python src/foundation/round5.py evaluate6  round-1 M3 anchored logit shift (R6 = anchored F3), references, R6 vs CDR-RO; secondaries
Embeddings: R5_EMB (default work/r5_embeddings; layout in README_embeddings.md). Outputs: results/round5/ (override ROUND5_DIR).
Population, gating and training data as round 4 (104 endpoint sessions). Fitting prints only sizes and run times.
"""
import os, sys, json, re, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
HERE = os.path.dirname(os.path.abspath(__file__)); AP = os.path.join(HERE, "..", "anchor_physics")
for p in [AP, os.path.join(HERE, ".."), os.path.join(HERE, "..", "label_sensitivity"), os.path.join(HERE, "..", "nested_selection")]:
    sys.path.insert(0, p)
import lobo_classical as lc, evaluate as ev, round4 as r4
EMB = os.environ.get("R5_EMB", "work/r5_embeddings"); R5 = os.environ.get("ROUND5_DIR", "results/round5")
V21 = r4.V21; R4 = r4.R4; B = ev.BATCHES

# ---------------------------------------------------------------- embedding files (fixed before the tag; see plan Section 2)
AUDIO = {  # arm code -> (encoder, window, pooled feature); files emb/audio_{encoder}_{window}_{feature}.npy (README_embeddings.md)
    "A1": ("beats", "W150", "maskedmean"), "A2": ("panns", "W150", "embedding"), "A3": ("clap", "W150", "repeatpad_projl2"),
    "A4": ("ast", "W150", "pooler"), "A5": ("wav2vec2", "W150", "alllayers_mean"), "A6": ("hubert", "W150", "alllayers_mean"),
    "A1w": ("beats", "W500", "maskedmean")}
VISION = {  # (modality, encoder) -> (file prefix, feature); files emb/{prefix}_{feature}.npy with emb/{prefix}_index.parquet
    ("thermal", "DINO"): ("thermal_dinov2_padsq_fixed20to40C", "pooler"), ("thermal", "SIGLIP"): ("thermal_siglip_padsq_fixed20to40C", "pooler"),
    ("thermal", "CLIP"): ("thermal_clip_padsq_fixed20to40C", "proj"), ("thermal_mm", "DINO"): ("thermal_dinov2_padsq_minmax", "pooler"),
    ("ms", "DINO"): ("multispectral_dinov2_band8", "pooler"), ("ms", "SIGLIP"): ("multispectral_siglip_band8", "pooler"),
    ("ms", "CLIP"): ("multispectral_clip_band8", "proj"),
    ("rgb", "DINO"): ("rgb_dinov2_padsq", "pooler"), ("rgb", "SIGLIP"): ("rgb_siglip_padsq", "pooler"), ("rgb", "CLIP"): ("rgb_clip_padsq", "proj")}


def _load(prefix, feat):
    return np.load(os.path.join(EMB, f"{prefix}_{feat}.npy")).astype(np.float64)


# ---------------------------------------------------------------- units
def context():
    d, _ = r4.recordings()
    T = pd.read_csv("results/session_table.csv", low_memory=False); T = T[T.n_modalities == 4].set_index("session_id")
    A = pd.read_csv(f"{V21}/anchor_sessions.csv").set_index("session_id")
    pop = sorted(set(d.session_id[~d.intake & d.has_base]) & set(T.index[~A.loc[T.index, "anchor"].astype(bool)]))
    assert len(pop) == 104, len(pop)
    lab = pd.read_csv("data/session_labels_common.csv").set_index("session_id").label_common
    cf = pd.read_csv("data/durian_mm_capture_folds.csv"); fb = cf.groupby("fruit_id").Maturity_Stage.first(); ff = cf.groupby("fruit_id").fold_fruit.first()
    return d, T, A, pop, lab, fb, ff


def audio_recordings(code, d):
    """knock embeddings -> recording mean, for the verified recordings of r4.recordings()"""
    enc, win, feat = AUDIO[code]
    ix = pd.read_parquet(os.path.join(EMB, "audio_knocks_index.parquet"))
    kn = pd.read_parquet("work/raw/sound/features_sound_knocks.parquet")[["file", "knock_idx"]]
    assert (ix.file.values == kn.file.values).all() and (ix.knock_idx.values == kn.knock_idx.values).all()
    Z = _load(f"audio_{enc}_{win}", feat); assert len(Z) == len(ix)
    g = pd.DataFrame(Z).groupby(ix.file.values).mean()
    assert set(d.file) <= set(g.index), "verified recording without knock embeddings"
    return g.loc[d.file.values].values


def donors(d, ff, fb):
    """round-4 K6 derangement: fruits with a baseline, within (harvest batch, primary fruit fold), rng 42"""
    f = np.array(sorted(set(d.fruit_id[d.intake]))); DON = {x: x for x in f}; r = np.random.default_rng(42)
    g = pd.DataFrame({"f": f.astype(float), "b": fb.loc[f].values, "k": ff.loc[f].values.astype(int)})
    for _, grp in g.groupby(["b", "k"]):
        v = grp.f.values
        if len(v) < 2: continue
        while True:
            p = r.permutation(v)
            if not (p == v).any(): break
        DON.update(dict(zip(v.astype(int), p.astype(int))))
    assert sum(DON[x] != x for x in DON) >= len(DON) - 6
    return DON


def image_embeddings(mod, enc, d, T, A, lab, index_only=False):
    """image-level embeddings of the analysis images (rows of data/durian_mm_capture_folds.csv), keyed to sessions by zip_member"""
    prefix, feat = VISION[(mod, enc)]
    ix = pd.read_parquet(os.path.join(EMB, f"{prefix}_index.parquet"))
    X = np.zeros((len(ix), 1)) if index_only else _load(prefix, feat); assert len(X) == len(ix)
    cf = pd.read_csv("data/durian_mm_capture_folds.csv"); cf = cf[cf.modality != "sound"]
    sid = dict(zip(cf.zip_member, cf.fruit_id.astype(int).astype(str) + "_" + cf.date_captured.astype(str)))
    m = ix.zip_member.isin(sid).values; ix = ix[m].reset_index(drop=True); X = X[m].reshape(m.sum(), -1)
    assert ix.zip_member.is_unique
    u, _ = image_units(ix.assign(session_id=ix.zip_member.map(sid)), d, T, A, lab)
    return X, u


def image_units(ix, d, T, A, lab):
    cf = pd.read_csv("data/durian_mm_capture_folds.csv")
    u = ix[["session_id"]].copy(); u["fruit_id"] = u.session_id.str.split("_").str[0].astype(int)
    u["batch"] = u.fruit_id.map(cf.groupby("fruit_id").Maturity_Stage.first()); u["label"] = u.session_id.map(lab)
    u["intake"] = u.session_id.map(A.anchor).fillna(False).astype(bool)
    u["has_base"] = u.fruit_id.isin(set(d.fruit_id[d.intake]))
    if "camera" in ix: u["camera"] = ix.camera.values
    assert u.batch.notna().all(), "image of a fruit outside the analysis set"
    return u.reset_index(drop=True), ix


# ---------------------------------------------------------------- probe
def splits(units, ff):
    fo = units.fruit_id.map(ff).values.astype(int); bt = units.batch.values
    return [("lobo", b, bt != b, bt == b) for b in B] + [("p0", str(k), fo != k, fo == k) for k in range(5)]


def probe(X, units, sel, ff, name, intake_rows=False):
    """train on endpoint units (non-intake, fruit with a baseline, label R/O) of training fruits; predict selected test units
    intake_rows=True (round 6 only): additionally predict the label-free intake units of ALL batches with the same fitted model
    (intake units are never training rows); each output row then carries `role` (endpoint | intake_test | intake_train)."""
    ep = (~units.intake & units.has_base).values & sel
    y = units.label.map({"Ripe": 1, "Overripe": 2}); trainable = ep & y.notna().values; y = y.fillna(0).astype(int).values
    rows = []
    for sp, arg, trm, tem in splits(units, ff):
        tr = trainable & trm; te = ep & tem
        if intake_rows: te = te | (units.intake.values & sel)
        est, grid = lc.estimator("embedding_logreg")
        gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="f1_macro", n_jobs=1).fit(X[tr], y[tr], groups=units.fruit_id.values[tr])
        q = gs.predict_proba(X[te]); assert list(gs.classes_) == [1, 2]
        t = pd.DataFrame({"session_id": units.session_id.values[te], "p_Ripe": q[:, 0], "p_Overripe": q[:, 1]})
        keys = ["session_id"]
        if intake_rows: t["role"] = _roles(units, te, tem); keys = ["session_id", "role"]
        s = t.groupby(keys)[["p_Ripe", "p_Overripe"]].mean().reset_index(); s["model"] = name; s["split"] = sp
        if intake_rows: s["held_out"] = arg                                 # which model (held-out batch / fold) produced the row
        rows.append(s)
    return rows


def _roles(units, te, tem):
    r = np.where(units.intake.values, np.where(tem, "intake_test", "intake_train"), "endpoint"); return r[te]


def fit():
    t0 = time.time(); d, T, A, pop, lab, fb, ff = context(); rows = []
    au = d[["session_id", "fruit_id", "batch", "label", "intake", "has_base"]].copy(); allsel = np.ones(len(au), bool)
    DON = donors(d, ff, fb); REC = {}
    for code in AUDIO:
        Z = audio_recordings(code, d); REC[code] = Z
        rows += probe(Z, au, allsel, ff, code)
        if code == "A1":
            base = pd.DataFrame(Z[d.intake.values]).groupby(d.fruit_id.values[d.intake.values]).mean()
            hb = d.has_base.values; z0 = np.full_like(Z, np.nan); z0[hb] = base.loc[d.fruit_id.values[hb]].values
            zs = np.full_like(Z, np.nan); zs[hb] = base.loc[[DON[f] for f in d.fruit_id.values[hb]]].values
            for nm, X in [("A1c", Z - z0), ("A1p", np.hstack([Z, Z - z0])), ("A1s", Z - zs), ("A1b", z0)]:
                X = np.where(np.isfinite(X), X, 0.0)                     # rows without a baseline are never trained or scored
                rows += probe(X, au, hb, ff, nm)
        print(code, "done", f"{time.time()-t0:.0f}s", flush=True)
    SESS = {"A1": pd.DataFrame(REC["A1"]).groupby(d.session_id.values).mean()}
    for (mod, enc), (prefix, feat) in VISION.items():
        X, u = image_embeddings(mod, enc, d, T, A, lab)
        rows += probe(X, u, np.ones(len(u), bool), ff, f"V-{mod}-{enc}")
        if enc == "DINO" and mod in ("thermal", "ms", "rgb"): SESS[mod] = pd.DataFrame(X).groupby(u.session_id.values).mean()
        print(mod, enc, "done", f"{time.time()-t0:.0f}s", flush=True)
    # F3: session-level embedding fusion; each block standardised on the training rows and scaled by 1/sqrt(dim)
    S = sorted(set.intersection(*[set(v.index) for v in SESS.values()]))
    su = pd.DataFrame({"session_id": S}); su["fruit_id"] = su.session_id.str.split("_").str[0].astype(int)
    su["batch"] = su.fruit_id.map(fb); su["label"] = su.session_id.map(lab); su["intake"] = su.session_id.map(A.anchor).fillna(False).astype(bool)
    su["has_base"] = su.fruit_id.isin(set(d.fruit_id[d.intake]))
    blocks = [SESS[k].loc[S].values for k in ["A1", "thermal", "ms", "rgb"]]
    rows += probe_blocks(blocks, su, ff, "F3")
    os.makedirs(R5, exist_ok=True); pd.concat(rows, ignore_index=True).to_parquet(f"{R5}/probe_sessions.parquet", index=False)
    print("fit written", f"{time.time()-t0:.0f}s")


def probe_blocks(blocks, su, ff, name, intake_rows=False):
    rows = []; ep = (~su.intake & su.has_base).values
    y = su.label.map({"Ripe": 1, "Overripe": 2}); trainable = ep & y.notna().values; y = y.fillna(0).astype(int).values
    for sp, arg, trm, tem in splits(su, ff):
        tr = trainable & trm; te = ep & tem; Xs = []
        if intake_rows: te = te | su.intake.values                       # round 6: also score the label-free intake sessions
        for Bk in blocks:
            mu = Bk[tr].mean(0); sd = Bk[tr].std(0) + 1e-8; Xs.append((Bk - mu) / sd / np.sqrt(Bk.shape[1]))
        X = np.hstack(Xs); est, grid = lc.estimator("embedding_logreg")
        gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="f1_macro", n_jobs=1).fit(X[tr], y[tr], groups=su.fruit_id.values[tr])
        q = gs.predict_proba(X[te])
        t = pd.DataFrame({"session_id": su.session_id.values[te], "p_Ripe": q[:, 0], "p_Overripe": q[:, 1], "model": name, "split": sp})
        if intake_rows: t["role"] = _roles(su, te, tem); t["held_out"] = arg
        rows.append(t)
    return rows


# ---------------------------------------------------------------- diagnostics
def diagnose():
    d, T, A, pop, lab, fb, ff = context(); out = []
    for code in AUDIO:
        Z = audio_recordings(code, d); m = d.intake.values
        out.append(batch_decode(Z[m], d.batch.values[m], d.fruit_id.values[m], code))
    for (mod, enc), (prefix, feat) in VISION.items():
        X, u = image_embeddings(mod, enc, d, T, A, lab); m = u.intake.values
        out.append(batch_decode(X[m], u.batch.values[m], u.fruit_id.values[m], f"V-{mod}-{enc}"))
    os.makedirs(R5, exist_ok=True); pd.DataFrame(out).to_csv(f"{R5}/batch_decodability.csv", index=False)


def batch_decode(X, b, g, name):
    b = np.asarray(b).astype(str); g = np.asarray(g).astype(int)
    est, grid = lc.estimator("embedding_logreg"); pred = np.empty(len(b), object)
    for tr, te in GroupKFold(5).split(X, b, g):
        gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="balanced_accuracy", n_jobs=1).fit(X[tr], b[tr], groups=g[tr]); pred[te] = gs.predict(X[te])
    return dict(embedding=name, n_units=len(b), n_fruits=len(set(g)), balanced_accuracy_batch=balanced_accuracy_score(b, pred.astype(str)))


# ---------------------------------------------------------------- evaluation
def evaluate():
    d, T, A, pop, lab, fb, ff = context()
    y = T.loc[pop, "y"].values.astype(int); assert set(y) <= {1, 2}
    fruit = T.loc[pop, "fruit_id"].values.astype(int); bat = T.loc[pop, "Maturity_Stage"].values
    PS = pd.read_parquet(f"{R5}/probe_sessions.parquet")
    prob = {sp: {} for sp in ["lobo", "p0"]}
    for (m, sp), g in PS.groupby(["model", "split"]):
        g = g.set_index("session_id"); assert set(pop) <= set(g.index), (m, sp); prob[sp][m] = g.loc[pop, ["p_Ripe", "p_Overripe"]].values
    for sp, kind, args in [("lobo", "lobo", B), ("p0", "outer", [str(i) for i in range(5)])]:
        S = r4.knock_sessions([f"{R5}/ft/beats_ft_{kind}_{a}.npy" for a in args]); prob[sp]["A1ft"] = S.loc[pop, ["p_Ripe", "p_Overripe"]].values
    for sp in prob:
        P = prob[sp]
        P["F1"] = np.mean([P["A1"], P["V-thermal-DINO"], P["V-ms-DINO"], P["V-rgb-DINO"]], 0)
        P["F2"] = np.mean([P["A1"], P["V-thermal-SIGLIP"], P["V-ms-SIGLIP"], P["V-rgb-SIGLIP"]], 0)
        P["F4"] = np.mean([P["A1"], P["V-thermal-DINO"], P["V-ms-DINO"]], 0)
        P["F5"] = np.mean([P["V-thermal-DINO"], P["V-ms-DINO"], P["V-rgb-DINO"]], 0)
    preds = {sp: {m: 1 + v.argmax(1) for m, v in prob[sp].items()} for sp in prob}          # gated
    # references, recomputed and asserted against round 4
    L1 = pd.read_parquet(f"{V21}/lobo_cnn_sessions.parquet"); pr = lambda S_: 1 + S_.loc[pop, ["p_Ripe", "p_Overripe"]].values.argmax(1)
    preds["lobo"]["CDR_RO"] = pr(L1[L1.variant == "M0"].set_index("session_id")); preds["lobo"]["M1_RO"] = pr(L1[L1.variant == "M1"].set_index("session_id"))
    preds["lobo"]["K1"] = pr(r4.knock_sessions([f"{R4}/cnn/current/paired_lobo_{b}.npy" for b in B]))
    preds["p0"]["K1"] = pr(r4.knock_sessions([f"{R4}/cnn/current/paired_outer_{i}.npy" for i in range(5)]))
    import retrain_without_questionable as rq
    for d_, n in [("intermediate_v14/cnn_cpu_p0", "CDR_RO"), (f"{V21}/cnn/M1_outer", "M1_RO")]:
        S_ = rq._sound_sessions(d_); preds["p0"][n] = 1 + S_.loc[pop, ["p_Ripe", "p_Overripe"]].values.argmax(1)
    S4 = pd.read_csv(f"{R4}/scores.csv").query("split == 'lobo' and batch == 'pooled'").set_index("model").macro_f1_RO
    for n in ["CDR_RO", "K1"]: assert abs(ev.endpoint_scores(y, preds["lobo"][n])["macro_f1_RO"] - S4[n]) < 1e-9, n
    rows = []
    for sp in preds:
        for m, p in preds[sp].items():
            rows.append(dict(split=sp, model=m, batch="pooled", **ev.endpoint_scores(y, p)))
            for b in B:
                k = bat == b; rows.append(dict(split=sp, model=m, batch=b, **ev.endpoint_scores(y[k], p[k])))
    # per-camera scores for RGB-containing arms (camera of a session = majority camera of its RGB images)
    _, u = image_embeddings("rgb", "DINO", d, T, A, lab, index_only=True)
    if "camera" in u:
        cam = u.groupby("session_id").camera.agg(lambda v: v.value_counts().index[0]).reindex(pop).values
        for sp in preds:
            for m in ["V-rgb-DINO", "V-rgb-SIGLIP", "F1", "F2", "F3", "F5"]:
                for c in sorted(set(cam[pd.notna(cam)])):
                    k = cam == c; rows.append(dict(split=sp, model=m, batch=f"camera:{c}", **ev.endpoint_scores(y[k], preds[sp][m][k])))
    os.makedirs(R5, exist_ok=True); pd.DataFrame(rows).to_csv(f"{R5}/scores.csv", index=False)
    prim = []
    for name, cand in [("Q1", "A1"), ("Q2", "F1")]:
        r = ev.paired(y, preds["lobo"]["CDR_RO"], preds["lobo"][cand], fruit); r.update(contrast=name, candidate=cand, reference="CDR_RO"); prim.append(r)
    PR = pd.DataFrame(prim); PR["p_holm"] = ev.holm(PR.p_perm.values)
    PR["beneficial"] = (PR.d_f1_pp >= 3) & (PR.ci_f1_lo_pp > 0) & (PR.p_holm < 0.05); PR.to_csv(f"{R5}/primary_contrasts.csv", index=False)
    fam = []
    for cand in ["A1", "A2", "A3", "A4"]:
        r = ev.paired(y, preds["lobo"]["CDR_RO"], preds["lobo"][cand], fruit); r.update(candidate=cand, reference="CDR_RO"); fam.append(r)
    FA = pd.DataFrame(fam); FA["p_holm_family"] = ev.holm(FA.p_perm.values); FA.to_csv(f"{R5}/audio_family_contrasts.csv", index=False)
    pairs = [("A1", "K1"), ("A1c", "A1s"), ("A1c", "A1"), ("A1ft", "A1"), ("A1ft", "CDR_RO"), ("A1p", "A1"), ("A1w", "A1"), ("A1b", "CDR_RO"),
             ("A5", "CDR_RO"), ("A6", "CDR_RO"), ("F2", "CDR_RO"), ("F3", "CDR_RO"), ("F4", "CDR_RO"), ("F5", "CDR_RO"), ("F1", "A1"), ("F4", "F1"),
             ("V-thermal-DINO", "CDR_RO"), ("V-ms-DINO", "CDR_RO"), ("V-rgb-DINO", "CDR_RO"), ("V-thermal_mm-DINO", "V-thermal-DINO"), ("K1", "CDR_RO")]
    sec = []
    for sp in preds:
        for cand, ref in pairs:
            r = ev.paired(y, preds[sp][ref], preds[sp][cand], fruit); r.update(split=sp, candidate=cand, reference=ref); sec.append(r)
    SE = pd.DataFrame(sec); SE.to_csv(f"{R5}/secondary_contrasts.csv", index=False)
    cond = lambda c, r: bool(((lambda s: (s.d_f1_pp >= 3) & (s.ci_f1_lo_pp > 0))(SE[(SE.split == "lobo") & (SE.candidate == c) & (SE.reference == r)].iloc[0])))
    wording = {"transfers_better_than_task_cnn": cond("A1", "K1"), "within_fruit_change": cond("A1c", "A1s") and cond("A1c", "A1"),
               "fine_tuning_helps": cond("A1ft", "A1")}
    # baseline-only diagnostic with fruit-cluster bootstrap CI
    rng = np.random.default_rng(42); fr = np.unique(fruit); idx = {f: np.flatnonzero(fruit == f) for f in fr}; diag = []
    for sp in preds:
        p = preds[sp]["A1b"]; bs = []
        for _ in range(2000):
            s = np.concatenate([idx[f] for f in rng.choice(fr, len(fr))]); bs.append(ev.endpoint_scores(y[s], p[s])["macro_f1_RO"])
        diag.append(dict(split=sp, model="A1b", macro_f1_RO=ev.endpoint_scores(y, p)["macro_f1_RO"], ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5))))
    pd.DataFrame(diag).to_csv(f"{R5}/baseline_only_diagnostic.csv", index=False)
    me = open(__file__).read(); ft = open(os.path.join(HERE, "beats_ft.py")).read()
    au = {"round5_uses_storage_or_weight": bool(re.search(r"storage|weight_loss|end_of|days_from|elapsed", me[me.index("# ---------------------------------------------------------------- units"):me.index("# ---------------------------------------------------------------- evaluation")])),
          "beats_ft_uses_storage_or_weight": bool(re.search(r"storage|weight_loss|end_of|days_from|elapsed", ft)),
          "beats_ft_asserts_fruit_disjoint": "assert not set(fruit[tr]) & set(fruit[te])" in ft}
    assert not au["round5_uses_storage_or_weight"] and not au["beats_ft_uses_storage_or_weight"] and au["beats_ft_asserts_fruit_disjoint"]
    json.dump({"audit": au, "wording_conditions": wording, "population": len(pop)}, open(f"{R5}/audit.json", "w"), indent=1)
    print("written: scores, primary_contrasts, audio_family_contrasts, secondary_contrasts, baseline_only_diagnostic, audit")


# ---------------------------------------------------------------- round 6 (docs/ANALYSIS_PLAN_round6.md): intake-anchored logit shift
R6 = os.environ.get("ROUND6_DIR", "results/round6")
R6_ARMS = ["A1", "A5", "V-thermal-DINO", "V-ms-DINO", "V-rgb-DINO", "F3"]      # the only arms refitted; identical estimator, seeds and training rows
R6_SHIFTED = ["F3", "A1", "A5", "F1"]                                        # R6 = anchored F3 (primary); the others are exploratory secondaries
EPS = 1e-12


def fit6():
    """Refit R6_ARMS with the round-5 probe (same training rows, estimator, grid, inner CV), predicting the endpoint units of the test split
    AND the label-free intake sessions of every batch. Control: the endpoint-session probabilities must reproduce results/round5/probe_sessions.parquet."""
    t0 = time.time(); d, T, A, pop, lab, fb, ff = context(); rows = []
    au = d[["session_id", "fruit_id", "batch", "label", "intake", "has_base"]].copy(); allsel = np.ones(len(au), bool); SESS = {}
    for code in ["A1", "A5"]:
        Z = audio_recordings(code, d); rows += probe(Z, au, allsel, ff, code, intake_rows=True)
        if code == "A1": SESS["A1"] = pd.DataFrame(Z).groupby(d.session_id.values).mean()
        print(code, "done", f"{time.time()-t0:.0f}s", flush=True)
    for mod in ["thermal", "ms", "rgb"]:
        X, u = image_embeddings(mod, "DINO", d, T, A, lab)
        rows += probe(X, u, np.ones(len(u), bool), ff, f"V-{mod}-DINO", intake_rows=True)
        SESS[mod] = pd.DataFrame(X).groupby(u.session_id.values).mean(); print(mod, "DINO done", f"{time.time()-t0:.0f}s", flush=True)
    S = sorted(set.intersection(*[set(v.index) for v in SESS.values()]))                          # as fit(): sessions with all four modalities
    su = pd.DataFrame({"session_id": S}); su["fruit_id"] = su.session_id.str.split("_").str[0].astype(int)
    su["batch"] = su.fruit_id.map(fb); su["label"] = su.session_id.map(lab); su["intake"] = su.session_id.map(A.anchor).fillna(False).astype(bool)
    su["has_base"] = su.fruit_id.isin(set(d.fruit_id[d.intake]))
    blocks = [SESS[k].loc[S].values for k in ["A1", "thermal", "ms", "rgb"]]
    rows += probe_blocks(blocks, su, ff, "F3", intake_rows=True)
    P6 = pd.concat(rows, ignore_index=True)
    bt = pd.read_csv("data/durian_mm_capture_folds.csv").groupby("fruit_id").Maturity_Stage.first()
    P6["batch"] = P6.session_id.str.split("_").str[0].astype(int).map(bt).values
    # control: endpoint rows identical to round 5
    P5 = pd.read_parquet(f"{R5}/probe_sessions.parquet"); chk = {}
    for (m, sp), g in P6[P6.role == "endpoint"].groupby(["model", "split"]):
        ref = P5[(P5.model == m) & (P5.split == sp)].set_index("session_id"); g = g.set_index("session_id")
        assert set(g.index) == set(ref.index), (m, sp, len(g), len(ref))
        dmax = float(np.abs(g.loc[ref.index, ["p_Ripe", "p_Overripe"]].values - ref[["p_Ripe", "p_Overripe"]].values).max())
        chk[f"{m}|{sp}"] = dmax; assert dmax < 1e-9, (m, sp, dmax)
    os.makedirs(R6, exist_ok=True); P6.to_parquet(f"{R6}/probe_sessions_with_intake.parquet", index=False)
    n = P6.groupby(["model", "split", "role"]).size().rename("n").reset_index()
    json.dump({"identity_max_abs_diff_vs_round5": chk, "rows": int(len(P6)),
               "counts": n.to_dict(orient="records")}, open(f"{R6}/fit6_identity_check.json", "w"), indent=1)
    print("fit6 written", len(P6), "rows; endpoint identity max|diff|", f"{max(chk.values()):.1e}", f"{time.time()-t0:.0f}s")


def _logit(q):
    q = np.clip(np.asarray(q, float), EPS, 1 - EPS); return np.log(q[:, 1]) - np.log(q[:, 0])        # R/O logit = log p(Overripe) / p(Ripe)


def _anchored(P6, model, pop, bat):
    """round-1 M3 rule on session scores. For held-out batch b: shifted logit = logit + (mean R/O logit of the intake sessions of the two
    training batches - mean R/O logit of the intake sessions of b), all three quantities from the model trained for split b (training
    batches only). Intake sessions are label-free (anchor_sessions.csv anchor == True). Gated prediction: Overripe if shifted logit > 0."""
    pred = np.zeros(len(pop), int); info = {}; L = P6[(P6.model == model) & (P6.split == "lobo")]
    for b in B:
        g = L[L.held_out == b]; k = bat == b
        ep = g[g.role == "endpoint"].set_index("session_id"); assert set(np.asarray(pop)[k]) <= set(ep.index), (model, b)
        it = g[g.role == "intake_test"]; tr = g[g.role == "intake_train"]
        assert (it.batch == b).all() and (tr.batch != b).all() and len(it) > 0 and len(tr) > 0, (model, b, len(it), len(tr))
        sh = float(_logit(tr[["p_Ripe", "p_Overripe"]].values).mean() - _logit(it[["p_Ripe", "p_Overripe"]].values).mean())
        z = _logit(ep.loc[np.asarray(pop)[k], ["p_Ripe", "p_Overripe"]].values) + sh
        pred[k] = np.where(z > 0, 2, 1); info[b] = dict(shift_logit=sh, n_intake_test=int(len(it)), n_intake_train=int(len(tr)))
    return pred, info


def _median_centred(prob, bat):
    """post-hoc rule already seen in round 5 (exploratory): within each held-out batch, Overripe if the R/O logit exceeds the batch median"""
    z = _logit(prob); pred = np.zeros(len(z), int)
    for b in B:
        k = bat == b; pred[k] = np.where(z[k] > np.median(z[k]), 2, 1)
    return pred


def _f1_rows(P6):
    """F1 (equal-weight probability mean of A1 and the three DINO arms) as session rows with role/held_out, for the sessions all four have"""
    L = P6[P6.split == "lobo"]; keys = ["session_id", "role", "held_out", "batch"]; parts = []
    for m in ["A1", "V-thermal-DINO", "V-ms-DINO", "V-rgb-DINO"]:
        parts.append(L[L.model == m].set_index(keys)[["p_Ripe", "p_Overripe"]])
    idx = parts[0].index
    for q in parts[1:]: idx = idx.intersection(q.index)
    F = sum(q.loc[idx] for q in parts) / 4; F = F.reset_index(); F["model"] = "F1"; F["split"] = "lobo"; return F


def _references_lobo(pop, y):
    """CDR-RO (round-1 M0) and K1 (round-4 current-knock CNN) gated predictions, asserted equal to the round-4 pooled scores"""
    L1 = pd.read_parquet(f"{V21}/lobo_cnn_sessions.parquet"); pr = lambda S_: 1 + S_.loc[pop, ["p_Ripe", "p_Overripe"]].values.argmax(1)
    refs = {"CDR_RO": pr(L1[L1.variant == "M0"].set_index("session_id")), "K1": pr(r4.knock_sessions([f"{R4}/cnn/current/paired_lobo_{b}.npy" for b in B]))}
    S4 = pd.read_csv(f"{R4}/scores.csv").query("split == 'lobo' and batch == 'pooled'").set_index("model").macro_f1_RO
    for n in refs: assert abs(ev.endpoint_scores(y, refs[n])["macro_f1_RO"] - S4[n]) < 1e-9, n
    return refs


def evaluate6():
    d, T, A, pop, lab, fb, ff = context()
    y = T.loc[pop, "y"].values.astype(int); assert set(y) <= {1, 2}
    fruit = T.loc[pop, "fruit_id"].values.astype(int); bat = T.loc[pop, "Maturity_Stage"].values
    P6 = pd.read_parquet(f"{R6}/probe_sessions_with_intake.parquet"); P6 = pd.concat([P6, _f1_rows(P6)], ignore_index=True)
    # controls: (i) endpoint rows of the refitted arms reproduce round 5; (ii) the unshifted arms reproduce the round-5 LOBO scores
    P5 = pd.read_parquet(f"{R5}/probe_sessions.parquet"); S5 = pd.read_csv(f"{R5}/scores.csv").query("split == 'lobo' and batch == 'pooled'").set_index("model").macro_f1_RO
    E = P6[(P6.role == "endpoint") & (P6.split == "lobo")]; prob = {}; ident = {}
    for m in R6_ARMS + ["F1"]:
        g = E[E.model == m].set_index("session_id"); assert set(pop) <= set(g.index), m; prob[m] = g.loc[pop, ["p_Ripe", "p_Overripe"]].values
        if m in R6_ARMS:
            ref = P5[(P5.model == m) & (P5.split == "lobo")].set_index("session_id").loc[pop, ["p_Ripe", "p_Overripe"]].values
            ident[m] = float(np.abs(prob[m] - ref).max()); assert ident[m] < 1e-9, (m, ident[m])
    preds = {m: 1 + prob[m].argmax(1) for m in prob}
    for m in R6_SHIFTED: assert abs(ev.endpoint_scores(y, preds[m])["macro_f1_RO"] - S5[m]) < 1e-9, m
    preds.update(_references_lobo(pop, y))
    shifts = {}
    for m in R6_SHIFTED:
        nm = "R6" if m == "F3" else f"{m}-anch"; preds[nm], shifts[nm] = _anchored(P6, m, pop, bat)
        preds[f"{m}-med"] = _median_centred(prob[m], bat)
    rows = []
    for m, p in preds.items():
        rows.append(dict(split="lobo", model=m, batch="pooled", **ev.endpoint_scores(y, p)))
        for b in B:
            k = bat == b; rows.append(dict(split="lobo", model=m, batch=b, **ev.endpoint_scores(y[k], p[k])))
    SC = pd.DataFrame(rows); os.makedirs(R6, exist_ok=True); SC.to_csv(f"{R6}/scores.csv", index=False)
    # primary: single contrast, Holm is the identity
    r = ev.paired(y, preds["CDR_RO"], preds["R6"], fruit); r.update(contrast="R6", candidate="R6", reference="CDR_RO")
    PR = pd.DataFrame([r]); PR["p_holm"] = ev.holm(PR.p_perm.values)
    PR["beneficial"] = (PR.d_f1_pp >= 3) & (PR.ci_f1_lo_pp > 0) & (PR.p_holm < 0.05); PR.to_csv(f"{R6}/primary_contrast.csv", index=False)
    pairs = [("R6", "F3"), ("R6", "K1"), ("A1-anch", "CDR_RO"), ("A1-anch", "A1"), ("A5-anch", "CDR_RO"), ("A5-anch", "A5"), ("F1-anch", "CDR_RO"),
             ("F1-anch", "F1"), ("R6", "F1-anch"), ("R6", "F3-med"), ("F3-med", "CDR_RO"), ("A1-med", "CDR_RO"), ("A5-med", "CDR_RO"), ("F1-med", "CDR_RO")]
    sec = []
    for cand, ref in pairs:
        r = ev.paired(y, preds[ref], preds[cand], fruit); r.update(split="lobo", candidate=cand, reference=ref, status="exploratory_unadjusted"); sec.append(r)
    SE = pd.DataFrame(sec); SE.to_csv(f"{R6}/secondary_contrasts.csv", index=False)
    s6 = SE[(SE.candidate == "R6") & (SE.reference == "F3")].iloc[0]; ove = SC[(SC.model == "R6") & (SC.batch == "Overmature")].macro_f1_RO.iloc[0]
    wording = {"shift_correction_recovers_overmature": bool(ove >= 0.70 and s6.d_f1_pp >= 3 and s6.ci_f1_lo_pp > 0)}
    me = open(__file__).read(); sec6 = me[me.index("# ---------------------------------------------------------------- round 6"):me.index("    me = open(__file__).read(); sec6")]
    au = {"round6_uses_storage_or_weight": bool(re.search(r"storage|weight_loss|end_of|days_from|elapsed", sec6)),
          "endpoint_identity_max_abs_diff_vs_round5": ident, "p0": "not evaluated for anchored arms (plan Section 4)"}
    assert not au["round6_uses_storage_or_weight"]
    json.dump({"audit": au, "shifts": shifts, "wording_conditions": wording, "population": len(pop),
               "primary_beneficial": bool(PR.beneficial.iloc[0])}, open(f"{R6}/audit.json", "w"), indent=1)
    print("written: scores, primary_contrast, secondary_contrasts, audit;", len(SC), "score rows")


if __name__ == "__main__":
    {"fit": fit, "diagnose": diagnose, "evaluate": evaluate, "fit6": fit6, "evaluate6": evaluate6}[sys.argv[1]]()
