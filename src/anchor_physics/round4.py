"""Round 4 (docs/ANALYSIS_PLAN_round4.md, tag prereg-round4): fruit-specific acoustic baseline.

  python src/anchor_physics/round4.py classical    L1-L5 feature-level comparators (LOBO and ten fruit partitions)
  python src/anchor_physics/round4.py evaluate     K1-K4 (paired_cnn.py outputs), L1-L5, references; contrasts P1/P2; audit
Population: the 104 endpoint sessions whose fruit has a verified intake recording. Every method is gated (intake Unripe; endpoints
Ripe/Overripe). Outputs in results/round4/ (override with ROUND4_DIR).
"""
import os, sys, json, re
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "label_sensitivity")); sys.path.insert(0, os.path.join(HERE, "..", "nested_selection"))
import lobo_classical as lc, evaluate as ev
V21 = os.environ.get("ROUND_V21", "results/v21"); R4 = os.environ.get("ROUND4_DIR", "results/round4"); lc.OUT = V21
PC = ev.PC; B = ev.BATCHES; KS = {"K1": "current", "K2": "baseline", "K3": "concat", "K4": "paired", "K5": "change", "K6": "shuffle"}
PHYS = ["f_dom", "f_first", "centroid", "decay20", "inv_q", "stiffness"]; MFCC = [f"mfcc{i:02d}_mean__mean" for i in range(1, 14)]


def recordings():
    """verified recordings with session, fruit, batch, common label, device, features F, intake flag, and the fruit baseline F_0"""
    p = pd.read_parquet(f"{V21}/physics_recordings.parquet").set_index("file")
    s = pd.read_parquet(lc.CFG["sound"]["features"]).set_index("file")                       # the 432 verified recordings
    man = pd.read_csv("data/sound_recording_manifest.csv").set_index("file")
    lab = pd.read_csv("data/session_labels_common.csv").set_index("session_id").label_common
    A = pd.read_csv(f"{V21}/anchor_sessions.csv"); anc = set(A.session_id[A.anchor])
    d = pd.DataFrame(index=s.index); d["session_id"] = d.index.map(man.session_id); d["fruit_id"] = d.session_id.str.split("_").str[0].astype(int)
    d["batch"] = s.Maturity_Stage; d["label"] = d.session_id.map(lab); d["intake"] = d.session_id.isin(anc)
    d["device_iphone"] = (d.index.map(p.device) == "iphone").astype(float)
    for c in PHYS: d[c] = d.index.map(p[c]).astype(float)
    for c in MFCC: d[c] = s[c].astype(float)
    F = PHYS + MFCC; base = d[d.intake].groupby("fruit_id")[F].mean()
    for c in F: d[c + "_0"] = d.fruit_id.map(base[c])
    d["has_base"] = d.fruit_id.isin(base.index)
    A2 = A[A.anchor].copy(); A2["date"] = pd.to_datetime(A2.date); intake_date = A2.groupby("fruit_id").date.min()   # T1 only
    d["elapsed_days"] = (pd.to_datetime(d.session_id.str.split("_").str[1]) - d.fruit_id.map(intake_date)).dt.days.astype(float)
    d["d_log_fdom"] = np.log(d.f_dom / d.f_dom_0); d["d_log_ffirst"] = np.log(d.f_first / d.f_first_0)
    for c in ["centroid", "decay20", "inv_q"]: d["d_" + c] = d[c] - d[c + "_0"]
    return d.reset_index().rename(columns={"index": "file"}), F


def feature_sets(F):
    D = ["d_log_fdom", "d_log_ffirst", "d_centroid", "d_decay20", "d_inv_q"]
    return {"L1": F + ["device_iphone"], "L2": [c + "_0" for c in F], "L3": F + [c + "_0" for c in F] + ["device_iphone"],
            "L4": D + ["device_iphone"], "L5": F + D + ["device_iphone"], "T1": ["elapsed_days"]}


def classical():
    d, F = recordings(); FS = feature_sets(F)
    ep = d[~d.intake & d.has_base].reset_index(drop=True)
    y = ep.label.map({"Ripe": 1, "Overripe": 2}); trainable = y.notna().values; y = y.fillna(0).astype(int).values
    parts = pd.read_csv("src/nested_selection/partitions.csv").set_index("fruit_id")
    splits = [("lobo", b, ep.batch.values != b, ep.batch.values == b) for b in B]
    for p in [f"p{i}" for i in range(10)]:
        fo = ep.fruit_id.map(parts[p]).values
        splits += [(p, str(k), fo != k, fo == k) for k in range(5)]
    rows = []
    for name, cols in FS.items():
        X = ep[cols].astype(float).values; assert np.isfinite(X).all(), name
        for sp, arg, trm, tem in splits:
            tr = trm & trainable; est, grid = lc.estimator("logreg")
            gs = GridSearchCV(est, grid, cv=GroupKFold(3), scoring="f1_macro", n_jobs=1).fit(X[tr], y[tr], groups=ep.fruit_id.values[tr])
            q = gs.predict_proba(X[tem]); assert list(gs.classes_) == [1, 2]
            t = pd.DataFrame({"session_id": ep.session_id.values[tem], "p_Ripe": q[:, 0], "p_Overripe": q[:, 1]})
            s = t.groupby("session_id")[["p_Ripe", "p_Overripe"]].mean().reset_index(); s["model"] = name; s["split"] = sp; rows.append(s)
        print(name, "done", flush=True)
    os.makedirs(R4, exist_ok=True); pd.concat(rows, ignore_index=True).to_parquet(f"{R4}/classical_sessions.parquet", index=False)


def knock_sessions(npy_paths):
    """K-model knock probabilities (several jobs) -> recording mean -> session mean over verified recordings"""
    kn = pd.read_parquet("work/raw/sound/features_sound_knocks.parquet")[["file"]] if os.path.exists("work/raw/sound/features_sound_knocks.parquet") \
        else pd.read_parquet("intermediate_v14/features_sound_knocks.parquet")[["file"]]
    P = np.full((len(kn), 3), np.nan)
    for f in npy_paths:
        a = np.load(f); m = np.isfinite(a[:, 1]); assert not (np.isfinite(P[m, 1])).any(), f; P[m] = a[m]
    excl = set(open("data/sound_exclude_wrong_session.txt").read().split()) if os.path.exists("data/sound_exclude_wrong_session.txt") else set()
    man = pd.read_csv("data/sound_recording_manifest.csv").set_index("file")
    t = pd.DataFrame(P[:, 1:], columns=["p_Ripe", "p_Overripe"]); t["file"] = kn.file.values; t = t[np.isfinite(t.p_Ripe) & ~t.file.isin(excl)]
    r = t.groupby("file")[["p_Ripe", "p_Overripe"]].mean(); r["session_id"] = r.index.map(man.session_id)
    return r.groupby("session_id")[["p_Ripe", "p_Overripe"]].mean()


def evaluate():
    d, F = recordings()
    T = pd.read_csv("results/session_table.csv", low_memory=False); T = T[T.n_modalities == 4].set_index("session_id")
    A = pd.read_csv(f"{V21}/anchor_sessions.csv").set_index("session_id")
    pop = sorted(set(d.session_id[~d.intake & d.has_base]) & set(T.index[~A.loc[T.index, "anchor"].astype(bool)]))
    assert len(pop) == 104, len(pop)
    y = T.loc[pop, "y"].values.astype(int); assert set(y) <= {1, 2}
    fruit = T.loc[pop, "fruit_id"].values.astype(int); bat = T.loc[pop, "Maturity_Stage"].values; fold = T.loc[pop, "fold_fruit"].values.astype(int)
    pr = lambda S: 1 + S.loc[pop, ["p_Ripe", "p_Overripe"]].values.argmax(1)               # gated: endpoints Ripe/Overripe
    preds = {"lobo": {}, "p0": {}}
    for k, h in KS.items():
        preds["lobo"][k] = pr(knock_sessions([f"{R4}/cnn/{h}/paired_lobo_{b}.npy" for b in B]))
        preds["p0"][k] = pr(knock_sessions([f"{R4}/cnn/{h}/paired_outer_{i}.npy" for i in range(5)]))
    C = pd.read_parquet(f"{R4}/classical_sessions.parquet")
    for m in ["L1", "L2", "L3", "L4", "L5", "T1"]:
        preds["lobo"][m] = pr(C[(C.model == m) & (C.split == "lobo")].set_index("session_id"))
        preds["p0"][m] = pr(C[(C.model == m) & (C.split == "p0")].set_index("session_id"))
    L1 = pd.read_parquet(f"{V21}/lobo_cnn_sessions.parquet")
    for v, n in [("M0", "CDR_RO"), ("M1", "M1_RO")]:
        preds["lobo"][n] = pr(L1[L1.variant == v].set_index("session_id").rename(columns={}))
    import retrain_without_questionable as rq
    for d_, n in [("intermediate_v14/cnn_cpu_p0", "CDR_RO"), (f"{V21}/cnn/M1_outer", "M1_RO")]:
        S = rq._sound_sessions(d_); preds["p0"][n] = 1 + S.loc[pop, ["p_Ripe", "p_Overripe"]].values.argmax(1)
    # control: CDR-RO on the 104 sessions equals round 1 restricted (gated M0 argmax over Ripe/Overripe)
    S1 = L1[L1.variant == "M0"].set_index("session_id"); assert (preds["lobo"]["CDR_RO"] == 1 + S1.loc[pop, ["p_Ripe", "p_Overripe"]].values.argmax(1)).all()
    rows = []
    for sp in ["lobo", "p0"]:
        for m, p in preds[sp].items():
            rows.append(dict(split=sp, model=m, batch="pooled", **ev.endpoint_scores(y, p)))
            for b in B:
                k = bat == b; rows.append(dict(split=sp, model=m, batch=b, **ev.endpoint_scores(y[k], p[k])))
    C10 = C[C.split.str.match(r"p\d$")]
    for m in ["L1", "L2", "L3", "L4", "L5"]:
        f10 = [ev.endpoint_scores(y, pr(C10[(C10.model == m) & (C10.split == f"p{i}")].set_index("session_id")))["macro_f1_RO"] for i in range(10)]
        rows.append(dict(split="ten_partitions_mean", model=m, batch="pooled", macro_f1_RO=float(np.mean(f10)), n=len(y)))
    SC = pd.DataFrame(rows); os.makedirs(R4, exist_ok=True); SC.to_csv(f"{R4}/scores.csv", index=False)
    prim = []
    for name, cand, ref in [("P1", "K4", "K1"), ("P2", "K4", "CDR_RO")]:
        r = ev.paired(y, preds["lobo"][ref], preds["lobo"][cand], fruit); r.update(contrast=name, candidate=cand, reference=ref); prim.append(r)
    PR = pd.DataFrame(prim); PR["p_holm"] = ev.holm(PR.p_perm.values)
    ok = (PR.d_f1_pp >= 3) & (PR.ci_f1_lo_pp > 0) & (PR.p_holm < 0.05); PR["meets_rule"] = ok; PR["K4_beneficial"] = bool(ok.all())
    PR.to_csv(f"{R4}/primary_contrasts.csv", index=False)
    sec = []
    for sp in ["lobo", "p0"]:
        for cand, ref in [("K3", "K1"), ("K4", "K3"), ("K2", "K1"), ("K1", "CDR_RO"), ("K3", "CDR_RO"), ("M1_RO", "CDR_RO"), ("L1", "CDR_RO"),
                          ("L3", "L1"), ("L4", "L1"), ("L5", "L1"), ("L5", "CDR_RO"), ("K4", "L5"), ("K4", "M1_RO"), ("K4", "K1"), ("K4", "CDR_RO"),
                          ("K5", "K1"), ("K4", "K6"), ("K6", "K1"), ("K4", "T1"), ("T1", "CDR_RO")]:
            r = ev.paired(y, preds[sp][ref], preds[sp][cand], fruit); r.update(split=sp, candidate=cand, reference=ref); sec.append(r)
    SE = pd.DataFrame(sec); SE.to_csv(f"{R4}/secondary_contrasts.csv", index=False)
    fs = SE[(SE.split == "lobo") & (SE.candidate == "K4") & (SE.reference == "K6")].iloc[0]
    fruit_specific = bool((fs.d_f1_pp >= 3) & (fs.ci_f1_lo_pp > 0))
    # baseline-only diagnostic: macro-F1 of K2 / L2 with fruit-cluster bootstrap CI
    diag = []
    rng = np.random.default_rng(42); fr = np.unique(fruit); idx = {f: np.flatnonzero(fruit == f) for f in fr}
    for sp in ["lobo", "p0"]:
        for m in ["K2", "L2"]:
            p = preds[sp][m]; bs = []
            for _ in range(2000):
                s = np.concatenate([idx[f] for f in rng.choice(fr, len(fr))]); bs.append(ev.endpoint_scores(y[s], p[s])["macro_f1_RO"])
            diag.append(dict(split=sp, model=m, macro_f1_RO=ev.endpoint_scores(y, p)["macro_f1_RO"], ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5))))
    pd.DataFrame(diag).to_csv(f"{R4}/baseline_only_diagnostic.csv", index=False)
    # cost of the baseline (verified intake recordings and knocks per fruit of the population)
    kn = pd.read_parquet("work/raw/sound/features_sound_knocks.parquet")[["file"]] if os.path.exists("work/raw/sound/features_sound_knocks.parquet") else pd.read_parquet("intermediate_v14/features_sound_knocks.parquet")[["file"]]
    b = d[d.intake & d.fruit_id.isin(set(fruit))]; kc = kn.file.value_counts()
    cost = dict(fruits=int(len(set(fruit))), intake_recordings_median=float(b.groupby("fruit_id").size().median()),
                intake_knocks_median=float(b.file.map(kc).groupby(b.fruit_id).sum().median()))
    # leakage audit (code-level)
    src = open(os.path.join(HERE, "paired_cnn.py")).read(); me = open(__file__).read()
    au = {"paired_cnn_uses_storage_or_weight": bool(re.search(r"storage|weight_loss|end_of|days_from|Days", src)),
          "round4_features_use_storage_or_weight": bool(re.search(r"storage|weight_loss|end_of|days_from", me[me.index("def recordings"):me.index("def classical")])),
          "baseline_from_label_free_intake_rule": "sess.isin(anc_sess)" in src and "A.anchor" in src}
    assert not au["paired_cnn_uses_storage_or_weight"] and not au["round4_features_use_storage_or_weight"] and au["baseline_from_label_free_intake_rule"]
    pk = pd.read_parquet(f"{V21}/physics_knocks.parquet"); man2 = pd.read_csv("data/sound_recording_manifest.csv").set_index("file")
    pk = pk[pk.file.isin(set(d.file[d.intake & d.fruit_id.isin(set(fruit))]))].copy(); pk["fruit_id"] = pk.file.map(man2.session_id).str.split("_").str[0].astype(int)
    noise = float(pk.groupby("fruit_id").f_dom.apply(lambda v: np.log(v).std(ddof=1)).median())
    chg = float(np.abs(d[~d.intake & d.has_base & d.session_id.isin(pop)].d_log_fdom).median())
    json.dump({"audit": au, "cost": cost, "population": len(pop), "fruit_specific_condition_met": fruit_specific,
               "intake_within_fruit_sd_log_fdom_median": noise, "intake_to_endpoint_abs_log_fdom_change_median": chg},
              open(f"{R4}/audit_and_cost.json", "w"), indent=1)
    print("written: scores, primary_contrasts, secondary_contrasts, baseline_only_diagnostic, audit_and_cost")


if __name__ == "__main__":
    {"classical": classical, "evaluate": evaluate}[sys.argv[1]]()
