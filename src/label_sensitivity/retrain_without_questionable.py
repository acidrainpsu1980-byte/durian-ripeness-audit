"""Round-3 comment 4: label-policy sensitivity with the questionable sessions removed from TRAINING as well as from scoring.

The earlier analysis (Table 11, src/sensitivity/sensitivity_tables.py) removed questionable sessions from scoring only, while the
models had been trained with them. Here the principal sound model (development pipeline: knock CNN) and the principal fusion
(thermal + sound + RGB mean rule: thermal SVM-RBF, knock CNN, RGB EfficientNet-B0 probe) are retrained on the primary partition
(p0, the frozen fruit-grouped 5-fold split) after removing, from every training fold, all captures of the questionable sessions,
and are scored on the same remaining sessions with the same fruit folds as the score-only analysis.

Nested session sets (defined over all 315 sessions of the 164 analysis fruits, as in Section 2.9):
  excl_disputed   12 sessions whose file-name codes disagree (9 of them among the 268 evaluation sessions)        -> 259 scored
  excl_loginc     + 7 unanimous sessions inconsistent with the fruit log                                           -> 252 scored
  excl_harvest    + 15 Unripe sessions whose fruit showed softening or aroma at harvest (unripe = v1.2 majority)   -> 237 scored
Fruit folds are unchanged (a fruit keeps its fold even if one of its sessions is removed).

usage (repo root; DURIAN_INTERMEDIATE=intermediate_common, OMP_NUM_THREADS>=4):
  python src/label_sensitivity/retrain_without_questionable.py prepare  OUT CNN_WORK   # sets, CNN exclusion lists, distributions
  python src/label_sensitivity/retrain_without_questionable.py classical OUT           # thermal SVM-RBF and RGB probe refits
  (cd CNN_WORK && CNN_THREADS=3 OMP_NUM_THREADS=3 CNN_EXCLUDE_FILE=OUT/cnn_exclude_<set>.txt CNN_OUT_DIR=OUT/cnn/<set> \\
       python REPO/src/common_label/cnn_common.py outer:0 outer:1 outer:2 outer:3 outer:4)                          # knock CNN
  python src/label_sensitivity/retrain_without_questionable.py evaluate OUT            # metrics and paired comparisons
CNN_WORK is the sound working directory of the raw-data level (work/raw/sound: handoff/mels.npy, rec2_meta.parquet, ...).
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "nested_selection"))
import cands, nsel, stats4

SETS = ["excl_disputed", "excl_loginc", "excl_harvest"]
CLS = ["p_Unripe", "p_Ripe", "p_Overripe"]; NAMES = ["Unripe", "Ripe", "Overripe"]
MODELS = [("thermal", "svm_rbf"), ("rgb", "efficientnetb0_probe")]
PART = pd.read_csv(os.path.join(HERE, "..", "nested_selection", "partitions.csv")).set_index("fruit_id")["p0"]
N_PERM, N_BOOT, SEED = 20000, 2000, 42


def question_sets():
    SL = pd.read_csv("data/session_labels_common.csv")
    log = pd.read_csv("data/zenodo/durian_characteristics_cleaned.csv"); log.insert(0, "fruit_id", np.arange(1, len(log) + 1))
    SL = SL.join(log.set_index("fruit_id")[["Pliability_Harvest", "Aroma_Harvest", "Maturity_Stage"]], on="fruit_id")
    SL["harvest_sign"] = (SL.label_v12_majority == "Unripe") & ((SL.Pliability_Harvest > 1) | (SL.Aroma_Harvest > 1))
    ses = pd.read_csv("audit/sessions.csv").set_index("session_id")
    SL["in_evaluation_268"] = SL.session_id.map(ses.in_evaluation_268).astype(bool)
    q1 = ~SL.unanimous.astype(bool); q2 = q1 | SL.log_inconsistent.astype(bool); q3 = q2 | SL.harvest_sign
    SL["q_excl_disputed"], SL["q_excl_loginc"], SL["q_excl_harvest"] = q1, q2, q3
    assert [int((SL.in_evaluation_268 & ~q).sum()) for q in (q1, q2, q3)] == [259, 252, 237]
    return SL[["session_id", "fruit_id", "Maturity_Stage", "label_common", "in_evaluation_268", "unanimous", "log_inconsistent",
               "harvest_sign", "q_excl_disputed", "q_excl_loginc", "q_excl_harvest"]]


def prepare(out, work):
    os.makedirs(out, exist_ok=True); Q = question_sets(); Q.to_csv(f"{out}/question_sets.csv", index=False)
    man = pd.read_csv("data/sound_recording_manifest.csv")
    base = set(open(f"{work}/handoff/exclude_recordings.txt").read().split()); assert len(base) == 106
    for s in SETS:
        qs = set(Q.session_id[Q[f"q_{s}"]])
        extra = set(man.file[man.session_id.isin(qs)]) - base
        open(f"{out}/cnn_exclude_{s}.txt", "w").write("\n".join(sorted(base | extra)) + "\n")
    # distributions: evaluation sessions (class x batch) and training captures per modality (class)
    rows = []
    for s in ["all"] + SETS:
        keep = Q.in_evaluation_268 & (True if s == "all" else ~Q[f"q_{s}"])
        d = Q[keep]
        for b in ["Immature", "Mature", "Overmature", "All"]:
            e = d if b == "All" else d[d.Maturity_Stage == b]
            c = e.label_common.value_counts()
            rows.append(dict(session_set=s, batch=b, sessions=len(e), fruits=e.fruit_id.nunique(), Unripe=int(c.get("Unripe", 0)),
                             Ripe=int(c.get("Ripe", 0)), Overripe=int(c.get("Overripe", 0))))
    pd.DataFrame(rows).to_csv(f"{out}/distribution_evaluation_sessions.csv", index=False)
    rows = []
    for mod in ["thermal", "sound", "rgb"]:
        m, _ = cands.load(mod)
        for s in ["all"] + SETS:
            qs = set() if s == "all" else set(Q.session_id[Q[f"q_{s}"]])
            mm = m[~m.session_id.astype(str).isin(qs)]
            c = mm.y.value_counts()
            rows.append(dict(modality=mod, session_set=s, captures=len(mm), sessions=mm.session_id.nunique(),
                             **{NAMES[k]: int(c.get(k, 0)) for k in range(3)}))
    pd.DataFrame(rows).to_csv(f"{out}/distribution_training_captures.csv", index=False)
    print(pd.read_csv(f"{out}/distribution_evaluation_sessions.csv").to_string(index=False))


def classical(out):
    Q = pd.read_csv(f"{out}/question_sets.csv")
    for mod, cand in MODELS:
        meta, X = cands.load(mod); fold = meta.fruit_id.map(PART).values.astype(int); sid = meta.session_id.astype(str).values
        for s in ["none"] + SETS:
            qs = set() if s == "none" else set(Q.session_id[Q[f"q_{s}"]])
            o = np.full((len(meta), 3), np.nan)
            for k in range(5):
                tr = np.flatnonzero((fold != k) & ~np.isin(sid, list(qs))); te = np.flatnonzero(fold == k)
                o[te] = cands.fit_predict(mod, cand, meta, X, tr, te)
            os.makedirs(f"{out}/fits/{s}", exist_ok=True)
            np.savez_compressed(f"{out}/fits/{s}/{mod}__{cand}.npz", files=meta.file.values.astype(str), outer=o)
            if s == "none":                       # control: no exclusion reproduces the shipped primary-partition fit exactly
                ref = np.load(f"intermediate_v14/fits/p0/{mod}__{cand}.npz", allow_pickle=True)["outer"]
                d = float(np.nanmax(np.abs(ref - o))); print(mod, cand, "control max|diff|", d, flush=True); assert d == 0.0
            print(mod, cand, s, "done", flush=True)


def _session(mod, arr):
    m, _ = cands.load(mod)
    t = pd.DataFrame(arr, columns=CLS); t["session_id"] = m.session_id.astype(str).values
    return t.dropna().groupby("session_id")[CLS].mean()


def _sound_sessions(cnn_dir):
    kn = pd.read_parquet("intermediate_v14/features_sound_knocks.parquet")[["file"]]
    ok = np.full((len(kn), 3), np.nan)
    for k in range(5):
        a = np.load(f"{cnn_dir}/cnn_common_outer_{k}.npy"); ok[~np.isnan(a[:, 0])] = a[~np.isnan(a[:, 0])]
    m, _ = cands.load("sound")
    t = pd.DataFrame(ok, columns=CLS); t["file"] = kn.file.values
    rec = t.groupby("file")[CLS].mean().reindex(m.file.values)
    rec["session_id"] = m.session_id.astype(str).values
    return rec.groupby("session_id")[CLS].mean()


def macro_f1_rows(yt, P):
    out = np.zeros(P.shape[0])
    for c in range(3):
        tp = ((P == c) & (yt == c)).sum(1); pp = (P == c).sum(1); tt = (yt == c).sum()
        out += np.where(pp + tt > 0, 2 * tp / np.maximum(pp + tt, 1), 0.0)
    return out / 3


def fruit_perm(yt, pr_ref, pr_cand, cl, stat):
    T0 = stat(yt, pr_cand[None])[0] - stat(yt, pr_ref[None])[0]
    u, inv = np.unique(cl, return_inverse=True); rng = np.random.default_rng(SEED)
    sw = rng.integers(0, 2, size=(N_PERM, len(u))).astype(bool)[:, inv]
    Tp = stat(yt, np.where(sw, pr_ref[None], pr_cand[None])) - stat(yt, np.where(sw, pr_cand[None], pr_ref[None]))
    return float(T0), float((np.sum(np.abs(Tp) >= abs(T0) - 1e-12) + 1) / (N_PERM + 1))


def evaluate(out):
    Q = pd.read_csv(f"{out}/question_sets.csv").set_index("session_id")
    so = pd.read_parquet("results/v17/matched_space_session_oof.parquet"); so = so[so.partition == "p0cpu"]
    base = so[so.estimate == "fixed_sound"].sort_values("session_id").set_index("session_id")
    sid = base.index.values; y = base.y.values.astype(int); fruit = base.fruit_id.values; fold = base.outer_fold.values
    P = lambda e: so[so.estimate == e].set_index("session_id").loc[sid, CLS].values
    score_only = {"sound": P("fixed_sound"), "thermal": P("fixed_thermal"), "rgb": P("fixed_rgb"), "tsr_mean": P("fixed_thermal+sound+rgb|mean")}
    # controls: the aggregation path reproduces the shipped session probabilities from the shipped capture predictions
    assert np.abs(_sound_sessions("intermediate_v14/cnn_cpu_p0").loc[sid].values - score_only["sound"]).max() < 1e-12
    for mod, cand in MODELS:
        z = np.load(f"{out}/fits/none/{mod}__{cand}.npz", allow_pickle=True)["outer"]
        assert np.abs(_session(mod, z).loc[sid].values - score_only[mod]).max() < 1e-12, mod
    assert np.abs(nsel.combine([score_only[m] for m in ("thermal", "sound", "rgb")], "mean") - score_only["tsr_mean"]).max() < 1e-12
    rows, comp = [], []
    for s in SETS:
        ev = ~Q.loc[sid, f"q_{s}"].values
        rt = {"sound": _sound_sessions(f"{out}/cnn/{s}").loc[sid].values}
        for mod, cand in MODELS:
            rt[mod] = _session(mod, np.load(f"{out}/fits/{s}/{mod}__{cand}.npz", allow_pickle=True)["outer"]).loc[sid].values
        rt["tsr_mean"] = nsel.combine([rt[m] for m in ("thermal", "sound", "rgb")], "mean")
        ye, fe, fr = y[ev], fold[ev], fruit[ev]
        pr = {(t, c): (d[c][ev]).argmax(1) for t, d in (("score_only", score_only), ("retrained", rt)) for c in d}
        for (t, c), p in pr.items():
            fa = [accuracy_score(ye[fe == k], p[fe == k]) for k in range(5)]; ff = [f1_score(ye[fe == k], p[fe == k], average="macro") for k in range(5)]
            rc = recall_score(ye, p, average=None, labels=[0, 1, 2])
            rows.append(dict(session_set=s, n_sessions=int(ev.sum()), training=t, config=c, foldmean_acc=np.mean(fa), foldsd_acc=np.std(fa, ddof=1),
                             foldmean_f1=np.mean(ff), foldsd_f1=np.std(ff, ddof=1), pooled_acc=accuracy_score(ye, p), pooled_f1=f1_score(ye, p, average="macro"),
                             recall_U=rc[0], recall_R=rc[1], recall_O=rc[2], n_errors=int((p != ye).sum())))
        pairs = [("fusion_vs_sound", "score_only", "tsr_mean", "score_only", "sound"), ("fusion_vs_sound", "retrained", "tsr_mean", "retrained", "sound"),
                 ("retrained_vs_score_only", "retrained", "sound", "score_only", "sound"), ("retrained_vs_score_only", "retrained", "tsr_mean", "score_only", "tsr_mean")]
        for kind, tc, cc, trf, cr in pairs:
            pc, prf = pr[(tc, cc)], pr[(trf, cr)]
            df1, pf1 = fruit_perm(ye, prf, pc, fr, macro_f1_rows); dac, pac = fruit_perm(ye, prf, pc, fr, lambda t, M: (M == t).mean(1))
            bt = stats4.fruit_bootstrap(ye, prf, pc, fr, n_boot=N_BOOT, seed=SEED); mc = stats4.clustered_mcnemar(prf == ye, pc == ye, fr)
            ff = lambda p: np.mean([f1_score(ye[fe == k], p[fe == k], average="macro") for k in range(5)])
            fa = lambda p: np.mean([accuracy_score(ye[fe == k], p[fe == k]) for k in range(5)])
            comp.append(dict(session_set=s, n_sessions=int(ev.sum()), comparison=kind, candidate=f"{cc} ({tc})", reference=f"{cr} ({trf})",
                             pooled_d_f1_pp=100 * df1, boot_d_f1_lo_pp=100 * bt["f1_ci"][0], boot_d_f1_hi_pp=100 * bt["f1_ci"][1], p_perm_fruit_f1=pf1,
                             pooled_d_acc_pp=100 * dac, boot_d_acc_lo_pp=100 * bt["acc_ci"][0], boot_d_acc_hi_pp=100 * bt["acc_ci"][1], p_perm_fruit_acc=pac,
                             ref_only=mc["ref_only"], cand_only=mc["cand_only"], p_mcnemar_fruit_clustered=mc["p"],
                             foldmean_d_f1_pp=100 * (ff(pc) - ff(prf)), foldmean_d_acc_pp=100 * (fa(pc) - fa(prf)),
                             sessions_changed=int((pc != prf).sum())))
    R = pd.DataFrame(rows); C = pd.DataFrame(comp)
    R.to_csv(f"{out}/results.csv", index=False); C.to_csv(f"{out}/comparisons.csv", index=False)
    print(R.round(3).to_string(index=False)); print(C.round(3).to_string(index=False))


CTRL = ["ctrl_r1", "ctrl_r2", "ctrl_r3"]


def control_prepare(out, work):
    """Data-quantity control for the strictest set: remove from TRAINING the same number of randomly chosen non-questionable
    sessions with the same harvest batch x common label composition as the 34 questionable sessions (questionable sessions stay in
    training); scoring is on the same 237 sessions. Three draws (seeds 1-3). Also refits thermal SVM-RBF and the RGB probe."""
    Q = pd.read_csv(f"{out}/question_sets.csv"); man = pd.read_csv("data/sound_recording_manifest.csv")
    base = set(open(f"{work}/handoff/exclude_recordings.txt").read().split())
    need = Q[Q.q_excl_harvest].groupby(["Maturity_Stage", "label_common"]).size()
    pool = Q[~Q.q_excl_harvest]; rows = []
    for i, c in enumerate(CTRL, start=1):
        rng = np.random.default_rng(i); pick = []
        for (b, l), n in need.items():
            cand = pool[(pool.Maturity_Stage == b) & (pool.label_common == l)].session_id.values
            pick += list(rng.choice(np.sort(cand), size=n, replace=False))
        rows += [dict(control=c, session_id=x) for x in pick]
        extra = set(man.file[man.session_id.isin(pick)]) - base
        open(f"{out}/cnn_exclude_{c}.txt", "w").write("\n".join(sorted(base | extra)) + "\n")
    R = pd.DataFrame(rows); R.to_csv(f"{out}/control_sessions.csv", index=False)
    for mod, cand in MODELS:
        meta, X = cands.load(mod); fold = meta.fruit_id.map(PART).values.astype(int); sid = meta.session_id.astype(str).values
        for c in CTRL:
            qs = set(R.session_id[R.control == c]); o = np.full((len(meta), 3), np.nan)
            for k in range(5):
                tr = np.flatnonzero((fold != k) & ~np.isin(sid, list(qs))); te = np.flatnonzero(fold == k)
                o[te] = cands.fit_predict(mod, cand, meta, X, tr, te)
            os.makedirs(f"{out}/fits/{c}", exist_ok=True)
            np.savez_compressed(f"{out}/fits/{c}/{mod}__{cand}.npz", files=meta.file.values.astype(str), outer=o)
            print(mod, cand, c, "done", flush=True)
    print(R.groupby("control").size().to_dict())


def control_evaluate(out):
    """Same scored sessions (the 237 of excl_harvest) and folds: score-only, retrained without the questionable sessions, and
    retrained without matched random sessions (three draws)."""
    Q = pd.read_csv(f"{out}/question_sets.csv").set_index("session_id")
    so = pd.read_parquet("results/v17/matched_space_session_oof.parquet"); so = so[so.partition == "p0cpu"]
    base = so[so.estimate == "fixed_sound"].sort_values("session_id").set_index("session_id")
    sid = base.index.values; y = base.y.values.astype(int); fruit = base.fruit_id.values; fold = base.outer_fold.values
    ev = ~Q.loc[sid, "q_excl_harvest"].values; ye, fe, fr = y[ev], fold[ev], fruit[ev]
    P = lambda e: so[so.estimate == e].set_index("session_id").loc[sid, CLS].values
    arms = {"score_only": {"sound": P("fixed_sound"), "tsr_mean": P("fixed_thermal+sound+rgb|mean")}}
    for a in ["excl_harvest"] + CTRL:
        d = {"sound": _sound_sessions(f"{out}/cnn/{a}").loc[sid].values}
        for mod, cand in MODELS:
            d[mod] = _session(mod, np.load(f"{out}/fits/{a}/{mod}__{cand}.npz", allow_pickle=True)["outer"]).loc[sid].values
        d["tsr_mean"] = nsel.combine([d[m] for m in ("thermal", "sound", "rgb")], "mean"); arms[a] = d
    rows = []
    for a, d in arms.items():
        for c in ("sound", "tsr_mean"):
            p = d[c][ev].argmax(1)
            ff = np.mean([f1_score(ye[fe == k], p[fe == k], average="macro") for k in range(5)])
            rows.append(dict(training=a, config=c, n_sessions=int(ev.sum()), foldmean_f1=ff, pooled_f1=f1_score(ye, p, average="macro"),
                             pooled_acc=accuracy_score(ye, p), **dict(zip(["recall_U", "recall_R", "recall_O"], recall_score(ye, p, average=None, labels=[0, 1, 2])))))
        ps, pt = d["sound"][ev].argmax(1), d["tsr_mean"][ev].argmax(1)
        rows[-1]["fusion_minus_sound_pooled_f1_pp"] = 100 * (f1_score(ye, pt, average="macro") - f1_score(ye, ps, average="macro"))
    R = pd.DataFrame(rows); R.to_csv(f"{out}/control_results.csv", index=False); print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare": prepare(sys.argv[2], sys.argv[3])
    elif cmd == "classical": classical(sys.argv[2])
    elif cmd == "evaluate": evaluate(sys.argv[2])
    elif cmd == "control_prepare": control_prepare(sys.argv[2], sys.argv[3])
    elif cmd == "control_evaluate": control_evaluate(sys.argv[2])
