"""Second-round comment 1: separate fixed development pipelines from selection procedures, on a matched candidate space.
Candidate space C26 = 4 single sensors + mean and product fusion of the 11 multi-sensor subsets (the nested configuration search);
C37 = C26 + nested stacking of the 11 subsets (the space of the post hoc 'best configuration' in v1.4).
Estimates per partition (primary partition with the CPU knock CNN, ten partitions with the GPU refits, as in v1.4):
  fixed_*            development pipelines (families fixed in v1.3), no selection at evaluation time
  posthoc_C26/C37    configuration with the best outer-fold score of the primary partition (fold-mean macro-F1, then accuracy),
                     then held fixed on all partitions
  nested_cfg_fixedfam   configuration chosen per outer fold from inner predictions, families fixed (v1.3)
  nested_cfg_nestedfam  configuration and families both chosen from inner predictions (v1.4 'nested_selected_configuration')
  nestedfam_*        pre-specified fusions (mean rule, stacking) built on the nested family choices of each fold
Metric: session accuracy / macro-F1 on the 268 common sessions, mean over the five outer folds of the partition.
usage (repo root): DURIAN_INTERMEDIATE=intermediate_common python src/nested_selection/matched_space.py"""
import os, sys, itertools, warnings
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cands, nsel

IV = "intermediate_v14"; OUT = os.environ.get("DURIAN_V17_OUT", "results/v17"); os.makedirs(OUT, exist_ok=True)
st = pd.read_csv("results/session_table.csv"); ST = st[st.in_main_analysis.astype(str) == "True"][["session_id", "fruit_id", "y"]].reset_index(drop=True)
ST["fruit_id"] = ST.fruit_id.astype(int); y = ST.y.values.astype(int); sid = ST.session_id.astype(str).values
MODS = nsel.MODS; recf = cands.load("sound")[0].file.values; KN = f"{IV}/features_sound_knocks.parquet"
C26 = [((m,), "single") for m in MODS] + [(s, r) for k in range(2, 5) for s in itertools.combinations(MODS, k) for r in ("mean", "product")]
SUBS = [s for k in range(2, 5) for s in itertools.combinations(MODS, k)]
nm = lambda sub, rule: "+".join(sub) + "|" + rule

def foldmean(P, fold):
    pr = P.argmax(1)
    return (float(np.mean([f1_score(y[fold == k], pr[fold == k], average="macro") for k in range(5)])),
            float(np.mean([accuracy_score(y[fold == k], pr[fold == k]) for k in range(5)])))

def run(P, cnn_dir, cnn_fmt):
    pr = nsel.Preds(P, f"{IV}/fits", cnn_dir, cnn_fmt, KN, recf)
    out, ch, fold = nsel.evaluate(pr, ST)                       # v1.4 outputs (fixed singles/fusions/stacking, nested selections)
    O, I = {}, {}
    for mod in MODS:
        for c in cands.CANDIDATES[mod]:
            O[(mod, c)] = pr.session(mod, c, "outer")
            for k in range(5): I[(mod, c, k)] = pr.inner(mod, c, k)
    chosen = {}
    fam = ch[ch.level == "family"]
    for (mod, k), g in fam.groupby(["modality", "fold"], sort=False):   # first candidate wins ties, as in nsel.evaluate
        chosen[(mod, int(k))] = g.sort_values(["inner_f1", "inner_acc"], ascending=False, kind="stable").iloc[0]["candidate"]
    for mod in MODS:                                                    # the reconstructed choices reproduce nsel's nested-family output
        for k in range(5):
            te = fold == k
            assert np.allclose(O[(mod, chosen[(mod, k)])].reindex(sid[te]).values, out[f"{mod}|nested_family"][te]), (P, mod, k)
    res = {}
    # --- fixed development pipelines
    for mod in MODS: res[f"fixed_{mod}"] = out[f"{mod}|fixed"]
    for s, r in C26[4:]: res[f"fixed_{nm(s, r)}"] = out[nm(s, r)]
    for s in SUBS: res[f"fixed_{nm(s, 'stacking')}"] = out[nm(s, "stacking")]
    # --- nested configuration choice, fixed families
    Psel = np.zeros((len(sid), 3)); picks = []
    for k in range(5):
        tr = fold != k; te = fold == k
        ik = {m: I[(m, cands.SELECTED_V13[m], k)].reindex(sid[tr]).values for m in MODS}
        best = None
        for sub, rule in C26:
            Pi = ik[sub[0]] if rule == "single" else nsel.combine([ik[m] for m in sub], rule)
            sc = nsel.score(y[tr], Pi)
            if best is None or sc > best[0]: best = (sc, sub, rule)
        _, sub, rule = best; picks.append(nm(sub, rule))
        Psel[te] = res[f"fixed_{sub[0]}"][te] if rule == "single" else out[nm(sub, rule)][te]
    res["nested_cfg_fixedfam"] = Psel
    res["nested_cfg_nestedfam"] = out["nested_selected_configuration"]; res["nested_single_nestedfam"] = out["nested_selected_single"]
    for mod in MODS: res[f"nestedfam_{mod}"] = out[f"{mod}|nested_family"]
    # --- pre-specified fusions on the nested family choices
    NF = {m: out[f"{m}|nested_family"] for m in MODS}
    for sub in [("thermal", "sound", "rgb"), tuple(MODS)]:
        res[f"nestedfam_{nm(sub, 'mean')}"] = nsel.combine([NF[m] for m in sub], "mean")
        Pn = np.zeros((len(sid), 3))
        for k in range(5):
            tr = fold != k; te = fold == k
            Xtr = np.hstack([I[(m, chosen[(m, k)], k)].reindex(sid[tr]).values for m in sub])
            gs = nsel.fit_stacker(Xtr, y[tr], ST.fruit_id.values[tr]); Pn[te] = gs.predict_proba(np.hstack([NF[m][te] for m in sub]))
        res[f"nestedfam_{nm(sub, 'stacking')}"] = Pn
    return res, fold, picks

PS = ["p0cpu"] + [f"p{i}" for i in range(10)]
R, F, PICK = {}, {}, {}
R["p0cpu"], F["p0cpu"], PICK["p0cpu"] = run("p0", f"{IV}/cnn_cpu_p0", "cnn_common_{kind}_{arg}.npy")
for i in range(10):
    R[f"p{i}"], F[f"p{i}"], PICK[f"p{i}"] = run(f"p{i}", f"{IV}/cnn", "cnn_{P}_{kind}_{arg}.npy"); print("partition", i, flush=True)
# --- post hoc choices on the primary partition (fold-mean macro-F1, then accuracy), held fixed elsewhere
sc0 = {k: foldmean(v, F["p0cpu"]) for k, v in R["p0cpu"].items() if k.startswith("fixed_")}
c26 = [f"fixed_{m}" for m in MODS] + [f"fixed_{nm(s, r)}" for s, r in C26[4:]]
c37 = c26 + [f"fixed_{nm(s, 'stacking')}" for s in SUBS]
best26 = max(c26, key=lambda k: sc0[k]); best37 = max(c37, key=lambda k: sc0[k])
rows = []
for p in PS:
    for k, v in R[p].items():
        f1, acc = foldmean(v, F[p]); rows.append(dict(partition=p, estimate=k, acc=acc, f1=f1))
    for lab, k in [("posthoc_C26", best26), ("posthoc_C37", best37)]:
        f1, acc = foldmean(R[p][k], F[p]); rows.append(dict(partition=p, estimate=lab, acc=acc, f1=f1, config=k))
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/matched_space_detail.csv", index=False)
pd.Series({p: ", ".join(PICK[p]) for p in PS}).to_csv(f"{OUT}/nested_cfg_fixedfam_choices.csv", header=["chosen per fold"])
prim = D[D.partition == "p0cpu"].set_index("estimate")[["acc", "f1"]]
ten = D[D.partition.isin([f"p{i}" for i in range(10)])].groupby("estimate").acc.agg(["mean", "min", "max"])
S = prim.join(ten).round(3); S.to_csv(f"{OUT}/matched_space_summary.csv")
print("post hoc best C26:", best26, "| C37:", best37)
print(S.loc[[i for i in S.index if not i.startswith("fixed_") or i.count("+") == 0 or "thermal+sound+rgb" in i or i.startswith("fixed_thermal+multispectral+sound+rgb") or i == best26 or i == best37]].to_string())

# --- per-fold accuracies and procedure-level comparisons (exploratory): both arms built on nested family choices
import stats4
pf_rows = []
for p in PS:
    for k_, v in R[p].items():
        pr_ = v.argmax(1)
        for j in range(5):
            mk = F[p] == j; pf_rows.append(dict(partition=p, estimate=k_, fold=j, acc=accuracy_score(y[mk], pr_[mk]), f1=f1_score(y[mk], pr_[mk], average="macro")))
PF = pd.DataFrame(pf_rows); PF.to_csv(f"{OUT}/matched_space_per_fold.csv", index=False)
COMP = [("P1", "nestedfam_thermal+sound+rgb|mean", "nestedfam_sound"), ("P2", "nestedfam_thermal+multispectral+sound+rgb|stacking", "nestedfam_sound"),
        ("P3", "nested_cfg_nestedfam", "nested_single_nestedfam"), ("P4", "nested_cfg_fixedfam", "fixed_sound")]
crow = []
g10 = PF[PF.partition != "p0cpu"]
for cid, cand, ref in COMP:
    okc = R["p0cpu"][cand].argmax(1) == y; okr = R["p0cpu"][ref].argmax(1) == y
    mc = stats4.clustered_mcnemar(okr, okc, ST.fruit_id.values)
    a = g10[g10.estimate == cand].sort_values(["partition", "fold"]); b = g10[g10.estimate == ref].sort_values(["partition", "fold"])
    nb = stats4.nb_corrected_t(a.acc.values - b.acc.values, 0.25)
    pm = g10.groupby(["estimate", "partition"]).acc.mean().unstack(0)
    crow.append(dict(id=cid, candidate=cand, reference=ref, d_acc_pp_primary=round(100 * (okc.mean() - okr.mean()), 1), ref_only=mc["ref_only"], cand_only=mc["cand_only"],
                     p_mcnemar_fruit_clustered=round(mc["p"], 3), nb_d_acc_pp=round(100 * nb["mean"], 1), nb_ci_lo=round(100 * nb["ci"][0], 1), nb_ci_hi=round(100 * nb["ci"][1], 1),
                     nb_p=round(nb["p"], 3), partitions_cand_better=int((pm[cand] > pm[ref]).sum())))
C = pd.DataFrame(crow); C.to_csv(f"{OUT}/procedure_level_comparisons.csv", index=False); print(C.to_string())
