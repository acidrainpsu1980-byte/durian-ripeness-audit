"""v1.4 driver: nested selection, ten partitions and dependence-aware comparisons -> results/v14/.
Inputs: intermediate_v14/fits/<P>/<mod>__<cand>.npz (run_fits.py), intermediate_v14/cnn/cnn_<P>_<kind>_<arg>.npy (cnn_part.py, GPU),
the CPU knock-CNN fits of the primary partition (intermediate_v14/cnn_cpu_p0/cnn_common_<kind>_<arg>.npy), results/session_table.csv.
usage (repo root): DURIAN_INTERMEDIATE=intermediate_common EMB_MOBILENET=<emb_mobilenetv2.npy> python src/nested_selection/evaluate_v14.py"""
import os, sys, itertools
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cands, nsel, stats4

IV = "intermediate_v14"; OUT = os.environ.get("DURIAN_V14_OUT", "results/v14"); os.makedirs(OUT, exist_ok=True)
st = pd.read_csv("results/session_table.csv"); ST = st[st.in_main_analysis.astype(str) == "True"][["session_id", "fruit_id", "y"]].reset_index(drop=True)
ST["fruit_id"] = ST.fruit_id.astype(int); y = ST.y.values.astype(int); fr = ST.fruit_id.values
recf = cands.load("sound")[0].file.values; KN = f"{IV}/features_sound_knocks.parquet"
res, ch = {}, []
pr = nsel.Preds("p0", f"{IV}/fits", f"{IV}/cnn_cpu_p0", "cnn_common_{kind}_{arg}.npy", KN, recf)
o, c, f = nsel.evaluate(pr, ST); res["p0cpu"] = (o, f); ch.append(c.assign(cnn="cpu"))
for p in [f"p{i}" for i in range(10)]:
    pr = nsel.Preds(p, f"{IV}/fits", f"{IV}/cnn", "cnn_{P}_{kind}_{arg}.npy", KN, recf)
    o, c, f = nsel.evaluate(pr, ST); res[p] = (o, f); ch.append(c.assign(cnn="gpu"))
CH = pd.concat(ch); CH.to_csv(f"{OUT}/nested_choices.csv", index=False)
KEY = ["thermal|fixed", "multispectral|fixed", "sound|fixed", "rgb|fixed", "thermal|nested_family", "multispectral|nested_family", "sound|nested_family", "rgb|nested_family",
       "nested_selected_single", "nested_selected_configuration", "thermal+sound+rgb|mean", "thermal+sound+rgb|product", "thermal+multispectral+sound+rgb|mean",
       "thermal+multispectral+sound+rgb|stacking", "thermal+sound+rgb|stacking", "sound+rgb|stacking", "sound+rgb|mean"]
rows = []
for p, (o, fo) in res.items():
    for k in KEY:
        pred = o[k].argmax(1)
        for j in range(5):
            m = fo == j; rows.append(dict(partition=p, config=k, fold=j, n=int(m.sum()), accuracy=accuracy_score(y[m], pred[m]), macro_f1=f1_score(y[m], pred[m], average="macro")))
PF = pd.DataFrame(rows); PF.to_csv(f"{OUT}/partition_per_fold.csv", index=False)
# v2.2 (single manifest): the ten partitions are the primary CPU run of partition 0 (p0cpu) and the GPU refits of partitions 1-9;
# the GPU refit of partition 0 ("p0") is kept only as a hardware-reproducibility check.
TEN = ["p0cpu"] + [f"p{i}" for i in range(1, 10)]
gp = PF[PF.partition.isin(TEN)]
o0 = res["p0cpu"][0]
conf = [("H1", "thermal+sound+rgb|mean", "sound|fixed"), ("H2", "thermal+multispectral+sound+rgb|stacking", "sound|fixed"), ("H3", "nested_selected_configuration", "nested_selected_single")]
expl = [("E1", "sound+rgb|stacking", "sound|fixed"), ("E2", "thermal+sound+rgb|stacking", "sound|fixed"), ("E3", "thermal+multispectral+sound+rgb|mean", "sound|fixed"),
        ("E4", "thermal+multispectral+sound+rgb|stacking", "thermal+sound+rgb|stacking"), ("E5", "thermal|fixed", "sound|fixed"), ("E6", "rgb|fixed", "sound|fixed"), ("E7", "multispectral|fixed", "sound|fixed")]
out = []
for hid, cand, ref in conf + expl:
    pc, pr_ = o0[cand].argmax(1), o0[ref].argmax(1); okc, okr = pc == y, pr_ == y
    mc = stats4.clustered_mcnemar(okr, okc, fr); sf = stats4.fruit_signflip(okr, okc, fr); bs = stats4.fruit_bootstrap(y, pr_, pc, fr)
    a = gp[gp.config == cand].sort_values(["partition", "fold"]); b = gp[gp.config == ref].sort_values(["partition", "fold"])
    na = stats4.nb_corrected_t(a.accuracy.values - b.accuracy.values, 0.25); nf = stats4.nb_corrected_t(a.macro_f1.values - b.macro_f1.values, 0.25)
    p0 = PF[PF.partition == "p0cpu"]; pm = gp.groupby(["config", "partition"]).accuracy.mean().unstack(0)
    out.append(dict(id=hid, role="confirmatory" if hid[0] == "H" else "exploratory", candidate=cand, reference=ref,
                    d_acc_pp_primary=100 * (p0[p0.config == cand].accuracy.mean() - p0[p0.config == ref].accuracy.mean()),
                    d_f1_pp_primary=100 * (p0[p0.config == cand].macro_f1.mean() - p0[p0.config == ref].macro_f1.mean()),
                    ref_only=mc["ref_only"], cand_only=mc["cand_only"], fruits_discordant=mc["fruits_discordant"],
                    p_mcnemar_session_exact=binomtest(mc["cand_only"], mc["ref_only"] + mc["cand_only"]).pvalue if mc["ref_only"] + mc["cand_only"] else 1.0,
                    p_mcnemar_fruit_clustered=mc["p"], p_signflip_fruit=sf["p"], boot_d_acc_lo=bs["acc_ci"][0], boot_d_acc_hi=bs["acc_ci"][1],
                    boot_d_f1_lo=bs["f1_ci"][0], boot_d_f1_hi=bs["f1_ci"][1], nb_d_acc_pp=100 * na["mean"], nb_acc_ci_lo_pp=100 * na["ci"][0], nb_acc_ci_hi_pp=100 * na["ci"][1],
                    nb_p_acc=na["p"], nb_d_f1_pp=100 * nf["mean"], nb_f1_ci_lo_pp=100 * nf["ci"][0], nb_f1_ci_hi_pp=100 * nf["ci"][1], nb_p_f1=nf["p"],
                    partitions_cand_better=int((pm[cand] > pm[ref]).sum())))
T4 = pd.DataFrame(out); cm = T4.role == "confirmatory"
for col in ["p_mcnemar_fruit_clustered", "p_signflip_fruit", "nb_p_acc", "nb_p_f1"]:
    T4.loc[cm, col + "_holm"] = stats4.holm(T4.loc[cm, col].values)
T4.to_csv(f"{OUT}/comparisons_v14.csv", index=False)
pcpu = PF[PF.partition == "p0cpu"].groupby("config")[["accuracy", "macro_f1"]].agg(["mean", "std"])
agg = gp.groupby(["config", "partition"])[["accuracy", "macro_f1"]].mean().groupby("config").agg(["mean", "min", "max"])
TS = pd.DataFrame([dict(config=k, primary_acc_mean=pcpu.loc[k, ("accuracy", "mean")], primary_acc_sd=pcpu.loc[k, ("accuracy", "std")],
                        primary_f1_mean=pcpu.loc[k, ("macro_f1", "mean")], primary_f1_sd=pcpu.loc[k, ("macro_f1", "std")],
                        partitions10_acc_mean=agg.loc[k, ("accuracy", "mean")], partitions10_acc_min=agg.loc[k, ("accuracy", "min")], partitions10_acc_max=agg.loc[k, ("accuracy", "max")],
                        partitions10_f1_mean=agg.loc[k, ("macro_f1", "mean")], partitions10_f1_min=agg.loc[k, ("macro_f1", "min")], partitions10_f1_max=agg.loc[k, ("macro_f1", "max")]) for k in KEY])
TS.to_csv(f"{OUT}/partition_sensitivity.csv", index=False)
fam = CH[CH.level == "family"].reset_index(drop=True).sort_values(["inner_f1", "inner_acc"], ascending=False).drop_duplicates(["partition", "cnn", "modality", "fold"])
fam[(fam.cnn == "cpu") | (fam.partition != "p0")].groupby(["modality", "candidate"]).size().rename("folds_selected").reset_index().to_csv(f"{OUT}/family_selection_frequency.csv", index=False)
print("written", OUT)
