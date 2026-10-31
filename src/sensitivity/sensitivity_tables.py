"""v1.8: the two sensitivity analyses of the paper as a script (previously computed in interactive cells during the v1.3 revision).

usage (repo root):  python src/sensitivity/sensitivity_tables.py [RESULTS_DIR]      (default results; run_all.sh uses results_repro)
reads:  RESULTS_DIR/session_table.csv (fusion stage), intermediate_common/oof/oof_sound__cnn__fold_fruit.csv,
        intermediate_common/nested/nested_stacking_probs.npz + nested_session_order.csv, data/session_labels_common.csv,
        data/sound_session_evidence.csv, data/zenodo/durian_characteristics_cleaned.csv (fruit log, from the Zenodo record)
writes: RESULTS_DIR/label_sets.csv                  the 268 evaluation sessions with the label-audit flags (unanimous file label,
                                                     log-inconsistent, ripening signs at harvest = Unripe session with harvest
                                                     pliability > 1 or aroma > 1)
        RESULTS_DIR/sound_matching_sensitivity.csv  Table 10 / Table S9: pooled accuracy / macro-F1 by sound-evidence stratum
        RESULTS_DIR/label_policy_sensitivity.csv    Table 11 / Table S11: principal comparisons on four session sets
        RESULTS_DIR/nested_vs_global_stacking.csv   Table S6: summary of intermediate_common/nested/nested_vs_global_stacking.csv
                                                     (deltas from the rounded means, as published)
The v1.3 nested-stacking probabilities in intermediate_common/nested/ (fusion-stage input) agree with the v1.4 recomputation in
nested_selection/nsel.py on the primary partition to within 0.0065 in probability, with the same predicted class for every
session in all 11 subsets (checked in REPRODUCE.md, section 2).
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import binomtest, ttest_rel
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(REPO)
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
CL = ["Unripe", "Ripe", "Overripe"]
ST = pd.read_csv(f"{RES}/session_table.csv", low_memory=False)
T = ST[ST.n_modalities == 4].reset_index(drop=True)
SL = pd.read_csv("data/session_labels_common.csv").set_index("session_id")
log = pd.read_csv("data/zenodo/durian_characteristics_cleaned.csv"); log.insert(0, "fruit_id", np.arange(1, len(log) + 1))   # rows are fruits 1..189
log = log.set_index("fruit_id")

# ---------------------------------------------------------------- label sets
# label_capture / unripe_lab: the v1.2 majority file-name label (as in the original cell, which used the pre-common-label session
# table); the label sets themselves are defined by unanimous / log_inconsistent / harvest_sign and the common label.
L = T[["session_id", "fruit_id", "Maturity_Stage"]].copy(); L["fruit_id"] = L.fruit_id.astype(int)
L["label_capture"] = L.session_id.map(SL.label_v12_majority)
L["unripe_lab"] = L.label_capture == "Unripe"
L["days_from_harvest"] = L.session_id.map(SL.days_from_recorded_harvest).astype(int)
L = L.join(SL[["label_common", "unanimous", "log_inconsistent", "label_rule"]], on="session_id")
L = L.join(log[["Pliability_Harvest", "Aroma_Harvest", "VQR_Harvest", "Harvest_Rind_Color"]], on="fruit_id")
L["harvest_sign"] = L.unripe_lab & ((L.Pliability_Harvest > 1) | (L.Aroma_Harvest > 1))
L.to_csv(f"{RES}/label_sets.csv", index=False)

# ---------------------------------------------------------------- session-level probabilities on the 268 sessions
mp = lambda m: T[[f"p_{c}__{m}" for c in CL]].values
NZ = np.load("intermediate_common/nested/nested_stacking_probs.npz")
od = pd.read_csv("intermediate_common/nested/nested_session_order.csv").session_id
nz = lambda k: pd.DataFrame(NZ[k], index=od).reindex(T.session_id).values
y = T.y.values.astype(int); fold = T.fold_fruit.values
cnn = pd.read_csv("intermediate_common/oof/oof_sound__cnn__fold_fruit.csv")
ip = cnn[cnn.device.str.lower().str.contains("iphone")].groupby("session_id")[[f"p_{c}" for c in CL]].mean().reindex(T.session_id)
assert ip.notna().all().all()
P = {m: mp(m) for m in ["thermal", "multispectral", "sound", "rgb"]}; P["sound_ip"] = ip.values
mean = lambda ms: np.mean([P[m] for m in ms], 0)

# ---------------------------------------------------------------- Table 10 / S9: sound session-matching strata
ev = pd.read_csv("data/sound_session_evidence.csv").set_index("session_id").loc[T.session_id]
cfg = {"Sound": P["sound"], "Sound (iPhone only)": P["sound_ip"], "Thermal": P["thermal"], "RGB": P["rgb"], "Multispectral": P["multispectral"],
       "T+S+R mean": mean(["thermal", "sound", "rgb"]), "T+S(iPhone)+R mean": mean(["thermal", "sound_ip", "rgb"]),
       "T+M+S+R mean": mean(["thermal", "multispectral", "sound", "rgb"]),
       "T+M+S+R nested stacking": T[[f"p_{c}__fusion[thermal_multispectral_sound_rgb|stacking]" for c in CL]].values}
strata = [("All sessions", np.ones(len(T), bool)), ("Verified same day", ev.session_evidence.values == "verified_same_day"),
          ("Verified, next-day harvest recording", ev.session_evidence.values == "verified_same_stage_next_day"),
          ("Android recording removed (iPhone only)", ev.session_evidence.values == "android_from_other_session")]
maj = lambda m: np.bincount(y[m]).max() / m.sum()
rows = []
for k, p in cfg.items():
    p = p.argmax(1)
    for n, mk in strata:
        rows.append(dict(configuration=k, stratum=n, n_sessions=int(mk.sum()), majority_rate=round(maj(mk), 4),
                         accuracy=round(accuracy_score(y[mk], p[mk]), 4),
                         macro_f1=(np.nan if len(np.unique(y[mk])) == 1 else round(f1_score(y[mk], p[mk], average="macro"), 4))))
pd.DataFrame(rows).to_csv(f"{RES}/sound_matching_sensitivity.csv", index=False)

# ---------------------------------------------------------------- Table 11 / S11: label-policy sensitivity
assert (T.label_capture.values == L.label_common.values).all()   # session table carries the common label
CF = {"Sound": P["sound"], "Thermal": P["thermal"], "RGB": P["rgb"], "Multispectral": P["multispectral"],
      "T+S+R mean": (P["thermal"] + P["sound"] + P["rgb"]) / 3, "T+M+S+R mean": sum(P[m] for m in ["thermal", "multispectral", "sound", "rgb"]) / 4,
      "T+M+S+R nested stacking": nz("thermal__multispectral__sound__rgb"), "S+R nested stacking": nz("sound__rgb"),
      "T+S+R nested stacking": nz("thermal__sound__rgb")}
u = L.unanimous.values.astype(bool); li = L.log_inconsistent.values.astype(bool); hs = L.harvest_sign.values.astype(bool)
sets = {"All 268 (common label)": np.ones(len(T), bool), "Excl. 9 disputed (259)": u,
        "Excl. disputed + 7 log-inconsistent (252)": u & ~li, "Also excl. 15 harvest sessions with ripening signs (237)": u & ~li & ~hs}
rows = []
for sn, mk in sets.items():
    ps = CF["Sound"].argmax(1)
    for cn, P_ in CF.items():
        pc = P_.argmax(1)
        a = [accuracy_score(y[mk & (fold == k)], pc[mk & (fold == k)]) for k in range(5)]
        f = [f1_score(y[mk & (fold == k)], pc[mk & (fold == k)], average="macro") for k in range(5)]
        aS = [accuracy_score(y[mk & (fold == k)], ps[mk & (fold == k)]) for k in range(5)]
        fS = [f1_score(y[mk & (fold == k)], ps[mk & (fold == k)], average="macro") for k in range(5)]
        b = int(((ps == y) & (pc != y) & mk).sum()); c = int(((ps != y) & (pc == y) & mk).sum())
        rows.append(dict(set=sn, n=int(mk.sum()), config=cn, acc_mean=np.mean(a), acc_sd=np.std(a, ddof=1), f1_mean=np.mean(f), f1_sd=np.std(f, ddof=1),
                         pooled_acc=accuracy_score(y[mk], pc[mk]), errors=int(((pc != y) & mk).sum()),
                         d_acc_pp=100 * (np.mean(a) - np.mean(aS)), d_f1_pp=100 * (np.mean(f) - np.mean(fS)),
                         t_p_acc=ttest_rel(a, aS).pvalue if cn != "Sound" else np.nan, t_p_f1=ttest_rel(f, fS).pvalue if cn != "Sound" else np.nan,
                         sound_only=b, cfg_only=c, mcnemar_p=(binomtest(min(b, c), b + c).pvalue if b + c else 1.0) if cn != "Sound" else np.nan))
pd.DataFrame(rows).to_csv(f"{RES}/label_policy_sensitivity.csv", index=False)
lf = pd.read_csv("intermediate_common/nested/nested_vs_global_stacking.csv")
w = lf.pivot_table(index=["task", "subset"], columns="scheme", values=["accuracy_mean", "macro_f1_mean"]).reset_index()
w.columns = ["task", "subset"] + [f"{a}_{b}" for a, b in w.columns[2:]]
w = w[["task", "subset", "accuracy_mean_global_oof", "accuracy_mean_nested", "macro_f1_mean_global_oof", "macro_f1_mean_nested"]].sort_values(["task", "subset"]).reset_index(drop=True)
for c in w.columns[2:]: w[c] = w[c].round(3)
w["delta_accuracy_pp"] = (100 * (w.accuracy_mean_nested - w.accuracy_mean_global_oof)).round(1)
w["delta_macro_f1_pp"] = (100 * (w.macro_f1_mean_nested - w.macro_f1_mean_global_oof)).round(1)
w.to_csv(f"{RES}/nested_vs_global_stacking.csv", index=False)
print("label_sets", L.shape, "| sound strata", len(cfg) * len(strata), "| label-policy rows", len(rows))
