"""v2.2 (review 3, comment 1): what "sound was the best single sensor in all 50 outer folds" counts.
For each of the 10 partitions x 5 outer folds (partition 0 = CPU knock CNN, partitions 1-9 = GPU refits, as in Table 3):
  (a) the single sensor chosen by the inner selection (results/v14/nested_choices.csv, level 'selected_single');
  (b) the single sensor with the highest outer-fold session macro-F1 among the four selection procedures ('|nested_family') and
      among the four development pipelines ('|fixed') (results/v14/partition_per_fold.csv).
The 50 folds reuse the same 164 fruits with overlapping training sets; they are not independent evaluations.
usage (repo root): python src/nested_selection/fold_ranking.py -> results/v19/single_sensor_fold_ranking.csv (+ _summary.csv)
"""
import os
import pandas as pd
RES = os.environ.get("DURIAN_RESULTS", "results"); OUT = os.environ.get("DURIAN_V19_OUT", os.path.join(RES, "v19")); os.makedirs(OUT, exist_ok=True)
MODS = ["thermal", "multispectral", "sound", "rgb"]
NC = pd.read_csv(os.path.join("results", "v14", "nested_choices.csv"))
sel = NC[(NC.level == "selected_single") & ~((NC.partition == "p0") & (NC.cnn == "gpu"))].copy()
sel["partition"] = sel.partition.replace({"p0": "p0cpu"})
sel = sel.rename(columns={"candidate": "inner_selected_single"})[["partition", "fold", "inner_selected_single"]]
PF = pd.read_csv(os.path.join("results", "v14", "partition_per_fold.csv"))
PF = PF[PF.partition.isin(["p0cpu"] + [f"p{i}" for i in range(1, 10)])]
rows = sel.set_index(["partition", "fold"])
for kind, suffix in [("procedure", "nested_family"), ("development", "fixed")]:
    pv = PF[PF.config.isin([f"{m}|{suffix}" for m in MODS])].pivot_table(index=["partition", "fold"], columns="config", values="macro_f1")
    pv.columns = [c.split("|")[0] for c in pv.columns]
    rows = rows.join(pv.add_prefix(f"{kind}_f1_"))
    rows[f"{kind}_best_test_fold"] = pv.idxmax(axis=1)
    rows[f"{kind}_sound_minus_best_other_pp"] = 100 * (pv["sound"] - pv.drop(columns="sound").max(axis=1))
rows = rows.reset_index(); assert len(rows) == 50 and rows.inner_selected_single.notna().all()
rows.to_csv(os.path.join(OUT, "single_sensor_fold_ranking.csv"), index=False)
summ = [dict(quantity="inner selection chose sound", folds=int((rows.inner_selected_single == "sound").sum()), of=len(rows))]
for kind in ["procedure", "development"]:
    for m in MODS:
        summ.append(dict(quantity=f"{kind}: highest outer-fold macro-F1 = {m}", folds=int((rows[f"{kind}_best_test_fold"] == m).sum()), of=len(rows)))
S = pd.DataFrame(summ); S.to_csv(os.path.join(OUT, "single_sensor_fold_ranking_summary.csv"), index=False)
print(S.to_string(index=False))
