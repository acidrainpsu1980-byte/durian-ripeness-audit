"""v1.8: Table 1 (dataset and confound structure) from shipped files.
usage (repo root):  python src/audit/table1.py            writes audit/table1_dataset_structure.csv and checks it against the paper
Columns: fruits by endpoint (164 analysis fruits), sessions by common label (268 evaluation sessions), capture dates,
RGB images per camera, thermal export quantization (1st-99th percentile of distinct temperature values per frame, before and
after 13 Sep 2025), mean ambient temperature of the thermal frames."""
import os
import numpy as np, pandas as pd
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(REPO)
AOUT = os.environ.get("DURIAN_AUDIT_OUT", "audit")
F = pd.read_csv(f"{AOUT}/fruits.csv"); S = pd.read_csv(f"{AOUT}/sessions.csv")
fa = F[F.status == "analysis_set"]; S["batch"] = S.fruit_id.map(F.set_index("fruit_id").Maturity_Stage); e = S[S.in_evaluation_268]
th = pd.read_parquet("intermediate_common/features/features_thermal.parquet"); th["d"] = th.date_captured.astype(str)
rgb = pd.read_csv("intermediate_common/rgb/rgb512_manifest.csv")
q = lambda s: f"{int(round(s.quantile(0.01)))}–{int(round(s.quantile(0.99)))}"
rows = []
for b in ["Immature", "Mature", "Overmature", "Total"]:
    f_ = fa if b == "Total" else fa[fa.Maturity_Stage == b]; s_ = e if b == "Total" else e[e.batch == b]
    t_ = th if b == "Total" else th[th.Maturity_Stage == b]; r_ = rgb if b == "Total" else rgb[rgb.Maturity_Stage == b]
    fc = f_.actual.value_counts(); sc = s_.label_common.value_counts()
    rows.append(dict(batch=b, fruits=len(f_), fruits_U_R_O="/".join(str(int(fc.get(k, 0))) for k in ["Unripe", "Ripe", "Overripe"]),
                     sessions=len(s_), sessions_U_R_O="/".join(str(int(sc.get(k, 0))) for k in ["Unripe", "Ripe", "Overripe"]),
                     first_capture=s_.date.min(), last_capture=s_.date.max(), n_capture_dates=s_.date.nunique(),
                     rgb_images=len(r_), rgb_R50=int((r_.camera == "Canon EOS R50").sum()), rgb_M50II=int((r_.camera == "Canon EOS M50m2").sum()),
                     thermal_quant_p1_p99_to_13Sep=q(t_[t_.d <= "2025-09-13"].n_unique_vals) if (t_.d <= "2025-09-13").any() else "",
                     thermal_quant_p1_p99_after_13Sep=q(t_[t_.d > "2025-09-13"].n_unique_vals) if (t_.d > "2025-09-13").any() else "",
                     ambient_mean_c=round(float(t_.ambient_c.mean()), 1)))
T = pd.DataFrame(rows); T.to_csv(f"{AOUT}/table1_dataset_structure.csv", index=False); print(T.to_string())
EXP = {"Immature": ("59", "19/21/19", "93", "59/23/11", 5, 481, 28.5), "Mature": ("54", "12/21/21", "95", "54/21/20", 9, 0, 30.2),
       "Overmature": ("51", "17/15/19", "80", "46/16/18", 10, 0, 29.9), "Total": ("164", "48/57/59", "268", "159/60/49", 18, 481, None)}
for r in T.itertuples():
    x = EXP[r.batch]
    assert (str(r.fruits), r.fruits_U_R_O, str(r.sessions), r.sessions_U_R_O, r.n_capture_dates, r.rgb_R50) == x[:6], r
    assert x[6] is None or r.ambient_mean_c == x[6], r
assert T.loc[T.batch == "Immature", "thermal_quant_p1_p99_to_13Sep"].item() == "9–33"
assert T.loc[T.batch == "Total", "thermal_quant_p1_p99_after_13Sep"].item() == "40–236" and T.rgb_images.iloc[-1] == 1999
print("Table 1 counts (fruits, sessions, capture dates, RGB cameras, ambient) reproduced; quantization column = 1st-99th percentile "
      "of distinct values per frame (9-33 up to 13 Sep, 40-236 afterwards)")
