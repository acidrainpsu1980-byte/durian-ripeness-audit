"""v2.2 (review 3, comment 5): sensitivity of the ten-partition summaries to the execution platform of partition 0.
Compares results/v19 (partition 0 = CPU knock CNN, the reported manifest) with results/v19/p0_gpu_check (partition 0 = GPU refit;
produced by DURIAN_P0_PLATFORM=gpu DURIAN_V19_OUT=results/v19/p0_gpu_check python src/nested_selection/primary_metric.py).
For every estimate of Table 3 (ten-partition mean macro-F1 and accuracy, range) and every comparison of Table 4 / S22 (corrected
resampled difference, 95% CI, p, Holm p) it reports both values, the difference, and whether the qualitative conclusion is the same
(CI includes zero under both manifests; p < 0.05 under both or neither).
usage (repo root): python src/nested_selection/p0_platform_sensitivity.py -> results/v19/partition0_platform_sensitivity.csv
"""
import os
import pandas as pd
A = os.environ.get("DURIAN_V19_OUT", "results/v19"); B = os.path.join(A, "p0_gpu_check")
rows = []
ea, eb = pd.read_csv(f"{A}/estimates_summary.csv").set_index("estimate"), pd.read_csv(f"{B}/estimates_summary.csv").set_index("estimate")
for e in ea.index:
    for m in ["ten_foldmean_f1_mean", "ten_foldmean_f1_min", "ten_foldmean_f1_max", "ten_foldmean_acc_mean"]:
        rows.append(dict(kind="estimate", item=e, quantity=m, cpu_p0=ea.loc[e, m], gpu_p0=eb.loc[e, m], diff_pp=100 * (eb.loc[e, m] - ea.loc[e, m])))
ca, cb = pd.read_csv(f"{A}/comparisons_macro_f1.csv").set_index("id"), pd.read_csv(f"{B}/comparisons_macro_f1.csv").set_index("id")
for c in ca.index:
    a, b = ca.loc[c], cb.loc[c]
    same = ((a.nb_f1_ci_lo_pp <= 0 <= a.nb_f1_ci_hi_pp) == (b.nb_f1_ci_lo_pp <= 0 <= b.nb_f1_ci_hi_pp)) and ((a.nb_p_f1 < 0.05) == (b.nb_p_f1 < 0.05))
    if pd.notna(a.get("nb_p_f1_holm")): same = same and ((a.nb_p_f1_holm < 0.05) == (b.nb_p_f1_holm < 0.05))
    rows.append(dict(kind="comparison", item=f"{c} {a.label}", quantity="corrected resampled d macro-F1 (pp) [95% CI]; p",
                     cpu_p0=f"{a.nb_d_f1_pp:+.1f} [{a.nb_f1_ci_lo_pp:.1f}, {a.nb_f1_ci_hi_pp:.1f}]; {a.nb_p_f1:.3f}",
                     gpu_p0=f"{b.nb_d_f1_pp:+.1f} [{b.nb_f1_ci_lo_pp:.1f}, {b.nb_f1_ci_hi_pp:.1f}]; {b.nb_p_f1:.3f}",
                     diff_pp=b.nb_d_f1_pp - a.nb_d_f1_pp, same_conclusion=bool(same)))
D = pd.DataFrame(rows); D.to_csv(f"{A}/partition0_platform_sensitivity.csv", index=False)
E = D[D.kind == "estimate"]; C = D[D.kind == "comparison"]
print("estimates: max |diff| macro-F1 mean (pp)", round(E[E.quantity == "ten_foldmean_f1_mean"].diff_pp.abs().max(), 2),
      "| comparisons: max |diff| (pp)", round(C.diff_pp.abs().max(), 2), "| same conclusion", int(C.same_conclusion.sum()), "of", len(C))
