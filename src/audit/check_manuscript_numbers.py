"""v1.8: number provenance of the principal tables.
usage (repo root):  python src/audit/check_manuscript_numbers.py MANUSCRIPT.md [RESULTS_DIR]
For Tables 3-12 of the manuscript (Markdown pipe tables), every number printed with three decimals must appear, formatted the
same way ('{:.3f}', or as percentage points '{:.1f}' of 100 x value), in the result files mapped to that table below.
Writes audit/manuscript_number_check.csv (one row per table: numbers checked, numbers traced, untraced values) and exits 1 if
any number cannot be traced.  Table 1 is checked separately by src/audit/table1.py; Tables 2 and 13 contain no computed results.
"""
import os, re, sys, glob
import numpy as np, pandas as pd
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(REPO)
MS = sys.argv[1]; RES = sys.argv[2] if len(sys.argv) > 2 else "results"
r = lambda *fs: [os.path.join(RES, f) for f in fs]
MAP = {3: r("v14/partition_sensitivity.csv", "v17/matched_space_summary.csv", "v17/matched_space_detail.csv"),
       4: r("v14/comparisons_v14.csv"),
       5: r("single_modality_common_denominator.csv", "fusion_results.csv", "leakage_table.csv"),
       6: r("fusion_results.csv", "stats_tests.csv"),
       7: r("leakage_table.csv"),
       8: r("lobo_table.csv", "fusion_results.csv", "leakage_table.csv"),
       9: r("two_class_results.csv", "fusion_results.csv"),
       10: r("sound_matching_sensitivity.csv", "fusion_results.csv", "label_policy_sensitivity.csv"),
       11: r("label_policy_sensitivity.csv"),
       12: sorted(glob.glob(os.path.join(RES, "v15", "*.csv"))) + r("v14/partition_sensitivity.csv", "v14/comparisons_v14.csv",
                                                                    "v17/matched_space_summary.csv", "lobo_table.csv")}
def strings(files):
    V = set()
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        for c in d.columns:
            x = pd.to_numeric(d[c], errors="coerce").dropna().values.astype(float)
            V |= {f"{v:.3f}" for v in x} | {f"{100 * v:.1f}" for v in x} | {f"{-100 * v:.1f}" for v in x}
            for s in d[c].values:
                if isinstance(s, str):
                    V |= {f"{float(t):.3f}" for t in re.findall(r"-?\d+\.\d+", s)}
    return V
lines = open(MS, encoding="utf-8").read().split("\n")
def table(n):
    i = next(k for k, l in enumerate(lines) if l.startswith(f"**Table {n}.**"))
    j = i + 1
    while not lines[j].startswith("|"): j += 1
    k = j
    while k < len(lines) and lines[k].startswith("|"): k += 1
    return "\n".join(lines[j:k]).replace("−", "-")
rows = []
for n, files in MAP.items():
    V = strings(files); tab = table(n)
    nums = re.findall(r"(?<![\d.])-?\d+\.\d{3}(?!\d)", tab)
    miss = [x for x in nums if x not in V and x.lstrip("-") not in V]
    rows.append(dict(table=f"Table {n}", sources="; ".join(os.path.relpath(f) for f in files), numbers_checked=len(nums),
                     numbers_traced=len(nums) - len(miss), untraced=" ".join(miss)))
out = pd.DataFrame(rows)
os.makedirs(os.environ.get("DURIAN_AUDIT_OUT", "audit"), exist_ok=True)
out.to_csv(os.path.join(os.environ.get("DURIAN_AUDIT_OUT", "audit"), "manuscript_number_check.csv"), index=False)
print(out[["table", "numbers_checked", "numbers_traced", "untraced"]].to_string(index=False))
sys.exit(1 if (out.numbers_checked != out.numbers_traced).any() else 0)
