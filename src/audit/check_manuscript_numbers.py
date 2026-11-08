"""v2.2: number provenance of every results table of the main text and the Supplementary Text.
usage (repo root):  python src/audit/check_manuscript_numbers.py MANUSCRIPT.md SUPPLEMENT.md [RESULTS_DIR]
For each Markdown pipe table mapped below, every number printed with the listed numbers of decimals must appear, formatted the
same way ('{:.3f}', '{:.2f}', '{:.1f}', or as percentage points '{:.1f}' of 100 x value), in the result files mapped to that
table. Table D5 holds integers (endpoint sessions, harvests) and is checked against the integer values of its CSV.
Writes audit/manuscript_number_check.csv (one row per table: numbers checked, numbers traced, untraced values) and exits 1 if
any number cannot be traced. The check is set membership per table (a printed value must exist in the mapped files), not a
cell-by-cell comparison; it catches transcription errors unless the wrong value happens to occur elsewhere in the same files. Table 1 is checked separately by src/audit/table1.py; Table 2 contains no computed results.
Table map (v2.2 manuscript):
  main 3 estimates_summary | 4 comparisons | 5 round scores and contrasts | 6 per-batch CIs (round 6) | 7 post hoc ranking
  supp A1 sound matching | B1 single sensor | B2 fusion | B3 leakage | B4 LOBO | C1 rounds 1-3 | C2 rounds 4-6 | D1 two-class |
  D2 label policy | D3 ablations | D5 harvest-variance scenarios
"""
import os, re, sys, glob
import numpy as np, pandas as pd
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); os.chdir(REPO)
MS, SUPP = sys.argv[1], sys.argv[2]; RES = sys.argv[3] if len(sys.argv) > 3 else "results"
r = lambda *fs: [os.path.join(RES, f) for f in fs]
g = lambda *pats: sorted(set(sum((glob.glob(os.path.join(RES, p)) for p in pats), [])))
ROUNDS = g("v21/*.csv", "round2/*.csv", "round3/*.csv", "round4/*.csv", "round5/*.csv", "round6/*.csv")
MAP = [  # (document, table id, source files, decimals checked: a regex range, e.g. "1,3" = one to three)
    ("main", "3", r("v19/estimates_summary.csv"), "1,3"),
    ("main", "4", r("v19/comparisons_macro_f1.csv"), "1,3"),
    ("main", "5", ROUNDS, "1,3"),
    ("main", "6", r("round6/per_batch_ci.csv", "round6/primary_contrast.csv"), "1,3"),
    ("main", "7", g("round5/posthoc_*.csv", "round5/scores.csv"), "2,3"),
    ("supp", "A1", r("sound_matching_sensitivity.csv", "fusion_results.csv", "label_policy_sensitivity.csv"), "3"),
    ("supp", "B1", r("single_modality_common_denominator.csv", "fusion_results.csv", "leakage_table.csv", "common_label/leakage_lgbm_common.csv"), "3"),
    ("supp", "B2", r("fusion_results.csv", "stats_tests.csv"), "3"),
    ("supp", "B3", r("leakage_table.csv", "common_label/leakage_table_v22.csv"), "1,3"),
    ("supp", "B4", r("lobo_table.csv", "fusion_results.csv", "leakage_table.csv"), "3"),
    ("supp", "C1", r("v21/scores.csv", "round2/scores.csv", "round3/scores.csv"), "3"),
    ("supp", "C2", g("round4/*.csv", "round5/*.csv", "round6/*.csv"), "3"),
    ("supp", "D1", r("two_class_results.csv", "fusion_results.csv"), "3"),
    ("supp", "D2", r("label_policy_sensitivity.csv", "v20/results.csv", "v20/comparisons.csv", "v20/distribution_evaluation_sessions.csv"), "3"),
    ("supp", "D3", g("v15/*.csv") + r("v14/partition_sensitivity.csv", "v14/comparisons_v14.csv", "v17/matched_space_summary.csv", "lobo_table.csv"), "3"),
    ("supp", "D5", ["docs/protocol/harvest_variance_scenarios.csv"], "int"),
]
def strings(files):
    V = set()
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        for c in d.columns:
            x = pd.to_numeric(d[c], errors="coerce").dropna().values.astype(float)
            x = x[np.isfinite(x)]
            V |= {f"{v:.3f}" for v in x} | {f"{100 * v:.1f}" for v in x} | {f"{-100 * v:.1f}" for v in x} | {f"{v:.2f}" for v in x} | {f"{v:.1f}" for v in x}
            V |= {f"{int(round(v))}" for v in x if abs(v - round(v)) < 1e-9}
            for s in d[c].values:
                if isinstance(s, str):
                    V |= {f"{float(t):.3f}" for t in re.findall(r"-?\d+\.\d+", s)}
    return V
DOC = {k: open(p, encoding="utf-8").read().split("\n") for k, p in (("main", MS), ("supp", SUPP))}
def table(doc, n):
    lines = DOC[doc]; i = [k for k, l in enumerate(lines) if l.startswith(f"**Table {n}.**")]
    assert len(i) == 1, (doc, n, len(i)); j = i[0] + 1
    while not lines[j].startswith("|"): j += 1
    k = j + 2                                                                  # skip header and separator rows
    while k < len(lines) and lines[k].startswith("|"): k += 1
    return "\n".join(lines[j + 2:k]).replace("−", "-")
rows = []
for doc, n, files, dec in MAP:
    missing_src = [f for f in files if not os.path.exists(f)]; assert not missing_src and files, (n, missing_src)
    V = strings(files); tab = table(doc, n)
    if dec == "int":
        tab = re.sub(r"(?<=\d),(?=\d{3})", "", tab); nums = re.findall(r"(?<![\d.])\d+(?![\d.])", tab)
    else:
        nums = re.findall(r"(?<![\d.])-?\d+\.\d{" + dec + r"}(?!\d)", tab)
    miss = [x for x in nums if x not in V and x.lstrip("-") not in V]
    rows.append(dict(table=f"{'Table' if doc == 'main' else 'Supplementary Table'} {n}", sources="; ".join(os.path.relpath(f) for f in files),
                     numbers_checked=len(nums), numbers_traced=len(nums) - len(miss), untraced=" ".join(miss)))
out = pd.DataFrame(rows)
os.makedirs(os.environ.get("DURIAN_AUDIT_OUT", "audit"), exist_ok=True)
out.to_csv(os.path.join(os.environ.get("DURIAN_AUDIT_OUT", "audit"), "manuscript_number_check.csv"), index=False)
print(out[["table", "numbers_checked", "numbers_traced", "untraced"]].to_string(index=False))
sys.exit(1 if (out.numbers_checked != out.numbers_traced).any() else 0)
