#!/usr/bin/env python
"""Compare a re-run of the fusion stage with the shipped tables.  Usage: verify_reproduction.py <shipped_dir> <repro_dir>
Numeric columns: max |difference| (NaN == NaN).  Text columns: number of differing cells after NaN alignment and after
stripping the artifact-version prefix that the original run carried in lobo_table.source."""
import sys, re, numpy as np, pandas as pd
A, B = sys.argv[1], sys.argv[2]
TABLES = ["fusion_results.csv", "fusion_results_per_fold.csv", "single_modality_common_denominator.csv", "two_class_results.csv",
          "stats_tests.csv", "leakage_table.csv", "lobo_table.csv", "error_analysis.csv", "session_table.csv"]
worst = 0.0; ok = True
for fn in TABLES:
    a = pd.read_csv(f"{A}/{fn}"); b = pd.read_csv(f"{B}/{fn}")
    if set(a.columns) != set(b.columns) or len(a) != len(b):
        print(f"{fn:42s} SHAPE/COLUMN MISMATCH {a.shape} vs {b.shape}"); ok = False; continue
    b = b[a.columns]
    num = a.select_dtypes("number").columns
    d = float((a[num].fillna(-9e9) - b[num].fillna(-9e9)).abs().max().max()) if len(num) else 0.0
    txt = 0
    for c in [c for c in a.columns if c not in num]:
        x = a[c].astype(str).str.replace(r"^v[0-9a-f]{8}_", "", regex=True); yv = b[c].astype(str).str.replace(r"^v[0-9a-f]{8}_", "", regex=True)
        txt += int((~((a[c].isna() & b[c].isna()) | (x == yv))).sum())
    worst = max(worst, d); ok &= (d < 1e-9 and txt == 0)
    print(f"{fn:42s} rows={len(a):4d}  max|diff|={d:.3g}  text-mismatches={txt}")
print("MAX ABS DIFF over all numeric columns:", worst, "| exact:", ok)
sys.exit(0 if ok else 1)
