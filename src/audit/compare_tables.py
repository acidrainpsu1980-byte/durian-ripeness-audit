"""v1.8: compare regenerated tables with the shipped ones (used by run_all.sh).

usage: python src/audit/compare_tables.py SHIPPED_DIR REGENERATED_DIR [glob ...]      (default glob: *.csv)
For each shipped file matching the globs: same shape and column names, identical non-numeric cells, and max |diff| over numeric
cells (NaN must match NaN).  Rows are compared in file order unless the table has no numeric column order guarantee, in which
case both are sorted by all non-numeric columns first.  Exit status 1 if any table differs by more than 1e-9 or is missing.
Parquet files are compared the same way.
"""
import sys, os, glob
import numpy as np, pandas as pd
ship, regen = sys.argv[1], sys.argv[2]; pats = sys.argv[3:] or ["*.csv"]
read = lambda f: pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f, low_memory=False)
bad = 0; n = 0
for pat in pats:
    for f in sorted(glob.glob(os.path.join(ship, pat))):
        g = os.path.join(regen, os.path.relpath(f, ship)); n += 1
        if not os.path.exists(g):
            print(f"MISSING  {os.path.relpath(f, ship)}"); bad += 1; continue
        a, b = read(f), read(g)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            print(f"SHAPE    {os.path.relpath(f, ship)} {a.shape} vs {b.shape}"); bad += 1; continue
        num = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
        obj = [c for c in a.columns if c not in num]
        if obj and not a[obj].astype(str).equals(b[obj].astype(str)):
            a = a.sort_values(obj, kind="mergesort").reset_index(drop=True); b = b.sort_values(obj, kind="mergesort").reset_index(drop=True)
        so = bool(a[obj].astype(str).equals(b[obj].astype(str))) if obj else True
        if num:
            x = a[num].astype(float).values; y = b[num].astype(float).values
            nan_ok = bool((np.isnan(x) == np.isnan(y)).all()); d = float(np.nanmax(np.abs(x - y))) if np.isfinite(x).any() else 0.0
        else:
            nan_ok, d = True, 0.0
        ok = so and nan_ok and d <= 1e-9; bad += (not ok)
        print(f"{'ok' if ok else 'DIFF':8s} {os.path.relpath(f, ship):60s} rows={len(a):6d} max|diff|={d:.3g}{'' if so else ' text-cells-differ'}{'' if nan_ok else ' nan-pattern-differs'}")
print(f"{n - bad}/{n} tables identical"); sys.exit(1 if bad else 0)
