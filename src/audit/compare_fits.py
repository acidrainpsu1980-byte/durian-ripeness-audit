"""v1.8: compare refitted prediction arrays (.npz from run_fits.py, .npy from the CNN scripts) with the shipped ones.
usage: python src/audit/compare_fits.py SHIPPED_DIR REFIT_DIR [glob]      (default glob: *.npz; exit 1 if max |diff| > 1e-9)"""
import sys, os, glob
import numpy as np
ship, new = sys.argv[1], sys.argv[2]; pat = sys.argv[3] if len(sys.argv) > 3 else "*.npz"
bad = 0; n = 0
for f in sorted(glob.glob(os.path.join(new, pat))):
    g = os.path.join(ship, os.path.basename(f)); n += 1
    if f.endswith(".npz"):
        a, b = np.load(f, allow_pickle=True), np.load(g, allow_pickle=True)
        keys = [k for k in b.files if k != "files"]
        same_files = bool((a["files"] == b["files"]).all())
        d = max(float(np.nanmax(np.abs(a[k] - b[k]))) for k in keys)
        nan_ok = all(np.array_equal(np.isnan(a[k]), np.isnan(b[k])) for k in keys)
    else:
        a, b = np.load(f), np.load(g); keys = ["array"]; same_files = True
        d = float(np.nanmax(np.abs(a - b))); nan_ok = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    ok = same_files and nan_ok and d <= 1e-9; bad += (not ok)
    print(f"{'ok' if ok else 'DIFF':5s} {os.path.basename(f):40s} arrays={len(keys):2d} max|diff|={d:.3g}")
print(f"{n - bad}/{n} prediction files identical"); sys.exit(1 if (bad or n == 0) else 0)
