"""Outer (5) and fold-pair (10) refits of candidate families on a fruit partition.
usage: python run_fits.py OUTDIR P mod:cand [mod:cand ...]      (P = partition column p0..p9 of partitions.csv)
Writes OUTDIR/P/<mod>__<cand>.npz with arrays outer (n x 3) and pair_ab (n x 3, NaN outside folds a, b)."""
import os, sys, time, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cands

out, P = sys.argv[1], sys.argv[2]
part = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "partitions.csv")).set_index("fruit_id")[P]
os.makedirs(f"{out}/{P}", exist_ok=True)
cache = {}
for job in sys.argv[3:]:
    mod, cand = job.split(":")
    fn = f"{out}/{P}/{mod}__{cand}.npz"
    if os.path.exists(fn):
        continue
    if mod not in cache:
        cache[mod] = cands.load(mod)
    meta, X = cache[mod]
    fold = meta.fruit_id.map(part).values.astype(int)
    t = time.time(); res = {}
    o = np.full((len(meta), 3), np.nan)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        o[te] = cands.fit_predict(mod, cand, meta, X, tr, te)
    res["outer"] = o
    for a, b in itertools.combinations(range(5), 2):
        pp = np.full((len(meta), 3), np.nan)
        tr, te = np.where((fold != a) & (fold != b))[0], np.where((fold == a) | (fold == b))[0]
        pp[te] = cands.fit_predict(mod, cand, meta, X, tr, te)
        res[f"pair_{a}{b}"] = pp
    np.savez_compressed(fn + ".tmp.npz", files=meta.file.values.astype(str), **res)
    os.replace(fn + ".tmp.npz", fn)
    print(f"{P} {job}: {time.time()-t:.0f}s", flush=True)
