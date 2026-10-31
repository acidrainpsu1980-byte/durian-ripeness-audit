"""Reviewer comment 8: does the multispectral result depend on the assumed band-to-wavelength mapping?
The montage tiles carry no wavelength tags; the model input assumed tile order = ascending nominal centres 720..980 nm.
Order/spacing-dependent features: slope_per100nm, curvature, slope_rawDN_per100nm (polynomial fits over wavelength) and the
ordered pair features ND_i_j (sign depends on order) and SR_i_j (reciprocal depends on order). All are recomputed from the stored
per-band pseudo-reflectance (refl_*) and raw DN (mean_*) values, so no images are needed.
Variants (selected multispectral model = standardized multinomial LR, C by inner fruit-grouped CV, as in cands.py):
  assumed        - features as published
  no_wavelength  - the three slope/curvature features removed
  alt_centres    - slope/curvature recomputed with the datasheet's alternative centres (720,760,795,830,870,905,940,980)
  perm_k         - 20 random tile orders: every order-dependent feature recomputed as if the tiles were in that order
  plsda_raw8     - PLS-DA on the 8 raw band means (order-free reference)
Metric: session accuracy / macro-F1 on the 268 common sessions, mean over the five outer folds (as Table S13).
usage (repo root): DURIAN_INTERMEDIATE=intermediate_common python src/ablations/ms_wavelength_ablation.py"""
import os, sys, warnings
import numpy as np, pandas as pd
from itertools import combinations
from joblib import parallel_backend
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nested_selection"))
import cands, nsel

R = os.environ["DURIAN_INTERMEDIATE"]; OUT = os.environ.get("DURIAN_V15_OUT", "results/v15"); os.makedirs(OUT, exist_ok=True)
st = pd.read_csv("results/session_table.csv"); ST = st[st.in_main_analysis.astype(str) == "True"][["session_id", "fruit_id", "y"]].reset_index(drop=True)
ST["session_id"] = ST.session_id.astype(str); yv = ST.y.values.astype(int); PS = [f"p{i}" for i in range(10)]
FOLD = {p: ST.fruit_id.astype(int).map(nsel.PARTS[p]).values for p in PS}

# ---- the published feature matrix, with column names (same rule as cands.load("multispectral"))
f = pd.read_parquet(f"{R}/features/features_multispectral.parquet")
o = pd.read_csv(f"{R}/oof/oof_multispectral__logreg__fold_fruit.csv")
meta = set(o.columns) | {"zip_member", "modality", "orientation", "device", "actual", "fold_random", "date_captured"}
excl = {"otsu_thr", "seg_band_idx", "seg_ok", "read_ok", "err", "cleaned_flag"}
cols = [c for c in f.columns if c not in meta and c not in excl and not c.startswith("bgmean_")]
m0, X0 = cands.load("multispectral"); assert np.allclose(f[cols].values.astype(float), X0["full"], equal_nan=True)
BN = ["B1_720", "B2_760", "B3_800", "B4_840", "B5_860", "B6_900", "B7_940", "B8_980"]
WL = np.array([720, 760, 800, 840, 860, 900, 940, 980.0]); WL_ALT = np.array([720, 760, 795, 830, 870, 905, 940, 980.0])
WDEP = ["slope_per100nm", "curvature", "slope_rawDN_per100nm"]
REFL = f[[f"refl_{b}" for b in BN]].values.astype(float); MEAN = f[[f"mean_{b}" for b in BN]].values.astype(float)

def polyfeat(wl, order):
    """slope/curvature with tile order[i] placed at wavelength wl[i]"""
    x = (wl - wl.mean()) / 100.0; Rr = REFL[:, order]; Mm = MEAN[:, order]
    c1 = np.polyfit(x, Rr.T, 1)[0]; c2 = np.polyfit(x, Rr.T, 2)[0]; c1r = np.polyfit(x, Mm.T, 1)[0]
    return c1, c2, c1r

# reproduce the published slope/curvature exactly
s, c, sr = polyfeat(WL, np.arange(8))
ok = f.seg_ok.astype(bool).values
assert np.allclose(s[ok], f.slope_per100nm.values[ok]) and np.allclose(c[ok], f.curvature.values[ok]) and np.allclose(sr[ok], f.slope_rawDN_per100nm.values[ok])

def variant(order=None, wl=WL, drop=()):
    g = f[cols].copy().astype(float)
    if order is not None or wl is not WL:
        order = np.arange(8) if order is None else np.asarray(order)
        s, c, sr = polyfeat(wl, order)
        g["slope_per100nm"], g["curvature"], g["slope_rawDN_per100nm"] = np.where(ok, s, np.nan), np.where(ok, c, np.nan), np.where(ok, sr, np.nan)
        pos = {t: k for k, t in enumerate(order)}            # position of tile t in the assumed sequence
        for i, j in combinations(range(8), 2):              # published pair (i<j) -> as computed under the new order
            a, b = REFL[:, i], REFL[:, j]
            if pos[i] > pos[j]: a, b = b, a
            with np.errstate(divide="ignore", invalid="ignore"):
                g[f"ND_{i+1}_{j+1}"] = np.where(ok, (a - b) / (a + b), np.nan); g[f"SR_{i+1}_{j+1}"] = np.where(ok, a / b, np.nan)
    g = g.drop(columns=list(drop))
    return g.values

def sess_eval(X, p, model="logreg"):
    fold = m0.fruit_id.map(nsel.PARTS[p]).values; y = m0.y.values; g = m0.fruit_id.values; P = np.full((len(X), 3), np.nan)
    for k in range(5):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]; _, est = cands._ms(model)
        with parallel_backend("threading", n_jobs=4):
            est.fit(X[tr], y[tr], groups=g[tr]) if isinstance(est, GridSearchCV) else est.fit(X[tr], y[tr]); P[te] = est.predict_proba(X[te])
    t = pd.DataFrame(P, columns=nsel.CL); t["s"] = m0.session_id.astype(str).values
    S = t.groupby("s")[nsel.CL].mean().reindex(ST.session_id).values; pr = S.argmax(1); fo = FOLD[p]
    return (float(np.mean([accuracy_score(yv[fo == k], pr[fo == k]) for k in range(5)])), float(np.mean([f1_score(yv[fo == k], pr[fo == k], average="macro") for k in range(5)])))

rows = []
V = {"assumed": dict(), "no_wavelength": dict(drop=WDEP), "alt_centres": dict(wl=WL_ALT)}
for name, kw in V.items():
    X = variant(**kw)
    for p in PS:
        a, f1 = sess_eval(X, p); rows.append(dict(variant=name, partition=p, acc=a, f1=f1))
    print(name, "done", flush=True)
X8 = f[[f"mean_{b}" for b in BN]].values.astype(float)
for p in PS:
    a, f1 = sess_eval(X8, p, "plsda_raw8"); rows.append(dict(variant="plsda_raw8", partition=p, acc=a, f1=f1))
rng = np.random.default_rng(42); PERMS = [rng.permutation(8) for _ in range(20)]
for k, order in enumerate(PERMS):
    a, f1 = sess_eval(variant(order=order), "p0"); rows.append(dict(variant=f"perm_{k:02d}", partition="p0", acc=a, f1=f1, order=" ".join(map(str, order + 1))))
print("permutations done", flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{OUT}/ms_wavelength_ablation_detail.csv", index=False)
print(D.groupby("variant")[["acc", "f1"]].agg(["mean", "min", "max"]).round(3).to_string())
