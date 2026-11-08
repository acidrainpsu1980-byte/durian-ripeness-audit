"""v1.5 driver (reviewer comment 6): targeted ablations -> results/v15/.
  * thermal: original vs 0.5 degC-harmonized features (thermal_harmonize.py), all vs fruit-relative, quantization-only; fruit-grouped
    (ten partitions) and leave-one-batch-out
  * RGB: EfficientNet-B0 probe on original, fruit-mask, stalk-free fruit-body and context-only crops (rgb_mask_embed.py); ten partitions;
    fruit-disjoint cross-camera on the primary partition
  * sound: knock CNN trained on iPhone / Android / iPhone-on-dual-sessions (cnn_dev.py, GPU) vs the dual-phone CNN, scored on the 163
    identical dual-phone sessions; paired fruit-clustered McNemar tests
  * fusion: thermal + sound + RGB mean rule with ablated inputs
Inputs: intermediate_v15/ (thermal_harmonized.parquet, emb_effb0_<variant>.npy, dev/cnn_<P>_<kind>_<k>.npy, knock_meta.npz) plus the
v1.4 intermediates. Metric: session accuracy / macro-F1 averaged over the five outer folds of each partition (as in Table S13); LOBO and
cross-camera results are pooled over their single held-out set. usage (repo root): DURIAN_INTERMEDIATE=intermediate_common python src/ablations/evaluate_v15.py"""
import os, sys, warnings
import numpy as np, pandas as pd
from joblib import parallel_backend
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nested_selection"))
import cands, nsel, stats4

IV, I5 = "intermediate_v14", "intermediate_v15"
OUT = os.environ.get("DURIAN_V15_OUT", "results/v15"); os.makedirs(OUT, exist_ok=True)
R = os.environ["DURIAN_INTERMEDIATE"]
st = pd.read_csv("results/session_table.csv"); ST = st[st.in_main_analysis.astype(str) == "True"][["session_id", "fruit_id", "y"]].reset_index(drop=True)
ST["fruit_id"] = ST.fruit_id.astype(int); ST["session_id"] = ST.session_id.astype(str); yv = ST.y.values.astype(int); fv = ST.fruit_id.values
CL = nsel.CL; PS = [f"p{i}" for i in range(10)]

FOLD = {p: ST.fruit_id.map(nsel.PARTS[p]).values for p in PS}
def pf(P, p="p0", y=None, fold=None):
    """Session accuracy / macro-F1 as the mean over the five outer folds of partition p (the paper's primary metric, as in Table S13)."""
    y = yv if y is None else y; fold = FOLD[p] if fold is None else fold; pr = P.argmax(1)
    a = [accuracy_score(y[fold == k], pr[fold == k]) for k in range(5)]; f = [f1_score(y[fold == k], pr[fold == k], average="macro") for k in range(5)]
    return round(float(np.mean(a)), 3), round(float(np.mean(f)), 3)

# ---------------------------------------------------------------- v1.4 fixed-family session probabilities (CPU p0 and GPU p0-p9)
recf = cands.load("sound")[0].file.values; KN = f"{IV}/features_sound_knocks.parquet"
FIX = {}
pr = nsel.Preds("p0", f"{IV}/fits", f"{IV}/cnn_cpu_p0", "cnn_common_{kind}_{arg}.npy", KN, recf); FIX["p0cpu"] = nsel.evaluate(pr, ST)[0]
for p in PS:
    pr = nsel.Preds(p, f"{IV}/fits", f"{IV}/cnn", "cnn_{P}_{kind}_{arg}.npy", KN, recf); FIX[p] = nsel.evaluate(pr, ST)[0]
print("fixed families loaded", flush=True)

# ---------------------------------------------------------------- thermal
fT = pd.read_parquet(f"{R}/features/features_thermal.parquet")
TH = pd.read_parquet(f"{I5}/thermal_harmonized.parquet").set_index("zip_member").reindex(fT.zip_member).reset_index()
meta = ["modality", "file", "zip_member", "fruit_id", "session_id", "orientation", "date_captured", "label_capture", "y", "actual", "Maturity_Stage",
        "Class", "fold_fruit", "fold_random", "device", "ambient_c", "seg_fallback", "otsu_thr_c", "n_unique_vals"]
fcols = [c for c in fT.columns if c not in meta]
ABS = ["fruit_mean", "fruit_median", "fruit_min", "fruit_max", "fruit_p5", "fruit_p25", "fruit_p75", "fruit_p95", "bg_mean", "bg_median", "bg_p5", "bg_p95",
       "frame_min", "frame_max", "frame_mean"]
rcols = [c for c in fcols if c not in ABS]
assert len(fcols) == 67 and len(rcols) == 52, (len(fcols), len(rcols))
for c in ["otsu_thr_c", "seg_quality", "seg_fallback", "area_px", "eccentricity", "solidity", "circularity"]:
    assert np.allclose(fT[c].astype(float).values, TH[c].astype(float).values, equal_nan=True), c      # identical segmentation
yT = fT.y.values; gT = fT.fruit_id.astype(int).values; sT = fT.session_id.astype(str).values; bT = fT.Maturity_Stage.values
XS = {"orig_all": fT[fcols].values.astype(float), "orig_rel": fT[rcols].values.astype(float), "harm_all": TH[fcols].values.astype(float),
      "harm_rel": TH[rcols].values.astype(float), "harm_quant_only": TH[["n_unique_vals"]].values.astype(float),
      "orig_quant_only": fT[["n_unique_vals"]].values.astype(float)}

def th_oof(model, X, fold):
    P = np.full((len(X), 3), np.nan)
    for k in np.unique(fold):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]; _, est = cands._thermal(model)
        with parallel_backend("threading", n_jobs=4):
            if isinstance(est, GridSearchCV): est.fit(X[tr], yT[tr], groups=gT[tr])
            else: est.fit(X[tr], yT[tr])
            P[te] = est.predict_proba(X[te])
    t = pd.DataFrame(P, columns=CL); t["s"] = sT
    return t.groupby("s")[CL].mean().reindex(ST.session_id).values

THR = {}
for xs in XS:
    for m in (["lgbm"] if "quant" in xs else ["svm_rbf", "lgbm"]):
        for p in PS: THR[(xs, m, p)] = th_oof(m, XS[xs], fT.fruit_id.astype(int).map(nsel.PARTS[p]).values)
        THR[(xs, m, "lobo")] = th_oof(m, XS[xs], bT)
print("thermal done", flush=True)
bS = fT.groupby("session_id").Maturity_Stage.first(); bS.index = bS.index.astype(str); bS = bS.reindex(ST.session_id).values
rows = []
for (xs, m, p) in [k for k in THR if k[2] == "p0"]:
    a = np.array([pf(THR[(xs, m, q)], q) for q in PS])
    r = dict(features=xs, model=m, acc_p0=a[0, 0], f1_p0=a[0, 1], acc_mean10=a[:, 0].mean().round(3), acc_min=a[:, 0].min(), acc_max=a[:, 0].max(),
             f1_mean10=a[:, 1].mean().round(3))
    for b in ["Immature", "Mature", "Overmature"]:
        i = bS == b; prd = THR[(xs, m, "lobo")][i].argmax(1)
        r[b + "_acc"] = round(accuracy_score(yv[i], prd), 3); r[b + "_f1"] = round(f1_score(yv[i], prd, average="macro"), 3)
    rows.append(r)
pd.DataFrame(rows).to_csv(f"{OUT}/thermal_ablation.csv", index=False)

# ---------------------------------------------------------------- RGB
mR, XR = cands.load("rgb"); man = pd.read_csv(f"{R}/rgb/rgb512_manifest.csv"); assert (man.file.values == mR.file.values).all()
R50 = (man.camera.values == "Canon EOS R50"); M50 = ~R50
EMB = {"original": XR["effnet"]}
for v in ["fruit_only", "fruit_body", "context_only"]:
    E = np.load(f"{I5}/emb_effb0_{v}.npy"); EMB[v] = E.mean(1) if E.ndim == 3 else E      # view-averaged (N,1280) as shipped

def rgb_oof(E, fold, trm=None, tem=None):
    P = np.full((len(mR), 3), np.nan)
    for k in range(5):
        tr, te = fold != k, fold == k
        if trm is not None: tr &= trm
        if tem is not None: te &= tem
        P[np.where(te)[0]] = cands.fit_predict("rgb", "efficientnetb0_probe", mR, {"effnet": E}, np.where(tr)[0], np.where(te)[0])
    t = pd.DataFrame(P, columns=CL); t["s"] = mR.session_id.astype(str).values
    return t.dropna().groupby("s")[CL].mean()

def recall(S, y):
    pr = S.argmax(1); return str([round(float(((pr == c) & (y == c)).sum() / max((y == c).sum(), 1)), 3) for c in range(3)])

RGB, rows, crow = {}, [], []
ysS = ST.set_index("session_id").y
f0 = mR.fruit_id.map(nsel.PARTS["p0"]).values
for v, E in EMB.items():
    for i, p in enumerate(PS):
        RGB[(v, p)] = FIX[p]["rgb|fixed"] if (v == "original" and i > 0) else rgb_oof(E, mR.fruit_id.map(nsel.PARTS[p]).values).reindex(ST.session_id).values
    a = np.array([pf(RGB[(v, p)], p) for p in PS])
    rows.append(dict(variant=v, acc_p0=a[0, 0], f1_p0=a[0, 1], acc_mean10=a[:, 0].mean().round(3), acc_min=a[:, 0].min(), acc_max=a[:, 0].max(),
                     f1_mean10=a[:, 1].mean().round(3), recall_URO_p0=recall(RGB[(v, "p0")], yv)))
    for trn, trm in [("R50", R50), ("M50", M50), ("both", None)]:
        for ten, tem in [("R50", R50), ("M50", M50)]:
            S = rgb_oof(E, f0, trm, tem); S = S[S.index.isin(ST.session_id)]; y = ysS.reindex(S.index).values; prd = S.values.argmax(1)
            crow.append(dict(variant=v, train=trn, test=ten, n_sess=len(S), acc=round(accuracy_score(y, prd), 3),
                             f1=round(f1_score(y, prd, average="macro", labels=[0, 1, 2]), 3), recall_URO=recall(S.values, y),
                             majority=round(pd.Series(y).value_counts(normalize=True).max(), 3)))
pd.DataFrame(rows).to_csv(f"{OUT}/rgb_masking.csv", index=False); pd.DataFrame(crow).to_csv(f"{OUT}/rgb_cross_camera.csv", index=False)
mc = []
for a_, b_ in [("original", "fruit_only"), ("original", "fruit_body"), ("original", "context_only"), ("fruit_only", "fruit_body")]:
    r = stats4.clustered_mcnemar(RGB[(a_, "p0")].argmax(1) == yv, RGB[(b_, "p0")].argmax(1) == yv, fv); mc.append(dict(A=a_, B=b_, **r))
pd.DataFrame(mc).to_csv(f"{OUT}/rgb_paired_tests.csv", index=False)
print("rgb done", flush=True)

# ---------------------------------------------------------------- sound devices on identical sessions
msnd, _ = cands.load("sound"); msnd["session_id"] = msnd.session_id.astype(str)
knk = pd.read_parquet(KN)[["file"]]; km = np.load(f"{I5}/knock_meta.npz", allow_pickle=True)
msnd["device"] = pd.Series(km["device"].astype(str), index=knk.file.values).groupby(level=0).first().reindex(msnd.file).values

def rec_probs(kind, P):
    ok = np.full((len(knk), 3), np.nan)
    for k in range(5):
        a = np.load(f"{IV}/cnn/cnn_{P}_outer_{k}.npy") if kind == "outer" else np.load(f"{I5}/dev/cnn_{P}_{kind}_{k}.npy")
        m = ~np.isnan(a[:, 0]); ok[m] = a[m]
    t = pd.DataFrame(ok, columns=CL); t["file"] = knk.file.values
    return t.groupby("file")[CL].mean().reindex(msnd.file.values).values

RP = {(k, p): rec_probs(k, p) for k in ["outer", "iphone", "android", "iphoneM"] for p in PS}
nd = msnd.groupby("session_id").device.nunique(); SB = nd[nd == 2].index.intersection(ysS.index); assert len(SB) == 163
yB = ysS.reindex(SB).values; fB = ST.set_index("session_id").fruit_id.reindex(SB).values

def sess_pred(P, devs):
    m = msnd.device.isin(devs).values; t = pd.DataFrame(P[m], columns=CL); t["s"] = msnd.session_id.values[m]
    return t.groupby("s")[CL].mean().reindex(SB).values.argmax(1)

D = [("dual-trained", "outer", ["iphone", "android"]), ("dual-trained", "outer", ["iphone"]), ("dual-trained", "outer", ["android"]),
     ("iPhone-trained", "iphone", ["iphone"]), ("iPhone-trained", "iphone", ["android"]), ("iPhone-trained", "iphone", ["iphone", "android"]),
     ("iPhoneM-trained", "iphoneM", ["iphone"]), ("iPhoneM-trained", "iphoneM", ["android"]),
     ("Android-trained", "android", ["android"]), ("Android-trained", "android", ["iphone"]), ("Android-trained", "android", ["iphone", "android"])]
PRED, rows = {}, []
for lab, kind, devs in D:
    acc, f1 = [], []
    for i, p in enumerate(PS):
        prd = sess_pred(RP[(kind, p)], devs); PRED[(lab, "+".join(devs), i)] = prd; fo = pd.Series(fB).map(nsel.PARTS[p]).values
        acc.append(np.mean([accuracy_score(yB[fo == k], prd[fo == k]) for k in range(5)])); f1.append(np.mean([f1_score(yB[fo == k], prd[fo == k], average="macro") for k in range(5)]))
    rows.append(dict(train=lab, test_recordings="+".join(devs), sess_acc_p0=round(acc[0], 3), sess_f1_p0=round(f1[0], 3), acc_mean10=round(np.mean(acc), 3),
                     acc_min=round(min(acc), 3), acc_max=round(max(acc), 3), f1_mean10=round(np.mean(f1), 3)))
pd.DataFrame(rows).to_csv(f"{OUT}/device_identical_sessions.csv", index=False)
rows = []
for a_, b_ in [("dual-trained|iphone+android", "dual-trained|iphone"), ("dual-trained|iphone", "dual-trained|android"),
               ("dual-trained|iphone+android", "dual-trained|android"), ("iPhoneM-trained|iphone", "Android-trained|android"),
               ("iPhone-trained|iphone", "iPhone-trained|android"), ("Android-trained|android", "Android-trained|iphone")]:
    A, B = tuple(a_.split("|")), tuple(b_.split("|")); ca, cb = PRED[A + (0,)] == yB, PRED[B + (0,)] == yB
    r = stats4.clustered_mcnemar(ca, cb, fB); d = [100 * ((PRED[A + (i,)] == yB).mean() - (PRED[B + (i,)] == yB).mean()) for i in range(10)]
    rows.append(dict(A=a_, B=b_, d_acc_pp_p0=round(100 * (ca.mean() - cb.mean()), 1), disc_A_only=int((ca & ~cb).sum()), disc_B_only=int((~ca & cb).sum()),
                     p_mcnemar_p0=round(r["p"], 4), d_mean10_pp=round(np.mean(d), 1), d_min10=round(min(d), 1), d_max10=round(max(d), 1)))
pd.DataFrame(rows).to_csv(f"{OUT}/device_paired_tests.csv", index=False)
print("devices done", flush=True)

# ---------------------------------------------------------------- fusion with ablated inputs
def src(i, mod, cpu): return FIX["p0cpu" if (cpu and i == 0) else PS[i]][f"{mod}|fixed"]
def rgbv(v, i, cpu): return src(i, "rgb", cpu) if (v == "original" and i == 0) else RGB[(v, PS[i])]
FUS = [("T+S+R (original)", "orig", "original"), ("T+S+R, fruit-only RGB", "orig", "fruit_only"), ("T+S+R, context-only RGB", "orig", "context_only"),
       ("T+S+R, stalk-removed fruit RGB", "orig", "fruit_body"), ("T+S+R, harmonized fruit-relative thermal", "harm", "original"),
       ("T+S+R, both ablations", "harm", "fruit_only"), ("both ablations (stalk-removed)", "harm", "fruit_body")]
rows = []
for nm, th, rv in FUS:
    def comb(i, cpu):
        return np.mean([src(i, "thermal", cpu) if th == "orig" else THR[("harm_rel", "svm_rbf", PS[i])], src(i, "sound", cpu), rgbv(rv, i, cpu)], 0)
    # v2.2 single manifest: partition 0 of the ten-partition summary is the primary CPU run
    p0 = pf(comb(0, True)); a = np.array([pf(comb(i, i == 0), PS[i]) for i in range(10)])
    rows.append(dict(configuration=nm, acc_p0_cpu=p0[0], f1_p0_cpu=p0[1], acc_mean10=a[:, 0].mean().round(3), acc_min=a[:, 0].min(), acc_max=a[:, 0].max(),
                     f1_mean10=a[:, 1].mean().round(3)))
a = np.array([pf(src(i, "sound", i == 0), PS[i]) for i in range(10)]); p0 = pf(src(0, "sound", True))
rows.append(dict(configuration="Sound alone", acc_p0_cpu=p0[0], f1_p0_cpu=p0[1], acc_mean10=a[:, 0].mean().round(3), acc_min=a[:, 0].min(),
                 acc_max=a[:, 0].max(), f1_mean10=a[:, 1].mean().round(3)))
pd.DataFrame(rows).to_csv(f"{OUT}/fusion_with_ablated_inputs.csv", index=False)
print("done", flush=True)
