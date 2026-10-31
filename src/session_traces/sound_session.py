# sound track - literal cells executed in the interactive analysis session (chronological, successful cells only,
# exploration/figure cells omitted).  Kept for transparency: this is the code that produced the shipped
# intermediate/oof files.  It is NOT a standalone script (it was executed cell by cell in one kernel, with files
# being edited between cells); the consolidated final-state driver is src/sound/sound_pipeline.py.
# Absolute paths were not present in these cells; the fold table was read from the working directory.

# ==========================================================================================
# cell 2: Loading sound fold table and checking decoders  [python]
# ==========================================================================================

import os, sys; sys.path.append(os.getcwd())
import durian_common as dc, pandas as pd, numpy as np, io, tempfile, time, shutil
df = dc.load_folds("durian_mm_capture_folds.csv", "sound")
print(df.shape, df.fruit_id.nunique(), df.session_id.nunique())
print(df.device.value_counts().to_dict(), df.label_capture.value_counts().to_dict())
print(df.groupby('fold_fruit').fruit_id.nunique().to_dict())
print(df.file.str.extract(r'\.(\w+)$')[0].value_counts().to_dict())
print(df.head(2).T)
print(shutil.which("ffmpeg"))
import librosa; print(librosa.__version__)
zf = dc.open_zip("sound")
names = zf.namelist(); print(len(names), sum(n.startswith("__MACOSX") for n in names))
for dev in ["android","iphone"]:
    r = df[df.device==dev].iloc[0]
    b = dc.read_member(zf, r.zip_member); t=time.time()
    suf = os.path.splitext(r.file)[1]
    with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as f: f.write(b); p=f.name
    y, sr = librosa.load(p, sr=None); os.unlink(p)
    print(dev, r.file, len(b), sr, len(y)/sr, round(time.time()-t,2))

# ==========================================================================================
# cell 14: Extracting knock features from all 1002 recordings  [python]
# ==========================================================================================

from concurrent.futures import ThreadPoolExecutor
import pickle
t0=time.time()
with ThreadPoolExecutor(4) as ex:
    outs = list(ex.map(lambda r: kf.process_file(r.zip_member, r.file), df.itertuples()))
print("done", round(time.time()-t0,1), "s")
ok = np.array([o["ok"] for o in outs]); print("unreadable:", (~ok).sum())
for o, r in zip(outs, df.itertuples()):
    if not o["ok"]: print(r.file, o["error"])
nk = np.array([o["n_knocks"] for o in outs])
print("n_knocks quantiles", np.percentile(nk,[0,5,25,50,75,95,100]), "total", nk.sum())
print(pd.Series(nk).groupby(df.device.values).describe().round(1))
dur = np.array([o["duration_s"] for o in outs]); print("duration quantiles", np.percentile(dur,[0,5,50,95,100]).round(1))
with open("handoff/knock_outs.pkl","wb") as fh: pickle.dump(outs, fh)

# ==========================================================================================
# cell 16: Building per-knock and per-recording feature tables  [python]
# ==========================================================================================

meta_cols = ["file","fruit_id","session_id","device","date_captured","label_capture","y","Maturity_Stage","Class","fold_fruit","fold_random"]
rows=[]; mels=[]
for o, r in zip(outs, df.itertuples()):
    for j,(f,t) in enumerate(zip(o["feats"], o["onsets_s"])):
        rows.append(dict(file=r.file, knock_idx=j, onset_s=float(t), **f))
    mels.append(o["mels"])
knock = pd.DataFrame(rows).merge(df[meta_cols], on="file", how="left")
MELS = np.concatenate(mels).astype(np.float32)
assert len(knock)==len(MELS)==nk.sum()
feat_cols = [c for c in knock.columns if c not in meta_cols+["knock_idx","onset_s"]]
print(len(feat_cols), knock.shape, MELS.shape)
# per-recording aggregation
g = knock.groupby("file")[feat_cols]
rec = pd.concat([g.mean().add_suffix("__mean"), g.std(ddof=0).add_suffix("__std")], axis=1)
rec["n_knocks"] = knock.groupby("file").size()
rec = df[meta_cols].merge(rec.reset_index(), on="file", how="left")
assert rec.n_knocks.notna().all() and len(rec)==len(df)
rec_feat_cols = [c for c in rec.columns if c not in meta_cols]
print(rec.shape, len(rec_feat_cols))
# save features
rec.to_parquet("features_sound.parquet", index=False)
knock.to_parquet("features_sound_knocks.parquet", index=False)
np.save("handoff/mels.npy", MELS)
print(knock[["centroid","dom_freq","decay20","rms","zcr"]].describe().round(3).T)
print(knock.groupby(["device","label_capture"])[["centroid","dom_freq","decay20"]].median().round(1))

# ==========================================================================================
# cell 19: Building feature tables keyed by unique recording  [python]
# ==========================================================================================

# unique-recording table (538): first row per file in frozen table; fold_fruit is file-consistent (check), fold_random take first row
chk = df.groupby("file")[["fold_fruit","fruit_id","label_capture","device","Maturity_Stage","Class"]].nunique().max(); print(chk.to_dict())
uniq = df.drop_duplicates("file").reset_index(drop=True).copy()
uniq["n_rows_in_fold_table"] = uniq.file.map(df.groupby("file").size())
uniq["n_sessions_in_fold_table"] = uniq.file.map(df.groupby("file").session_id.nunique())
print(len(uniq), uniq.fruit_id.nunique(), uniq.session_id.nunique(), uniq.device.value_counts().to_dict(), uniq.label_capture.value_counts().to_dict())
print(uniq.groupby("fold_fruit").agg(n=("file","size"), fruits=("fruit_id","nunique")).T)
# knock table from unique files
out_by_file = {r.file: o for o, r in zip(outs, df.itertuples())}
rows=[]; mels=[]
for r in uniq.itertuples():
    o = out_by_file[r.file]
    for j,(f,t) in enumerate(zip(o["feats"], o["onsets_s"])):
        rows.append(dict(file=r.file, knock_idx=j, onset_s=float(t), **f))
    mels.append(o["mels"])
knock = pd.DataFrame(rows).merge(uniq[meta_cols], on="file", how="left")
MELS = np.concatenate(mels).astype(np.float32)
assert len(knock)==len(MELS)
feat_cols = [c for c in knock.columns if c not in meta_cols+["knock_idx","onset_s"]]
g = knock.groupby("file", sort=False)[feat_cols]
rec = pd.concat([g.mean().add_suffix("__mean"), g.std(ddof=0).add_suffix("__std")], axis=1)
rec["n_knocks"] = knock.groupby("file", sort=False).size()
rec = uniq[meta_cols+["n_rows_in_fold_table","n_sessions_in_fold_table"]].merge(rec.reset_index(), on="file", how="left")
rec_feat_cols = [c for c in rec.columns if c.endswith(("__mean","__std"))] + ["n_knocks"]
assert rec[rec_feat_cols].notna().all().all() and len(rec)==len(uniq)
print(knock.shape, MELS.shape, rec.shape, len(rec_feat_cols))
rec.to_parquet("features_sound_recordings.parquet", index=False)
df.merge(rec[["file"]+rec_feat_cols], on="file", how="left").to_parquet("features_sound.parquet", index=False)   # 1002 rows aligned to frozen table
knock.to_parquet("features_sound_knocks.parquet", index=False)
np.save("handoff/mels.npy", MELS)
nk_u = rec.n_knocks.values; print("n_knocks (unique files) quantiles", np.percentile(nk_u,[0,5,25,50,75,95,100]), nk_u.sum())
print(knock.groupby(["device","label_capture"])[["centroid","dom_freq","decay20","flatness"]].median().round(3))

# ==========================================================================================
# cell 21: Training five classical models with fruit-grouped folds  [python]
# ==========================================================================================

exec(open("train_classical.py").read())
Xrec = rec[rec_feat_cols].values.astype(np.float64)
def broadcast(oof_u):  # unique-file OOF -> 1002-row frozen table
    idx = pd.Series(np.arange(len(uniq)), index=uniq.file)
    return oof_u[idx.loc[df.file].values]
def summarize(ev):
    s = ev["summary"]; return {lvl: f"{s[lvl]['accuracy_mean']:.3f}±{s[lvl]['accuracy_sd']:.3f} / F1 {s[lvl]['macro_f1_mean']:.3f}±{s[lvl]['macro_f1_sd']:.3f}" for lvl in s}
oofs = {}; summ = {}
for name in ["logreg","knn","rf","svm","lgbm"]:
    t=time.time(); oof_u = run_oof(name, uniq, Xrec, "fold_fruit"); oofs[name]=oof_u
    oof = broadcast(oof_u)
    p1,p2 = dc.save_predictions(df, oof, "fold_fruit", "sound", name)
    ev = dc.evaluate_oof(df, oof, "fold_fruit"); summ[name]=summarize(ev)
    # also unique-recording-level metrics
    ev_u = dc.evaluate_oof(uniq, oof_u, "fold_fruit")
    print(name, round(time.time()-t,1), "s | table rows:", summ[name], "| unique rec capture:", f"{ev_u['summary']['capture']['accuracy_mean']:.3f}")

# ==========================================================================================
# cell 22: Training five classical models with threading backend  [python]
# ==========================================================================================

from joblib import parallel_config
import joblib
_pc = parallel_config(backend="threading"); _pc.__enter__()
oofs = {}; summ = {}; evs={}
for name in ["logreg","knn","rf","svm","lgbm"]:
    t=time.time(); oof_u = run_oof(name, uniq, Xrec, "fold_fruit"); oofs[name]=oof_u
    oof = broadcast(oof_u)
    dc.save_predictions(df, oof, "fold_fruit", "sound", name)
    ev = dc.evaluate_oof(df, oof, "fold_fruit"); summ[name]=summarize(ev); evs[name]=ev
    ev_u = dc.evaluate_oof(uniq, oof_u, "fold_fruit")
    print(name, round(time.time()-t,1), "s |", summ[name], "| unique-rec:", f"{ev_u['summary']['capture']['accuracy_mean']:.3f} ses {ev_u['summary']['session']['accuracy_mean']:.3f}")
# session-level class balance in frozen table vs unique
ys,_,_ = dc.aggregate_to_session(df, np.zeros((len(df),3))); print("session labels (table):", np.bincount(ys))
print("confusion pooled logreg:", evs["logreg"]["pooled_capture"]["confusion"])

# ==========================================================================================
# cell 23: Checking session label consistency in frozen sound table  [python]
# ==========================================================================================

import warnings; warnings.filterwarnings("ignore")
ses = df.groupby("session_id").agg(n=("file","size"), nlab=("label_capture","nunique"), nfiles=("file","nunique"))
print("sessions:", len(ses), "| mixed-label sessions:", (ses.nlab>1).sum(), "| rows in mixed sessions:", ses.n[ses.nlab>1].sum())
print(df[df.fruit_id==8][["file","date_captured","label_capture","session_id","fold_random"]].sort_values("date_captured").to_string())
# unique-file table sessions
ses_u = uniq.groupby("session_id").agg(n=("file","size"), nlab=("label_capture","nunique")); print("uniq sessions:", len(ses_u), "mixed:", (ses_u.nlab>1).sum(), "labels:", uniq.drop_duplicates("session_id").label_capture.value_counts().to_dict())
# how many dates per fruit in sound table vs other modalities
full = pd.read_csv("durian_mm_capture_folds.csv")
print(full.groupby("modality").apply(lambda d: d.groupby("fruit_id").date_captured.nunique().mean()).round(2).to_dict())
print(full[(full.fruit_id==8)].groupby(["modality","date_captured","label_capture"]).size())

# ==========================================================================================
# cell 24: Reconstructing sound session dates from other modalities  [python]
# ==========================================================================================

other = full[full.modality!="sound"].groupby(["fruit_id","label_capture"]).date_captured.agg(lambda s: sorted(set(s)))
print("fruit×label with >1 date in other modalities:", (other.map(len)>1).sum(), "of", len(other))
print(other[other.map(len)>1].head(8))

# ==========================================================================================
# cell 25: Reconstructing sound session dates from other modalities  [python]
# ==========================================================================================

oth = full[full.modality!="sound"]
other = oth.groupby(["fruit_id","label_capture"]).date_captured.agg(lambda s: sorted(set(s)))
multi = other[other.map(len)>1]
print("fruit x label combos with >1 date in other modalities:", len(multi), "of", len(other))
print(multi.head(8))

# ==========================================================================================
# cell 26: Building corrected 538-row sound table and re-evaluating  [python]
# ==========================================================================================

date_map = other.map(lambda l: l[0]).to_dict()
rec2 = uniq.copy()
rec2["date_captured_frozen"] = rec2.date_captured
rec2["date_captured"] = [date_map.get((f, l)) for f, l in zip(rec2.fruit_id, rec2.label_capture)]
assert rec2.date_captured.notna().all()
rec2["session_id"] = rec2.fruit_id.astype(int).astype(str) + "_" + rec2.date_captured.astype(str)
ses2 = rec2.groupby("session_id").agg(n=("file","size"), nlab=("label_capture","nunique"), ndev=("device","nunique"))
print("sessions:", len(ses2), "mixed-label:", (ses2.nlab>1).sum(), "| files/session:", ses2.n.value_counts().to_dict(), "| devices/session:", ses2.ndev.value_counts().to_dict())
print("session labels:", rec2.drop_duplicates("session_id").label_capture.value_counts().to_dict())
print("fold_random (first-row) balance:", rec2.fold_random.value_counts().sort_index().to_dict())
# fruits with both sessions in same fold_fruit? (should be: fruit-grouped)
print("fruits spanning >1 fold_fruit:", (rec2.groupby("fruit_id").fold_fruit.nunique()>1).sum())
rec2.to_parquet("handoff/rec2_meta.parquet", index=False)
# re-evaluate the five models on corrected table + save
def save_both(name, oof_u, split):
    # primary: corrected 538-row table
    p1,p2 = dc.save_predictions(rec2, oof_u, split, "sound", name)
    ev = dc.evaluate_oof(rec2, oof_u, split)
    # secondary: frozen 1002-row alignment
    oof = broadcast(oof_u)
    pred = df.copy()
    for i,c in enumerate(dc.CLASSES): pred[f"p_{c}"] = oof[:,i]
    pred["model"]=name; pred.to_csv(f"results/oof_sound__{name}__{split}__frozen1002rows.csv", index=False)
    return ev
summ={}; evs={}
for name in ["logreg","knn","rf","svm","lgbm"]:
    evs[name] = save_both(name, oofs[name], "fold_fruit"); summ[name]=summarize(evs[name]); print(name, summ[name])

# ==========================================================================================
# cell 28: Filling endpoint dates for label-mismatch fruits and re-evaluating  [python]
# ==========================================================================================

print(df[df.fruit_id.isin(miss.fruit_id)][["fruit_id","file","label_capture","actual"]].drop_duplicates("file").query("label_capture!='Unripe'").to_string())
endpoint_date = oth.groupby("fruit_id").date_captured.max().to_dict()
rec2["date_src"] = np.where(rec2.date_captured.notna(), "matched_label", "endpoint_fallback")
rec2["date_captured"] = rec2.date_captured.fillna(rec2.fruit_id.map(endpoint_date))
assert rec2.date_captured.notna().all()
rec2["session_id"] = rec2.fruit_id.astype(int).astype(str) + "_" + rec2.date_captured.astype(str)
ses2 = rec2.groupby("session_id").agg(n=("file","size"), nlab=("label_capture","nunique"), ndev=("device","nunique"))
print("sessions:", len(ses2), "mixed-label:", (ses2.nlab>1).sum(), "| files/session:", ses2.n.value_counts().to_dict(), "| devices/session:", ses2.ndev.value_counts().to_dict())
print("session labels:", rec2.drop_duplicates("session_id").label_capture.value_counts().to_dict())
print("fold_random balance:", rec2.fold_random.value_counts().sort_index().to_dict(), "| fruits spanning >1 fold_fruit:", (rec2.groupby("fruit_id").fold_fruit.nunique()>1).sum())
rec2.to_parquet("handoff/rec2_meta.parquet", index=False)
summ={}; evs={}
for name in ["logreg","knn","rf","svm","lgbm"]:
    evs[name] = save_both(name, oofs[name], "fold_fruit"); summ[name]=summarize(evs[name]); print(name, summ[name])

# ==========================================================================================
# cell 29: Saving fold_fruit predictions for five classical models  [python]
# ==========================================================================================

def save_both(name, oof_u, split):
    dc.save_predictions(rec2, oof_u, split, "sound", name)
    ev = dc.evaluate_oof(rec2, oof_u, split)
    oof = broadcast(oof_u); pred = df.copy()
    for i,c in enumerate(dc.CLASSES): pred[f"p_{c}"] = oof[:,i]
    pred["model"]=name; pred.to_csv(f"results/oof_sound__{name}__{split}__frozen1002rows.csv", index=False)
    return ev
summ={}; evs={}
for name in ["logreg","knn","rf","svm","lgbm"]:
    evs[name] = save_both(name, oofs[name], "fold_fruit"); summ[name]=summarize(evs[name]); print(name, summ[name])

# ==========================================================================================
# cell 34: Running leakage, LOBO and cross-device experiments  [python]
# ==========================================================================================

NJ = 2
import torch; torch.set_num_threads(1)
from sklearn.metrics import accuracy_score, f1_score
res_rand = {}
for name in ["lgbm","knn","rf"]:
    oof_r = run_oof(name, rec2, Xrec, "fold_random")
    ev = save_both(name, oof_r, "fold_random"); res_rand[name] = summarize(ev); print("fold_random", name, res_rand[name])
# LOBO
lobo_rows = []
for name in ["lgbm","knn"]:
    for b, tr, te in dc.leave_one_batch_out(rec2):
        _, p = fit_predict(name, Xrec[tr], rec2.y.values[tr], rec2.fruit_id.values[tr], Xrec[te])
        m = dc.metrics_from_probs(rec2.y.values[te], p)
        ys, Ps, _ = dc.aggregate_to_session(rec2.iloc[te], p); ms = dc.metrics_from_probs(ys, Ps)
        lobo_rows.append(dict(model=name, held_out_batch=b, n_test=len(te), n_train=len(tr), n_test_fruits=rec2.iloc[te].fruit_id.nunique(),
                              test_label_dist=str(rec2.iloc[te].label_capture.value_counts().to_dict()),
                              capture_acc=m["accuracy"], capture_macro_f1=m["macro_f1"], session_acc=ms["accuracy"], session_macro_f1=ms["macro_f1"],
                              recall_per_class=str(m["recall_per_class"]), confusion=str(m["confusion"])))
lobo = pd.DataFrame(lobo_rows); print(lobo[["model","held_out_batch","n_test","capture_acc","capture_macro_f1","session_acc","session_macro_f1","test_label_dist"]].round(3).to_string())
# cross-device: per fruit fold, train on device A (train fruits) -> test device B (test fruits)
cd_rows=[]
for name in ["lgbm","knn"]:
    for trd, ted in [("iphone","android"),("android","iphone"),("both","android"),("both","iphone"),("iphone","iphone"),("android","android")]:
        accs=[]; f1s=[]
        for k, tr, te in dc.iter_folds(rec2, "fold_fruit"):
            tr2 = tr if trd=="both" else tr[rec2.device.values[tr]==trd]
            te2 = te[rec2.device.values[te]==ted]
            _, p = fit_predict(name, Xrec[tr2], rec2.y.values[tr2], rec2.fruit_id.values[tr2], Xrec[te2])
            yp = p.argmax(1); accs.append(accuracy_score(rec2.y.values[te2], yp)); f1s.append(f1_score(rec2.y.values[te2], yp, average="macro"))
        cd_rows.append(dict(model=name, train_device=trd, test_device=ted, acc_mean=np.mean(accs), acc_sd=np.std(accs,ddof=1), f1_mean=np.mean(f1s), f1_sd=np.std(f1s,ddof=1), n_folds=5))
cd = pd.DataFrame(cd_rows); print(cd.round(3).to_string())
lobo.to_csv("handoff/lobo_all.csv", index=False); cd.to_csv("handoff/crossdevice_all.csv", index=False)

# ==========================================================================================
# cell 42: Aggregating CNN knock probabilities to recordings and sessions  [python]
# ==========================================================================================

print(os.path.exists("handoff/cnn_oof_knock__fold_fruit.npy"))
def cnn_eval(split):
    Pk = np.load(f"handoff/cnn_oof_knock__{split}.npy")
    yk = knock.y.values; fold_k = knock.file.map(rec2.set_index("file")[split]).values
    knock_acc = [accuracy_score(yk[fold_k==k], Pk[fold_k==k].argmax(1)) for k in range(5)]
    knock_f1 = [f1_score(yk[fold_k==k], Pk[fold_k==k].argmax(1), average="macro") for k in range(5)]
    tmp = pd.DataFrame(Pk, columns=dc.CLASSES); tmp["file"]=knock.file.values
    Prec = tmp.groupby("file")[dc.CLASSES].mean().loc[rec2.file].values
    return Pk, Prec, dict(knock_acc_mean=np.mean(knock_acc), knock_acc_sd=np.std(knock_acc,ddof=1), knock_f1_mean=np.mean(knock_f1), knock_f1_sd=np.std(knock_f1,ddof=1), n_knocks=len(yk))
Pk_f, Prec_cnn, kstats = cnn_eval("fold_fruit"); oofs["cnn"] = Prec_cnn
evs["cnn"] = save_both("cnn", Prec_cnn, "fold_fruit"); summ["cnn"] = summarize(evs["cnn"])
print("knock-level:", {k: round(v,3) for k,v in kstats.items()}); print("cnn", summ["cnn"])
json.dump(kstats, open("results/metrics_sound__cnn__knocklevel__fold_fruit.json","w"), indent=1)
yp = Prec_cnn.argmax(1); print({d: round(accuracy_score(rec2.y.values[rec2.device.values==d], yp[rec2.device.values==d]),3) for d in ["iphone","android"]})

# ==========================================================================================
# cell 50: Saving CNN recording probabilities and nearest-neighbour diagnostic  [python]
# ==========================================================================================

np.save("handoff/cnn_recprobs_fold_fruit.npy", Prec_cnn)
nnres.to_csv("results/diag_sound_random_split_nearest_neighbour.csv", index=False)
rec2["n_knocks"] = rec.n_knocks.values
print("ok")

# ==========================================================================================
# cell 52: Building CNN leave-one-batch-out table  [python]
# ==========================================================================================

lb = json.load(open("handoff/cnn_lobo_recprobs.json"))
rows=[]
for b, d in lb.items():
    t = pd.DataFrame(d).set_index("file").loc[rec2.file[rec2.Maturity_Stage==b]]
    sub = rec2[rec2.Maturity_Stage==b]; P = t[dc.CLASSES].values
    m = dc.metrics_from_probs(sub.y.values, P); ys, Ps, _ = dc.aggregate_to_session(sub, P); mss = dc.metrics_from_probs(ys, Ps)
    rows.append(dict(model="cnn", held_out_batch=b, n_test=len(sub), n_train=len(rec2)-len(sub), n_test_fruits=sub.fruit_id.nunique(),
                     test_label_dist=str(sub.label_capture.value_counts().to_dict()), capture_acc=m["accuracy"], capture_macro_f1=m["macro_f1"],
                     session_acc=mss["accuracy"], session_macro_f1=mss["macro_f1"], recall_per_class=str(m["recall_per_class"]), confusion=str(m["confusion"])))
lobo_cnn = pd.DataFrame(rows)
lobo_final = pd.concat([lobo_cnn, lobo[lobo.model=="lgbm"]], ignore_index=True)
lobo_cnn.to_csv("results/lobo_sound__cnn.csv", index=False); lobo_final.to_csv("results/lobo_sound__all_models.csv", index=False)
print(lobo_final[["model","held_out_batch","n_test","capture_acc","capture_macro_f1","session_acc","session_macro_f1","recall_per_class"]].round(3).to_string())

# ==========================================================================================
# cell 53: Evaluating CNN random-split run  [python]
# ==========================================================================================

Pk_r, Prec_cnn_r, kstats_r = cnn_eval("fold_random")
ev_r = save_both("cnn", Prec_cnn_r, "fold_random"); print("cnn fold_random:", summarize(ev_r)); print({k: round(float(v),3) for k,v in kstats_r.items()})
json.dump({k: float(v) for k,v in kstats_r.items()}, open("results/metrics_sound__cnn__knocklevel__fold_random.json","w"), indent=1)
json.dump({k: float(v) for k,v in kstats.items()}, open("results/metrics_sound__cnn__knocklevel__fold_fruit.json","w"), indent=1)

# ==========================================================================================
# cell 55: Building cross-device table and methods document  [python]
# ==========================================================================================

xd = json.load(open("handoff/cnn_crossdevice_recprobs.json"))
rows=[]
for pair in ["iphone->android","android->iphone"]:
    accs=[]; f1s=[]
    for k in range(5):
        t = pd.DataFrame(xd[f"{pair}|{k}"]).set_index("file"); sub = rec2.set_index("file").loc[t.index]
        yp_ = t[dc.CLASSES].values.argmax(1); accs.append(accuracy_score(sub.y, yp_)); f1s.append(f1_score(sub.y, yp_, average="macro"))
    rows.append(dict(model="cnn", train_device=pair.split("->")[0], test_device=pair.split("->")[1], acc_mean=np.mean(accs), acc_sd=np.std(accs,ddof=1), f1_mean=np.mean(f1s), f1_sd=np.std(f1s,ddof=1), n_folds=5))
# both -> device from main OOF
for ted in ["android","iphone"]:
    accs=[]; f1s=[]
    for k,tr,te in dc.iter_folds(rec2,"fold_fruit"):
        te2 = te[rec2.device.values[te]==ted]; yp_ = Prec_cnn[te2].argmax(1)
        accs.append(accuracy_score(rec2.y.values[te2], yp_)); f1s.append(f1_score(rec2.y.values[te2], yp_, average="macro"))
    rows.append(dict(model="cnn", train_device="both", test_device=ted, acc_mean=np.mean(accs), acc_sd=np.std(accs,ddof=1), f1_mean=np.mean(f1s), f1_sd=np.std(f1s,ddof=1), n_folds=5))
cd_cnn = pd.DataFrame(rows); cd_cnn.to_csv("results/crossdevice_sound__cnn.csv", index=False)
cd_final = pd.concat([cd_cnn, cd[cd.model=="lgbm"]], ignore_index=True); cd_final.to_csv("results/crossdevice_sound__all_models.csv", index=False)
print(cd_final.round(3).to_string())
exec(open("build_methods.py").read())
