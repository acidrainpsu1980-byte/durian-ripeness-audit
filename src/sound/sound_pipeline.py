#!/usr/bin/env python
"""Sound (knock) track, consolidated from the interactive analysis session (final state).

Unit of analysis is the RECORDING (538 unique files: 269 sessions x 2 phones).  The table
data/durian_mm_sound_recordings.csv carries one row per recording with the frozen fruit-grouped fold and the
session date matched to the image modalities (column date_src documents how each date was assigned).

Run from work/sound/ (DURIAN_DATA_ROOT must point at the folder with dataset_clean_sound.zip; ffmpeg is needed
for the iPhone .m4a files):

    python ../../src/sound/sound_pipeline.py features        # knock detection + per-knock / per-recording features, log-mels
    python ../../src/sound/sound_pipeline.py classical       # logreg / knn / rf / svm / lgbm, leakage, LOBO, cross-device
    python ../../src/sound/cnn_train.py fold_fruit 20        # 2-D CNN on per-knock log-mels (knock-level OOF)
    python ../../src/sound/cnn_train.py fold_random 20
    python ../../src/sound/cnn_extra.py lobo 20
    python ../../src/sound/cnn_extra.py crossdevice 20
    python ../../src/sound/sound_pipeline.py cnn_aggregate   # knock -> recording -> session; oof_sound__cnn__*.csv, lobo_sound__cnn.csv

Literal executed cells: src/session_traces/sound_session.py.  Seed 42 throughout.
"""
import os, sys, json, time, pickle
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.abspath("numba_cache"))
import numpy as np, pandas as pd
import durian_common as dc
from sklearn.metrics import accuracy_score, f1_score

STAGE = sys.argv[1] if len(sys.argv) > 1 else "features"
os.makedirs("results", exist_ok=True); os.makedirs("handoff", exist_ok=True)
META_COLS = ["file", "fruit_id", "session_id", "device", "date_captured", "label_capture", "y", "Maturity_Stage", "Class", "fold_fruit", "fold_random"]

rec2 = dc.load_folds(dc.SOUND_RECORDINGS_CSV, "sound")           # 538 rows; session_id = fruit_id_date (matched dates)
assert rec2.file.is_unique and len(rec2) == 538, rec2.shape
# v1.2: DURIAN_SOUND_EXCLUDE = text file of recordings to drop (data/sound_exclude_wrong_session.txt lists the 106 Android
# recordings whose audio was recorded at the fruit's OTHER session; see data/sound_recording_manifest.csv).
_EXCL_PATH = os.environ.get("DURIAN_SOUND_EXCLUDE")
if _EXCL_PATH:
    _excl = set(open(_EXCL_PATH).read().split())
    assert _excl <= set(rec2.file), "exclusion list names unknown recordings"
    rec2 = rec2[~rec2.file.isin(_excl)].reset_index(drop=True)
    print("excluded", len(_excl), "recordings ->", len(rec2), flush=True)
print("recordings", len(rec2), "| sessions", rec2.session_id.nunique(), "| fruits", rec2.fruit_id.nunique(), flush=True)


# ============================================================================= features
if STAGE == "features":
    import knock_features as kf
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(int(os.environ.get("DURIAN_N_JOBS", "4"))) as ex:
        outs = list(ex.map(lambda r: kf.process_file(r.zip_member, r.file), rec2.itertuples()))
    print("decoded + segmented", len(outs), "files in", round(time.time() - t0, 1), "s", flush=True)
    bad = [(r.file, o["error"]) for o, r in zip(outs, rec2.itertuples()) if not o["ok"]]
    print("unreadable:", len(bad), bad[:5])
    with open("handoff/knock_outs.pkl", "wb") as fh: pickle.dump(outs, fh)

    rows, mels = [], []
    for o, r in zip(outs, rec2.itertuples()):
        for j, (f, t) in enumerate(zip(o["feats"], o["onsets_s"])):
            rows.append(dict(file=r.file, knock_idx=j, onset_s=float(t), **f))
        mels.append(o["mels"])
    knock = pd.DataFrame(rows).merge(rec2[META_COLS], on="file", how="left")
    MELS = np.concatenate(mels).astype(np.float32)
    assert len(knock) == len(MELS)
    feat_cols = [c for c in knock.columns if c not in META_COLS + ["knock_idx", "onset_s"]]
    g = knock.groupby("file", sort=False)[feat_cols]
    rec = pd.concat([g.mean().add_suffix("__mean"), g.std(ddof=0).add_suffix("__std")], axis=1)
    rec["n_knocks"] = knock.groupby("file", sort=False).size()
    rec = rec2[META_COLS + ["n_rows_in_fold_table", "n_sessions_in_fold_table"]].merge(rec.reset_index(), on="file", how="left")
    rec_feat_cols = [c for c in rec.columns if c.endswith(("__mean", "__std"))] + ["n_knocks"]
    assert rec[rec_feat_cols].notna().all().all() and len(rec) == len(rec2), "recordings without detected knocks"
    print("knocks", knock.shape, "mels", MELS.shape, "recording features", len(rec_feat_cols))
    rec.to_parquet("features_sound_recordings.parquet", index=False)
    knock.to_parquet("features_sound_knocks.parquet", index=False)
    np.save("handoff/mels.npy", MELS)
    rec2.to_parquet("handoff/rec2_meta.parquet", index=False)        # consumed by cnn_train.py / cnn_extra.py
    nk = rec.n_knocks.values
    json.dump(dict(n_recordings=int(len(rec)), n_knocks=int(nk.sum()), knocks_per_recording_quantiles=np.percentile(nk, [0, 5, 25, 50, 75, 95, 100]).tolist(),
                   unreadable=bad), open("results/knock_detection_summary.json", "w"), indent=1)


# ============================================================================= classical models
elif STAGE == "classical":
    from joblib import parallel_config
    exec(open(os.path.join(_HERE, "train_classical.py")).read())     # defines make_model / fit_predict / run_oof (NJ, SEED)
    rec = pd.read_parquet("features_sound_recordings.parquet")
    rec = rec.set_index("file").loc[rec2.file].reset_index()        # no-op without an exclusion list
    assert (rec.file.values == rec2.file.values).all()
    rec_feat_cols = [c for c in rec.columns if c.endswith(("__mean", "__std"))] + ["n_knocks"]
    Xrec = rec[rec_feat_cols].values.astype(np.float64)

    def summarize(ev):
        s = ev["summary"]
        return {lvl: f"{s[lvl]['accuracy_mean']:.3f}±{s[lvl]['accuracy_sd']:.3f} / F1 {s[lvl]['macro_f1_mean']:.3f}±{s[lvl]['macro_f1_sd']:.3f}" for lvl in s}

    _pc = parallel_config(backend="threading"); _pc.__enter__()
    oofs, evs = {}, {}
    for name in ["logreg", "knn", "rf", "svm", "lgbm"]:
        t = time.time(); oof_u = run_oof(name, rec2, Xrec, "fold_fruit"); oofs[name] = oof_u
        dc.save_predictions(rec2, oof_u, "fold_fruit", "sound", name)
        evs[name] = dc.evaluate_oof(rec2, oof_u, "fold_fruit"); print(name, round(time.time() - t, 1), "s", summarize(evs[name]), flush=True)
    # leakage control (capture-level random folds)
    for name in ["lgbm", "knn", "rf"]:
        oof_r = run_oof(name, rec2, Xrec, "fold_random"); dc.save_predictions(rec2, oof_r, "fold_random", "sound", name)
        print("fold_random", name, summarize(dc.evaluate_oof(rec2, oof_r, "fold_random")), flush=True)
    # leave-one-harvest-batch-out
    lobo_rows = []
    for name in ["lgbm", "knn"]:
        for b, tr, te in dc.leave_one_batch_out(rec2):
            _, p = fit_predict(name, Xrec[tr], rec2.y.values[tr], rec2.fruit_id.values[tr], Xrec[te])
            m = dc.metrics_from_probs(rec2.y.values[te], p); ys, Ps, _ = dc.aggregate_to_session(rec2.iloc[te], p); ms = dc.metrics_from_probs(ys, Ps)
            lobo_rows.append(dict(model=name, held_out_batch=b, n_test=len(te), n_train=len(tr), n_test_fruits=rec2.iloc[te].fruit_id.nunique(),
                                  test_label_dist=str(rec2.iloc[te].label_capture.value_counts().to_dict()),
                                  capture_acc=m["accuracy"], capture_macro_f1=m["macro_f1"], session_acc=ms["accuracy"], session_macro_f1=ms["macro_f1"],
                                  recall_per_class=str(m["recall_per_class"]), confusion=str(m["confusion"])))
    lobo = pd.DataFrame(lobo_rows); lobo.to_csv("handoff/lobo_all.csv", index=False)
    print(lobo[["model", "held_out_batch", "n_test", "capture_acc", "capture_macro_f1", "session_acc", "session_macro_f1"]].round(3).to_string())
    # cross-device transfer (train phone A -> test phone B within the fruit folds)
    cd_rows = []
    for name in ["lgbm", "knn"]:
        for trd, ted in [("iphone", "android"), ("android", "iphone"), ("both", "android"), ("both", "iphone"), ("iphone", "iphone"), ("android", "android")]:
            accs, f1s = [], []
            for k, tr, te in dc.iter_folds(rec2, "fold_fruit"):
                tr2 = tr if trd == "both" else tr[rec2.device.values[tr] == trd]; te2 = te[rec2.device.values[te] == ted]
                _, p = fit_predict(name, Xrec[tr2], rec2.y.values[tr2], rec2.fruit_id.values[tr2], Xrec[te2])
                yp = p.argmax(1); accs.append(accuracy_score(rec2.y.values[te2], yp)); f1s.append(f1_score(rec2.y.values[te2], yp, average="macro"))
            cd_rows.append(dict(model=name, train_device=trd, test_device=ted, acc_mean=np.mean(accs), acc_sd=np.std(accs, ddof=1), f1_mean=np.mean(f1s), f1_sd=np.std(f1s, ddof=1), n_folds=5))
    cd = pd.DataFrame(cd_rows); cd.to_csv("handoff/crossdevice_all.csv", index=False); print(cd.round(3).to_string())


# ============================================================================= CNN aggregation
elif STAGE == "cnn_aggregate":
    knock = pd.read_parquet("features_sound_knocks.parquet")

    def cnn_eval(split):
        Pk = np.load(f"handoff/cnn_oof_knock__{split}.npy")
        yk = knock.y.values; fold_k = knock.file.map(rec2.set_index("file")[split]).values
        knock_acc = [accuracy_score(yk[fold_k == k], Pk[fold_k == k].argmax(1)) for k in range(5)]
        knock_f1 = [f1_score(yk[fold_k == k], Pk[fold_k == k].argmax(1), average="macro") for k in range(5)]
        tmp = pd.DataFrame(Pk, columns=dc.CLASSES); tmp["file"] = knock.file.values
        Prec = tmp.groupby("file")[dc.CLASSES].mean().loc[rec2.file].values          # mean knock probability per recording
        return Pk, Prec, dict(knock_acc_mean=float(np.mean(knock_acc)), knock_acc_sd=float(np.std(knock_acc, ddof=1)),
                              knock_f1_mean=float(np.mean(knock_f1)), knock_f1_sd=float(np.std(knock_f1, ddof=1)), n_knocks=int(len(yk)))

    for split in ["fold_fruit", "fold_random"]:
        if not os.path.exists(f"handoff/cnn_oof_knock__{split}.npy"):
            print("missing handoff/cnn_oof_knock__%s.npy - run cnn_train.py %s first" % (split, split)); continue
        Pk, Prec, kstats = cnn_eval(split)
        dc.save_predictions(rec2, Prec, split, "sound", "cnn")
        json.dump(kstats, open(f"results/metrics_sound__cnn__knocklevel__{split}.json", "w"), indent=1)
        ev = dc.evaluate_oof(rec2, Prec, split)["summary"]
        print(split, "knock-level", {k: round(v, 3) for k, v in kstats.items() if k != "n_knocks"},
              "| session acc %.3f f1 %.3f" % (ev["session"]["accuracy_mean"], ev["session"]["macro_f1_mean"]), flush=True)
        if split == "fold_fruit": np.save("handoff/cnn_recprobs_fold_fruit.npy", Prec)
    # LOBO (per-recording probabilities written by cnn_extra.py lobo)
    if os.path.exists("handoff/cnn_lobo_recprobs.json"):
        lb = json.load(open("handoff/cnn_lobo_recprobs.json")); rows = []
        for b, d in lb.items():
            sub = rec2[rec2.Maturity_Stage == b]; P = pd.DataFrame(d).set_index("file").loc[sub.file][dc.CLASSES].values
            m = dc.metrics_from_probs(sub.y.values, P); ys, Ps, _ = dc.aggregate_to_session(sub, P); mss = dc.metrics_from_probs(ys, Ps)
            rows.append(dict(model="cnn", held_out_batch=b, n_test=len(sub), n_train=len(rec2) - len(sub), n_test_fruits=sub.fruit_id.nunique(),
                             test_label_dist=str(sub.label_capture.value_counts().to_dict()), capture_acc=m["accuracy"], capture_macro_f1=m["macro_f1"],
                             session_acc=mss["accuracy"], session_macro_f1=mss["macro_f1"], recall_per_class=str(m["recall_per_class"]), confusion=str(m["confusion"])))
        lobo_cnn = pd.DataFrame(rows); lobo_cnn.to_csv("results/lobo_sound__cnn.csv", index=False)
        if os.path.exists("handoff/lobo_all.csv"):
            lobo = pd.read_csv("handoff/lobo_all.csv")
            pd.concat([lobo_cnn, lobo[lobo.model == "lgbm"]], ignore_index=True).to_csv("results/lobo_sound__all_models.csv", index=False)
        print(lobo_cnn[["held_out_batch", "n_test", "capture_acc", "session_acc", "session_macro_f1"]].round(3).to_string())
    # cross-device (cnn_extra.py crossdevice)
    if os.path.exists("handoff/cnn_crossdevice_recprobs.json") and os.path.exists("handoff/cnn_recprobs_fold_fruit.npy"):
        xd = json.load(open("handoff/cnn_crossdevice_recprobs.json")); Prec_cnn = np.load("handoff/cnn_recprobs_fold_fruit.npy"); rows = []
        for pair in ["iphone->android", "android->iphone"]:
            accs, f1s = [], []
            for k in range(5):
                t = pd.DataFrame(xd[f"{pair}|{k}"]).set_index("file"); t = t[t.index.isin(rec2.file)]; sub = rec2.set_index("file").loc[t.index]
                yp_ = t[dc.CLASSES].values.argmax(1); accs.append(accuracy_score(sub.y, yp_)); f1s.append(f1_score(sub.y, yp_, average="macro"))
            rows.append(dict(model="cnn", train_device=pair.split("->")[0], test_device=pair.split("->")[1], acc_mean=np.mean(accs), acc_sd=np.std(accs, ddof=1), f1_mean=np.mean(f1s), f1_sd=np.std(f1s, ddof=1), n_folds=5))
        for ted in ["android", "iphone"]:
            accs, f1s = [], []
            for k, tr, te in dc.iter_folds(rec2, "fold_fruit"):
                te2 = te[rec2.device.values[te] == ted]; yp_ = Prec_cnn[te2].argmax(1)
                accs.append(accuracy_score(rec2.y.values[te2], yp_)); f1s.append(f1_score(rec2.y.values[te2], yp_, average="macro"))
            rows.append(dict(model="cnn", train_device="both", test_device=ted, acc_mean=np.mean(accs), acc_sd=np.std(accs, ddof=1), f1_mean=np.mean(f1s), f1_sd=np.std(f1s, ddof=1), n_folds=5))
        cd_cnn = pd.DataFrame(rows); cd_cnn.to_csv("results/crossdevice_sound__cnn.csv", index=False)
        if os.path.exists("handoff/crossdevice_all.csv"):
            cd = pd.read_csv("handoff/crossdevice_all.csv")
            pd.concat([cd_cnn, cd[cd.model == "lgbm"]], ignore_index=True).to_csv("results/crossdevice_sound__all_models.csv", index=False)
        print(cd_cnn.round(3).to_string())
else:
    raise SystemExit(f"unknown stage {STAGE!r}: use features | classical | cnn_aggregate")
