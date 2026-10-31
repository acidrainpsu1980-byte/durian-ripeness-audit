#!/usr/bin/env python
"""RGB track, consolidated from the interactive analysis session (final state).

Run from work/rgb/ after build_cache.py (512-px cache + manifest) and extract_features.py (hand-crafted features
+ 128-px masks):

    python ../../src/rgb/build_cache.py
    python ../../src/rgb/extract_features.py
    python ../../src/rgb/rgb_models.py handcrafted     # logreg / rf / lgbm on colour+texture features (fold_fruit; lgbm also fold_random)
    python ../../src/rgb/rgb_models.py embed           # fruit-centred 224 crops, 5-view TTA embeddings of MobileNetV2 / EfficientNet-B0 (ImageNet)
    python ../../src/rgb/rgb_models.py probe           # linear probes (fold_fruit), then for EfficientNet-B0: fold_random, LOBO, camera shift
    python ../../src/rgb/rgb_models.py finetune        # optional: light fine-tuning (projection conv + head), config C_frozenBN_unweighted

Outputs in ./results: oof_rgb__hc_<model>__fold_fruit.csv, oof_rgb__hc_lgbm__fold_random.csv,
oof_rgb__<net>_probe__fold_fruit.csv, oof_rgb__efficientnetb0_probe__fold_random.csv, lobo_rgb__efficientnetb0_probe.csv,
camerashift_rgb__efficientnetb0_probe.csv, crosstab_camera_x_class_rgb.csv; cache/emb_<net>.npy.
Literal executed cells: src/session_traces/rgb_session.py.  Seed 42 throughout.
"""
import os, sys, time, json, copy, warnings
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
import numpy as np, pandas as pd
import durian_common as dc
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import f1_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
SEED = 42; N_JOBS = int(os.environ.get("DURIAN_N_JOBS", "4"))
STAGE = sys.argv[1] if len(sys.argv) > 1 else "handcrafted"
os.makedirs("results", exist_ok=True); os.makedirs("cache", exist_ok=True)

df = dc.load_folds(dc.FOLDS_CSV, "rgb")
man = pd.read_csv("cache/rgb512_manifest.csv")
assert (man.file.values == df.file.values).all(), "cache manifest not aligned with fold table"
y = df.y.values; groups = df.fruit_id.values
NETS = {"mobilenetv2": 17, "efficientnetb0": 7}          # index at which features[] is split into trunk / rest


def make_probe():
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))
    return GridSearchCV(pipe, {"logisticregression__C": [0.001, 0.01, 0.1]}, cv=GroupKFold(3), scoring="f1_macro", n_jobs=N_JOBS)


def run_probe(E, split):
    oof = np.zeros((len(y), 3))
    for k, tr, te in dc.iter_folds(df, split):
        mdl = make_probe(); mdl.fit(E[tr], y[tr], groups=groups[tr]); oof[te] = mdl.predict_proba(E[te])
    return oof


def show(tag, ev):
    s = ev["summary"]
    print(tag, {l: (round(s[l]["accuracy_mean"], 3), round(s[l]["macro_f1_mean"], 3)) for l in ["capture", "session"]}, flush=True)


# ============================================================================= hand-crafted features
if STAGE == "handcrafted":
    F = pd.read_parquet("features_rgb.parquet")
    meta_cols = ["modality", "file", "fruit_id", "session_id", "orientation", "date_captured", "label_capture", "y", "Maturity_Stage", "Class",
                 "fold_fruit", "fold_random", "camera", "seg_ok", "mask_frac"]
    feat_cols = [c for c in F.columns if c not in meta_cols]
    assert (F.file.values == df.file.values).all()
    Xh = F[feat_cols].values.astype(np.float32); print(len(feat_cols), "features")

    def make_model(name):
        if name == "logreg":
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))
            return GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0]}, cv=GroupKFold(3), scoring="f1_macro", n_jobs=N_JOBS)
        if name == "rf":
            return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=N_JOBS)
        if name == "lgbm":
            return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, class_weight="balanced", random_state=SEED, n_jobs=N_JOBS, verbose=-1)

    def run_cv(name, split):
        oof = np.zeros((len(y), 3))
        for k, tr, te in dc.iter_folds(df, split):
            mdl = make_model(name)
            if name == "logreg": mdl.fit(Xh[tr], y[tr], groups=groups[tr])
            else: mdl.fit(Xh[tr], y[tr])
            oof[te] = mdl.predict_proba(Xh[te])
        return oof

    with parallel_backend("threading", n_jobs=N_JOBS):
        for name in ["logreg", "rf", "lgbm"]:
            t = time.time(); oof = run_cv(name, "fold_fruit"); dc.save_predictions(df, oof, "fold_fruit", "rgb", f"hc_{name}")
            show(f"hc_{name} ({time.time()-t:.0f}s)", dc.evaluate_oof(df, oof, "fold_fruit"))
        oof = run_cv("lgbm", "fold_random"); dc.save_predictions(df, oof, "fold_random", "rgb", "hc_lgbm")
        show("hc_lgbm fold_random", dc.evaluate_oof(df, oof, "fold_random"))


# ============================================================================= embeddings
elif STAGE == "embed":
    import cv2, torch, torch.nn.functional as Fnn, torchvision.transforms.functional as TF
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
    torch.set_num_threads(N_JOBS); torch.manual_seed(SEED)
    X = np.load("cache/rgb512.npy", mmap_mode="r"); masks = np.load("cache/rgb_masks128.npy")
    nets = {"mobilenetv2": mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval(),
            "efficientnetb0": efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).eval()}

    def fruit_crop(i, out=224, margin=0.12):
        m = man.loc[i]; img = X[i]
        mk = cv2.resize(masks[i], (512, 512), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(mk); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2; side = max(y1 - y0, x1 - x0) * (1 + 2 * margin); side = max(side, 96)
        iy0, iy1, ix0, ix1 = m.y0, m.y0 + m.h, m.x0, m.x0 + m.w
        side = min(side, m.h, m.w)
        cy = np.clip(cy, iy0 + side / 2, iy1 - side / 2); cx = np.clip(cx, ix0 + side / 2, ix1 - side / 2)
        a0, b0 = int(round(cy - side / 2)), int(round(cx - side / 2)); s = int(round(side))
        return cv2.resize(np.ascontiguousarray(img[a0:a0 + s, b0:b0 + s]), (out, out), interpolation=cv2.INTER_AREA)

    t = time.time(); C224 = np.stack([fruit_crop(i) for i in range(len(man))]); np.save("cache/rgb_crops224.npy", C224)
    print("crops", C224.shape, f"{time.time()-t:.0f}s", flush=True)
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def views(batch_u8):      # 5 deterministic views: identity, hflip, vflip, rot +12, rot -12
        x = torch.from_numpy(batch_u8).permute(0, 3, 1, 2).float() / 255.
        vs = [x, TF.hflip(x), TF.vflip(x), TF.rotate(x, 12, interpolation=TF.InterpolationMode.BILINEAR), TF.rotate(x, -12, interpolation=TF.InterpolationMode.BILINEAR)]
        return [(v - MEAN) / STD for v in vs]

    with torch.no_grad():
        for nm, net in nets.items():
            cut = NETS[nm]; trunk = net.features[:cut]; rest = net.features[cut:]; t = time.time(); tr_list, emb_list = [], []
            for s in range(0, len(C224), 32):
                tb, eb = [], []
                for v in views(C224[s:s + 32]):
                    a = trunk(v); tb.append(a.half()); eb.append(Fnn.adaptive_avg_pool2d(rest(a), 1).flatten(1))
                tr_list.append(torch.stack(tb, 1)); emb_list.append(torch.stack(eb, 1))
            TRn = torch.cat(tr_list).numpy(); EMBn = torch.cat(emb_list).numpy()
            np.save(f"cache/trunk_{nm}.npy", TRn); np.save(f"cache/emb_{nm}.npy", EMBn)      # (N,5,C,h,w) float16 / (N,5,1280)
            print(nm, TRn.shape, EMBn.shape, f"{time.time()-t:.0f}s", flush=True)


# ============================================================================= linear probes + controls
elif STAGE == "probe":
    EMB = {nm: np.load(f"cache/emb_{nm}.npy") for nm in NETS}
    with parallel_backend("threading", n_jobs=N_JOBS):
        for nm in NETS:
            E = EMB[nm].mean(1)                                   # feature-level TTA over the 5 views
            t = time.time(); oof = run_probe(E, "fold_fruit"); dc.save_predictions(df, oof, "fold_fruit", "rgb", f"{nm}_probe")
            show(f"{nm}_probe ({time.time()-t:.0f}s)", dc.evaluate_oof(df, oof, "fold_fruit"))
        BEST = "efficientnetb0_probe"; E = EMB["efficientnetb0"].mean(1)
        oof_best = np.zeros((len(y), 3)); Cs = []
        for k, tr, te in dc.iter_folds(df, "fold_fruit"):
            m = make_probe(); m.fit(E[tr], y[tr], groups=groups[tr]); Cs.append(m.best_params_["logisticregression__C"]); oof_best[te] = m.predict_proba(E[te])
        print("best C per fold:", Cs)
        # leakage control
        oof_rand = run_probe(E, "fold_random"); dc.save_predictions(df, oof_rand, "fold_random", "rgb", BEST)
        show("fold_random", dc.evaluate_oof(df, oof_rand, "fold_random"))
        # leave-one-harvest-batch-out
        lobo = []
        for b, tr, te in dc.leave_one_batch_out(df):
            m = make_probe(); m.fit(E[tr], y[tr], groups=groups[tr]); p = m.predict_proba(E[te])
            cm = dc.metrics_from_probs(y[te], p); ys, Ps, _ = dc.aggregate_to_session(df.iloc[te], p); sm = dc.metrics_from_probs(ys, Ps)
            lobo.append(dict(held_out_batch=b, n_captures=len(te), n_sessions=len(ys), train_classes=np.bincount(y[tr], minlength=3).tolist(),
                             test_classes=np.bincount(y[te], minlength=3).tolist(), capture_acc=cm["accuracy"], capture_macro_f1=cm["macro_f1"],
                             session_acc=sm["accuracy"], session_macro_f1=sm["macro_f1"], session_recall_per_class=sm["recall_per_class"], confusion_session=sm["confusion"]))
        lobo = pd.DataFrame(lobo); lobo.to_csv(f"results/lobo_rgb__{BEST}.csv", index=False)
        print(lobo[["held_out_batch", "n_sessions", "capture_acc", "session_acc", "session_macro_f1", "session_recall_per_class"]].round(3).to_string(index=False))
        # camera shift (the camera changed mid-collection and is nested in harvest batch)
        cam = man.camera.values; cams_ = ["Canon EOS M50m2", "Canon EOS R50"]; rows = []
        for ctr in cams_:
            for cte in cams_:
                oof_c = np.full((len(y), 3), np.nan)
                for k, tr, te in dc.iter_folds(df, "fold_fruit"):
                    tr_c = tr[cam[tr] == ctr]; te_c = te[cam[te] == cte]
                    if len(te_c) == 0: continue
                    m = make_probe(); m.fit(E[tr_c], y[tr_c], groups=groups[tr_c]); oof_c[te_c] = m.predict_proba(E[te_c])
                ok = ~np.isnan(oof_c[:, 0]); sub = df[ok]; p = oof_c[ok]
                cm = dc.metrics_from_probs(sub.y.values, p); ys, Ps, _ = dc.aggregate_to_session(sub, p); sm = dc.metrics_from_probs(ys, Ps)
                rows.append(dict(train_camera=ctr, test_camera=cte, n_train_imgs_per_fold_mean=float(np.mean([(cam[tr] == ctr).sum() for _, tr, _ in dc.iter_folds(df, "fold_fruit")])),
                                 n_test_captures=int(ok.sum()), n_test_sessions=len(ys), test_classes=np.bincount(sub.y.values, minlength=3).tolist(),
                                 capture_acc=cm["accuracy"], capture_macro_f1=cm["macro_f1"], session_acc=sm["accuracy"], session_macro_f1=sm["macro_f1"], session_recall_per_class=sm["recall_per_class"]))
        for cte in cams_:
            ok = cam == cte; sub = df[ok]; p = oof_best[ok]
            cm = dc.metrics_from_probs(sub.y.values, p); ys, Ps, _ = dc.aggregate_to_session(sub, p); sm = dc.metrics_from_probs(ys, Ps)
            rows.append(dict(train_camera="both", test_camera=cte, n_train_imgs_per_fold_mean=float(np.mean([len(tr) for _, tr, _ in dc.iter_folds(df, "fold_fruit")])),
                             n_test_captures=int(ok.sum()), n_test_sessions=len(ys), test_classes=np.bincount(sub.y.values, minlength=3).tolist(),
                             capture_acc=cm["accuracy"], capture_macro_f1=cm["macro_f1"], session_acc=sm["accuracy"], session_macro_f1=sm["macro_f1"], session_recall_per_class=sm["recall_per_class"]))
        pd.DataFrame(rows).to_csv(f"results/camerashift_rgb__{BEST}.csv", index=False)
    ct = pd.crosstab(man.camera, man.label_capture); ct.to_csv("results/crosstab_camera_x_class_rgb.csv"); print(ct)
    from scipy.stats import chi2_contingency; print("chi2 p =", chi2_contingency(ct.values)[1])


# ============================================================================= optional light fine-tuning
elif STAGE == "finetune":
    import torch, torch.nn as nn, torch.nn.functional as Fnn
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
    torch.set_num_threads(N_JOBS)
    nets = {"mobilenetv2": mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval(),
            "efficientnetb0": efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).eval()}
    FT_KW = dict(freeze_bn=True, lr_proj=3e-4, lr_fc=1e-3, epochs=20, weighted=False)   # config "C_frozenBN_unweighted" selected on an inner split

    class Head2(nn.Module):
        def __init__(self, net):
            super().__init__(); self.proj = copy.deepcopy(net.features[-1]); self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 3))
        def forward(self, a): return self.fc(Fnn.adaptive_avg_pool2d(self.proj(a), 1).flatten(1))

    def finetune_fold(nm, LBn, tr, te, epochs=10, bs=64, seed=SEED, lr_proj=1e-4, lr_fc=5e-4, freeze_bn=False, weighted=True):
        torch.manual_seed(seed); rng = np.random.RandomState(seed); net = nets[nm]; A = LBn; model = Head2(net).train()
        if freeze_bn:
            for mod in model.proj.modules():
                if isinstance(mod, nn.BatchNorm2d): mod.eval(); mod.weight.requires_grad_(False); mod.bias.requires_grad_(False)
        cw = torch.tensor(len(tr) / (3 * np.bincount(y[tr], minlength=3)), dtype=torch.float32) if weighted else None
        crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
        opt = torch.optim.AdamW([{"params": [p for p in model.proj.parameters() if p.requires_grad], "lr": lr_proj}, {"params": model.fc.parameters(), "lr": lr_fc}], weight_decay=1e-4)
        steps = epochs * int(np.ceil(len(tr) / bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_proj, lr_fc], total_steps=steps, pct_start=0.15)
        yt = torch.from_numpy(y)
        for ep in range(epochs):
            perm = rng.permutation(tr)
            for s in range(0, len(perm), bs):
                idx = perm[s:s + bs]; v = rng.randint(0, 5, len(idx)); a = torch.from_numpy(A[idx, v].astype(np.float32))
                loss = crit(model(a), yt[idx]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval(); out = np.zeros((len(te), 3))
        with torch.no_grad():
            for s in range(0, len(te), 128):
                idx = te[s:s + 128]; a = torch.from_numpy(A[idx].reshape(-1, *A.shape[2:]).astype(np.float32))
                out[s:s + 128] = torch.softmax(model(a), 1).view(len(idx), 5, 3).mean(1).numpy()
        return out

    for nm, net in nets.items():
        TRn = np.load(f"cache/trunk_{nm}.npy", mmap_mode="r"); blk = net.features[NETS[nm]:-1].eval(); outs = []
        with torch.no_grad():               # cache the frozen last-block output; only the 1x1 projection conv + head are trained
            for s in range(0, len(TRn), 64):
                a = torch.from_numpy(np.asarray(TRn[s:s + 64]).reshape(-1, *TRn.shape[2:]).astype(np.float32))
                o = blk(a); outs.append(o.view(-1, 5, *o.shape[1:]).half())
        LBn = torch.cat(outs).numpy()
        oof = np.zeros((len(y), 3))
        for k, tr, te in dc.iter_folds(df, "fold_fruit"):
            t = time.time(); oof[te] = finetune_fold(nm, LBn, tr, te, **FT_KW); print(f"  {nm} fold {k} {time.time()-t:.0f}s", flush=True)
        dc.save_predictions(df, oof, "fold_fruit", "rgb", f"{nm}_ft"); show(f"{nm}_ft", dc.evaluate_oof(df, oof, "fold_fruit"))
else:
    raise SystemExit(f"unknown stage {STAGE!r}: use handcrafted | embed | probe | finetune")
