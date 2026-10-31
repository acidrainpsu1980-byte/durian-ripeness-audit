# rgb track - literal cells executed in the interactive analysis session (chronological, successful cells only,
# exploration/figure cells omitted).  Kept for transparency: this is the code that produced the shipped
# intermediate/oof files.  It is NOT a standalone script (it was executed cell by cell in one kernel, with files
# being edited between cells); the consolidated final-state driver is src/rgb/rgb_models.py.
# Absolute paths were not present in these cells; the fold table was read from the working directory.

# ==========================================================================================
# cell 1: Loading RGB fold table and checking packages  [python]
# ==========================================================================================
import sys, os, time, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
import durian_common as dc
df = dc.load_folds("durian_mm_capture_folds.csv", "rgb")
print(df.shape, df.fruit_id.nunique(), df.session_id.nunique())
print(df.label_capture.value_counts().to_dict())
print(df.orientation.value_counts().to_dict())
print(df.device.value_counts(dropna=False).to_dict())
print(df.Maturity_Stage.value_counts().to_dict())
print(pd.crosstab(df.fold_fruit, df.label_capture))
print(df.zip_member.head(3).tolist())
import importlib
for m in ["sklearn","lightgbm","torch","torchvision","PIL","cv2","skimage","pyarrow"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod,"__version__","?"))
    except Exception as e: print(m, "MISSING", e)
import torch; print("threads", torch.get_num_threads())

# ==========================================================================================
# cell 18: Training LR, RF, LightGBM on hand-crafted features  [python]
# ==========================================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
import lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
SEED=42
F = pd.read_parquet("features_rgb.parquet")
meta_cols = ["modality","file","fruit_id","session_id","orientation","date_captured","label_capture","y","Maturity_Stage","Class","fold_fruit","fold_random","camera","seg_ok","mask_frac"]
feat_cols = [c for c in F.columns if c not in meta_cols]
print(len(feat_cols), "features")
assert (F.file.values == df.file.values).all()
Xh = F[feat_cols].values.astype(np.float32); y = df.y.values; groups = df.fruit_id.values

def make_model(name):
    if name == "logreg":
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED))
        return GridSearchCV(pipe, {"logisticregression__C":[0.01,0.1,1.0]}, cv=GroupKFold(3), scoring="f1_macro", n_jobs=4)
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=4)
    if name == "lgbm":
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, class_weight="balanced", random_state=SEED, n_jobs=4, verbose=-1)

def run_cv(Xm, name, split, fold_iter=None):
    oof = np.zeros((len(y),3))
    for k, tr, te in (fold_iter or dc.iter_folds(df, split)):
        mdl = make_model(name)
        if name=="logreg": mdl.fit(Xm[tr], y[tr], groups=groups[tr])
        else: mdl.fit(Xm[tr], y[tr])
        oof[te] = mdl.predict_proba(Xm[te])
    return oof

res = {}
for name in ["logreg","rf","lgbm"]:
    t=time.time(); oof = run_cv(Xh, name, "fold_fruit")
    p1,p2 = dc.save_predictions(df, oof, "fold_fruit", "rgb", f"hc_{name}")
    ev = dc.evaluate_oof(df, oof, "fold_fruit"); res[name]=ev["summary"]
    print(name, f"{time.time()-t:.0f}s", {l:(round(ev['summary'][l]['accuracy_mean'],3), round(ev['summary'][l]['macro_f1_mean'],3)) for l in ["capture","session"]})

# ==========================================================================================
# cell 19: Training LR, RF, LightGBM with threading backend  [python]
# ==========================================================================================
from joblib import parallel_backend
res = {}
with parallel_backend("threading", n_jobs=4):
    for name in ["logreg","rf","lgbm"]:
        t=time.time(); oof = run_cv(Xh, name, "fold_fruit")
        p1,p2 = dc.save_predictions(df, oof, "fold_fruit", "rgb", f"hc_{name}")
        ev = dc.evaluate_oof(df, oof, "fold_fruit"); res[name]=ev["summary"]
        print(name, f"{time.time()-t:.0f}s", {l:(round(ev['summary'][l]['accuracy_mean'],3), round(ev['summary'][l]['macro_f1_mean'],3)) for l in ["capture","session"]})

# ==========================================================================================
# cell 20: Loading ImageNet weights and building fruit-centred 224 crops  [python]
# ==========================================================================================
import torch, torchvision, torch.nn as nn, torch.nn.functional as Fnn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, efficientnet_b0, EfficientNet_B0_Weights
torch.set_num_threads(4); torch.manual_seed(SEED)
t=time.time()
mnv2 = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).eval()
effb0 = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).eval()
print("weights loaded", round(time.time()-t,1))

# fruit-centred square crops at 224 from the 512 cache using the 128-px masks
def fruit_crop(i, out=224, margin=0.12):
    m = man.loc[i]; img = X[i]
    mk = cv2.resize(masks[i], (512,512), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mk)
    y0,y1,x0,x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0+y1)/2, (x0+x1)/2; side = max(y1-y0, x1-x0)*(1+2*margin); side = max(side, 96)
    # clip to actual image area (not the black padding)
    iy0, iy1, ix0, ix1 = m.y0, m.y0+m.h, m.x0, m.x0+m.w
    side = min(side, m.h, m.w)
    cy = np.clip(cy, iy0+side/2, iy1-side/2); cx = np.clip(cx, ix0+side/2, ix1-side/2)
    a0, b0 = int(round(cy-side/2)), int(round(cx-side/2)); s=int(round(side))
    crop_ = img[a0:a0+s, b0:b0+s]
    return cv2.resize(np.ascontiguousarray(crop_), (out,out), interpolation=cv2.INTER_AREA)

t=time.time()
C224 = np.stack([fruit_crop(i) for i in range(len(man))])
print(C224.shape, round(time.time()-t,1))
np.save("cache/rgb_crops224.npy", C224)
fig, axes = plt.subplots(3, 8, figsize=(16,6))
for r, cls in enumerate(dc.CLASSES):
    idx = np.random.RandomState(1).choice(man.index[man.label_capture==cls], 8, replace=False)
    for ax, i in zip(axes[r], idx): ax.imshow(C224[i]); ax.set_axis_off(); ax.set_title(f"{cls} {man.loc[i,'orientation']} {man.loc[i,'camera'][-5:]}", fontsize=7)
fig.tight_layout(); fig.savefig("figs/_crops.png", dpi=70); plt.close(fig)

# ==========================================================================================
# cell 21: Extracting MobileNetV2 and EfficientNet-B0 trunk activations  [python]
# ==========================================================================================
import torchvision.transforms.functional as TF
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1); STD = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
def views(batch_u8):  # (B,224,224,3) uint8 -> list of 5 normalized tensors
    x = torch.from_numpy(batch_u8).permute(0,3,1,2).float()/255.
    vs = [x, TF.hflip(x), TF.vflip(x), TF.rotate(x, 12, interpolation=TF.InterpolationMode.BILINEAR), TF.rotate(x, -12, interpolation=TF.InterpolationMode.BILINEAR)]
    return [(v-MEAN)/STD for v in vs]

nets = {"mobilenetv2": (mnv2, 17), "efficientnetb0": (effb0, 7)}
TR = {}; EMB = {}
with torch.no_grad():
    for nm, (net, cut) in nets.items():
        trunk = net.features[:cut]; rest = net.features[cut:]
        t=time.time(); tr_list=[]; emb_list=[]
        for s in range(0, len(C224), 32):
            vs = views(C224[s:s+32]); tb=[]; eb=[]
            for v in vs:
                a = trunk(v); tb.append(a.half())
                e = Fnn.adaptive_avg_pool2d(rest(a), 1).flatten(1); eb.append(e)
            tr_list.append(torch.stack(tb,1)); emb_list.append(torch.stack(eb,1))
        TR[nm] = torch.cat(tr_list).numpy(); EMB[nm] = torch.cat(emb_list).numpy()
        np.save(f"cache/trunk_{nm}.npy", TR[nm]); np.save(f"cache/emb_{nm}.npy", EMB[nm])
        print(nm, TR[nm].shape, EMB[nm].shape, f"{time.time()-t:.0f}s")

# ==========================================================================================
# cell 22: Fitting linear probes on frozen CNN embeddings  [python]
# ==========================================================================================
def make_probe():
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED))
    return GridSearchCV(pipe, {"logisticregression__C":[0.001,0.01,0.1]}, cv=GroupKFold(3), scoring="f1_macro", n_jobs=4)

def run_probe(E, split=None, fold_iter=None):
    oof = np.zeros((len(y),3))
    for k, tr, te in (fold_iter or dc.iter_folds(df, split)):
        mdl = make_probe(); mdl.fit(E[tr], y[tr], groups=groups[tr]); oof[te] = mdl.predict_proba(E[te])
    return oof

probe_res = {}
with parallel_backend("threading", n_jobs=4):
    for nm in nets:
        E = EMB[nm].mean(1)  # feature-level TTA over the 5 views
        t=time.time(); oof = run_probe(E, "fold_fruit")
        dc.save_predictions(df, oof, "fold_fruit", "rgb", f"{nm}_probe")
        ev = dc.evaluate_oof(df, oof, "fold_fruit"); probe_res[nm]=ev["summary"]
        print(nm, f"{time.time()-t:.0f}s", {l:(round(ev['summary'][l]['accuracy_mean'],3), round(ev['summary'][l]['macro_f1_mean'],3)) for l in ["capture","session"]})

# ==========================================================================================
# cell 23: Fine-tuning last block and head per fold  [python]
# ==========================================================================================
import copy
class Head(nn.Module):
    def __init__(self, net, cut):
        super().__init__()
        self.block = copy.deepcopy(net.features[cut:]); self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 3))
    def forward(self, a):
        return self.fc(Fnn.adaptive_avg_pool2d(self.block(a), 1).flatten(1))

def finetune_fold(nm, tr, te, epochs=10, bs=64, seed=SEED, verbose=False):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    net, cut = nets[nm]; A = TR[nm]
    model = Head(net, cut).train()
    cw = torch.tensor(len(tr)/(3*np.bincount(y[tr], minlength=3)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    opt = torch.optim.AdamW([{"params": model.block.parameters(), "lr": 1e-4}, {"params": model.fc.parameters(), "lr": 5e-4}], weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr)/bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4,5e-4], total_steps=steps, pct_start=0.15)
    yt = torch.from_numpy(y)
    for ep in range(epochs):
        perm = rng.permutation(tr); tot=0; n=0
        for s in range(0, len(perm), bs):
            idx = perm[s:s+bs]; v = rng.randint(0, 5, len(idx))          # random augmentation view per sample
            a = torch.from_numpy(A[idx, v].astype(np.float32))
            loss = crit(model(a), yt[idx]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()*len(idx); n += len(idx)
        if verbose: print(ep, round(tot/n,4))
    model.eval(); out = np.zeros((len(te),3))
    with torch.no_grad():
        for s in range(0, len(te), 128):
            idx = te[s:s+128]
            a = torch.from_numpy(A[idx].reshape(-1, *A.shape[2:]).astype(np.float32))
            p = torch.softmax(model(a), 1).view(len(idx), 5, 3).mean(1)
            out[s:s+128] = p.numpy()
    return out, model

def run_ft(nm, split=None, fold_iter=None, keep_models=False):
    oof = np.zeros((len(y),3)); models={}
    for k, tr, te in (fold_iter or dc.iter_folds(df, split)):
        t=time.time(); oof[te], mdl = finetune_fold(nm, tr, te)
        if keep_models: models[k]=mdl
        print(f"  {nm} fold {k} {time.time()-t:.0f}s acc={ (oof[te].argmax(1)==y[te]).mean():.3f}", flush=True)
    return oof, models

ft_res = {}; FT_MODELS = {}
for nm in nets:
    oof, mdls = run_ft(nm, "fold_fruit", keep_models=True); FT_MODELS[nm]=mdls
    dc.save_predictions(df, oof, "fold_fruit", "rgb", f"{nm}_ft")
    ev = dc.evaluate_oof(df, oof, "fold_fruit"); ft_res[nm]=ev["summary"]
    print(nm, {l:(round(ev['summary'][l]['accuracy_mean'],3), round(ev['summary'][l]['macro_f1_mean'],3)) for l in ["capture","session"]})

# ==========================================================================================
# cell 24: Caching last-block outputs for light fine-tuning  [python]
# ==========================================================================================
print('TR' in dir(), 'EMB' in dir())
# frozen forward of the last residual block -> cache its output (B,5,C,7,7); only the 1x1 projection conv + fc will be trained
LB = {}
with torch.no_grad():
    for nm, (net, cut) in nets.items():
        blk = net.features[cut:-1].eval(); A = TR[nm]; t=time.time(); outs=[]
        for s in range(0, len(A), 64):
            a = torch.from_numpy(A[s:s+64].reshape(-1, *A.shape[2:]).astype(np.float32))
            outs.append(blk(a).view(-1, 5, *blk(a[:1]).shape[1:]).half())
        LB[nm] = torch.cat(outs).numpy(); print(nm, LB[nm].shape, f"{time.time()-t:.0f}s")

# ==========================================================================================
# cell 25: Fine-tuning projection conv and head per fold  [python]
# ==========================================================================================
class Head2(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.proj = copy.deepcopy(net.features[-1]); self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 3))
    def forward(self, a):
        return self.fc(Fnn.adaptive_avg_pool2d(self.proj(a), 1).flatten(1))

def finetune_fold(nm, tr, te, epochs=10, bs=64, seed=SEED):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    net, _ = nets[nm]; A = LB[nm]
    model = Head2(net).train()
    cw = torch.tensor(len(tr)/(3*np.bincount(y[tr], minlength=3)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    opt = torch.optim.AdamW([{"params": model.proj.parameters(), "lr": 1e-4}, {"params": model.fc.parameters(), "lr": 5e-4}], weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr)/bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4,5e-4], total_steps=steps, pct_start=0.15)
    yt = torch.from_numpy(y)
    for ep in range(epochs):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), bs):
            idx = perm[s:s+bs]; v = rng.randint(0, 5, len(idx))
            a = torch.from_numpy(A[idx, v].astype(np.float32))
            loss = crit(model(a), yt[idx]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval(); out = np.zeros((len(te),3))
    with torch.no_grad():
        for s in range(0, len(te), 128):
            idx = te[s:s+128]; a = torch.from_numpy(A[idx].reshape(-1, *A.shape[2:]).astype(np.float32))
            out[s:s+128] = torch.softmax(model(a), 1).view(len(idx), 5, 3).mean(1).numpy()
    return out, model

t0=time.time(); oof_test, _ = finetune_fold("mobilenetv2", *list(dc.iter_folds(df,"fold_fruit"))[0][1:], epochs=1)
print("1 epoch fold time", round(time.time()-t0,1))

# ==========================================================================================
# cell 26: Running light fine-tuning for both nets over 5 folds  [python]
# ==========================================================================================
ft_res = {}; FT_MODELS = {}
for nm in nets:
    oof, mdls = run_ft(nm, "fold_fruit", keep_models=True); FT_MODELS[nm]=mdls
    dc.save_predictions(df, oof, "fold_fruit", "rgb", f"{nm}_ft")
    ev = dc.evaluate_oof(df, oof, "fold_fruit"); ft_res[nm]=ev["summary"]
    print(nm, {l:(round(ev['summary'][l]['accuracy_mean'],3), round(ev['summary'][l]['macro_f1_mean'],3)) for l in ["capture","session"]})

# ==========================================================================================
# cell 27: Comparing fine-tune configs on inner validation split  [python]
# ==========================================================================================
def finetune_fold(nm, tr, te, epochs=10, bs=64, seed=SEED, lr_proj=1e-4, lr_fc=5e-4, freeze_bn=False, weighted=True):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    net, _ = nets[nm]; A = LB[nm]
    model = Head2(net).train()
    if freeze_bn:
        for mod in model.proj.modules():
            if isinstance(mod, nn.BatchNorm2d): mod.eval(); mod.weight.requires_grad_(False); mod.bias.requires_grad_(False)
    cw = torch.tensor(len(tr)/(3*np.bincount(y[tr], minlength=3)), dtype=torch.float32) if weighted else None
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    opt = torch.optim.AdamW([{"params": [p for p in model.proj.parameters() if p.requires_grad], "lr": lr_proj}, {"params": model.fc.parameters(), "lr": lr_fc}], weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(tr)/bs)); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr_proj,lr_fc], total_steps=steps, pct_start=0.15)
    yt = torch.from_numpy(y)
    for ep in range(epochs):
        perm = rng.permutation(tr)
        for s in range(0, len(perm), bs):
            idx = perm[s:s+bs]; v = rng.randint(0, 5, len(idx))
            a = torch.from_numpy(A[idx, v].astype(np.float32))
            loss = crit(model(a), yt[idx]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval(); out = np.zeros((len(te),3))
    with torch.no_grad():
        for s in range(0, len(te), 128):
            idx = te[s:s+128]; a = torch.from_numpy(A[idx].reshape(-1, *A.shape[2:]).astype(np.float32))
            out[s:s+128] = torch.softmax(model(a), 1).view(len(idx), 5, 3).mean(1).numpy()
    return out, model

from sklearn.metrics import f1_score
_, tr0, _ = list(dc.iter_folds(df,"fold_fruit"))[0]
itr, ite = next(GroupKFold(4).split(tr0, y[tr0], groups[tr0])); itr, ite = tr0[itr], tr0[ite]
cfgs = {"A_default": dict(), "B_frozenBN_lr3e-4_20ep": dict(freeze_bn=True, lr_proj=3e-4, lr_fc=1e-3, epochs=20),
        "C_frozenBN_unweighted": dict(freeze_bn=True, lr_proj=3e-4, lr_fc=1e-3, epochs=20, weighted=False),
        "D_fc_only_20ep": dict(lr_proj=0.0, freeze_bn=True, lr_fc=1e-3, epochs=20)}
for cn, kw in cfgs.items():
    for nm in nets:
        t=time.time(); p,_ = finetune_fold(nm, itr, ite, **kw)
        print(cn, nm, f"{time.time()-t:.0f}s acc={(p.argmax(1)==y[ite]).mean():.3f} f1={f1_score(y[ite], p.argmax(1), average='macro'):.3f}")
# reference: probe on same inner split
with parallel_backend("threading", n_jobs=4):
    for nm in nets:
        E = EMB[nm].mean(1); m = make_probe(); m.fit(E[itr], y[itr], groups=groups[itr]); p = m.predict_proba(E[ite])
        print("probe", nm, f"acc={(p.argmax(1)==y[ite]).mean():.3f} f1={f1_score(y[ite], p.argmax(1), average='macro'):.3f}")

# ==========================================================================================
# cell 28: Re-running fine-tune with selected config  [python]
# ==========================================================================================
FT_KW = cfgs["C_frozenBN_unweighted"]
def run_ft(nm, split=None, fold_iter=None, keep_models=False):
    oof = np.zeros((len(y),3)); models={}
    for k, tr, te in (fold_iter or dc.iter_folds(df, split)):
        oof[te], mdl = finetune_fold(nm, tr, te, **FT_KW)
        if keep_models: models[k]=mdl
    return oof, models
ft_res = {}; FT_MODELS = {}
for nm in nets:
    t=time.time(); oof, mdls = run_ft(nm, "fold_fruit", keep_models=True); FT_MODELS[nm]=mdls
    dc.save_predictions(df, oof, "fold_fruit", "rgb", f"{nm}_ft")
    ev = dc.evaluate_oof(df, oof, "fold_fruit"); ft_res[nm]=ev["summary"]
    print(nm, f"{time.time()-t:.0f}s", {l:(round(ev['summary'][l]['accuracy_mean'],3), round(ev['summary'][l]['macro_f1_mean'],3)) for l in ["capture","session"]})
# summary table of all fold_fruit models so far
rows=[]
for f in sorted(os.listdir("results")):
    if f.startswith("metrics_rgb__") and f.endswith("fold_fruit.json"):
        s = json.load(open("results/"+f))["summary"]; mdl = f[len("metrics_rgb__"):-len("__fold_fruit.json")]
        rows.append(dict(model=mdl, cap_acc=s["capture"]["accuracy_mean"], cap_f1=s["capture"]["macro_f1_mean"], ses_acc=s["session"]["accuracy_mean"], ses_acc_sd=s["session"]["accuracy_sd"], ses_f1=s["session"]["macro_f1_mean"], ses_f1_sd=s["session"]["macro_f1_sd"]))
summ = pd.DataFrame(rows).sort_values("ses_f1", ascending=False); print(summ.round(3).to_string(index=False))

# ==========================================================================================
# cell 29: Running leakage, LOBO and camera-shift experiments  [python]
# ==========================================================================================
BEST = "efficientnetb0_probe"; E = EMB["efficientnetb0"].mean(1)
from sklearn.metrics import accuracy_score
with parallel_backend("threading", n_jobs=4):
    # keep fold models for Grad-CAM
    PROBES = {}; oof_best = np.zeros((len(y),3))
    for k, tr, te in dc.iter_folds(df, "fold_fruit"):
        m = make_probe(); m.fit(E[tr], y[tr], groups=groups[tr]); PROBES[k]=m.best_estimator_; oof_best[te]=m.predict_proba(E[te])
    print("best C per fold:", [PROBES[k].named_steps["logisticregression"].C for k in range(5)])
    # 1) leakage: fold_random
    oof_rand = run_probe(E, "fold_random"); dc.save_predictions(df, oof_rand, "fold_random", "rgb", BEST)
    ev_r = dc.evaluate_oof(df, oof_rand, "fold_random"); ev_f = dc.evaluate_oof(df, oof_best, "fold_fruit")
    print("fold_random", {l:(round(ev_r['summary'][l]['accuracy_mean'],3), round(ev_r['summary'][l]['macro_f1_mean'],3)) for l in ["capture","session"]})
    # 2) LOBO
    lobo=[]
    for b, tr, te in dc.leave_one_batch_out(df):
        m = make_probe(); m.fit(E[tr], y[tr], groups=groups[tr]); p = m.predict_proba(E[te])
        cm = dc.metrics_from_probs(y[te], p); ys, Ps, _ = dc.aggregate_to_session(df.iloc[te], p); sm = dc.metrics_from_probs(ys, Ps)
        lobo.append(dict(held_out_batch=b, n_captures=len(te), n_sessions=len(ys), train_classes=np.bincount(y[tr],minlength=3).tolist(), test_classes=np.bincount(y[te],minlength=3).tolist(),
                         capture_acc=cm["accuracy"], capture_macro_f1=cm["macro_f1"], session_acc=sm["accuracy"], session_macro_f1=sm["macro_f1"], session_recall_per_class=sm["recall_per_class"], confusion_session=sm["confusion"]))
    lobo = pd.DataFrame(lobo); lobo.to_csv(f"results/lobo_rgb__{BEST}.csv", index=False)
    print(lobo[["held_out_batch","n_sessions","test_classes","capture_acc","capture_macro_f1","session_acc","session_macro_f1","session_recall_per_class"]].round(3).to_string(index=False))
    # 3) camera shift: within fold_fruit, train on fold-train images of camera A, test on fold-test images of camera B (and A->A as reference)
    cam = man.camera.values; cams_ = ["Canon EOS M50m2","Canon EOS R50"]
    rows=[]
    for ctr in cams_:
        for cte in cams_:
            oof_c = np.full((len(y),3), np.nan)
            for k, tr, te in dc.iter_folds(df, "fold_fruit"):
                tr_c = tr[cam[tr]==ctr]; te_c = te[cam[te]==cte]
                if len(te_c)==0: continue
                m = make_probe(); m.fit(E[tr_c], y[tr_c], groups=groups[tr_c]); oof_c[te_c] = m.predict_proba(E[te_c])
            ok = ~np.isnan(oof_c[:,0]); sub = df[ok]; p = oof_c[ok]
            cm = dc.metrics_from_probs(sub.y.values, p); ys, Ps, _ = dc.aggregate_to_session(sub, p); sm = dc.metrics_from_probs(ys, Ps)
            rows.append(dict(train_camera=ctr, test_camera=cte, n_train_imgs_per_fold_mean=float(np.mean([ (cam[tr]==ctr).sum() for _,tr,_ in dc.iter_folds(df,"fold_fruit")])), n_test_captures=int(ok.sum()), n_test_sessions=len(ys),
                             test_classes=np.bincount(sub.y.values,minlength=3).tolist(), capture_acc=cm["accuracy"], capture_macro_f1=cm["macro_f1"], session_acc=sm["accuracy"], session_macro_f1=sm["macro_f1"], session_recall_per_class=sm["recall_per_class"]))
    # reference: all-camera training evaluated per test camera (from the main OOF)
    for cte in cams_:
        ok = cam==cte; sub=df[ok]; p=oof_best[ok]; cm = dc.metrics_from_probs(sub.y.values, p); ys, Ps, _ = dc.aggregate_to_session(sub, p); sm = dc.metrics_from_probs(ys, Ps)
        rows.append(dict(train_camera="both", test_camera=cte, n_train_imgs_per_fold_mean=float(np.mean([len(tr) for _,tr,_ in dc.iter_folds(df,"fold_fruit")])), n_test_captures=int(ok.sum()), n_test_sessions=len(ys), test_classes=np.bincount(sub.y.values,minlength=3).tolist(),
                         capture_acc=cm["accuracy"], capture_macro_f1=cm["macro_f1"], session_acc=sm["accuracy"], session_macro_f1=sm["macro_f1"], session_recall_per_class=sm["recall_per_class"]))
    cs = pd.DataFrame(rows); cs.to_csv(f"results/camerashift_rgb__{BEST}.csv", index=False)
    print(cs.drop(columns=["n_train_imgs_per_fold_mean"]).round(3).to_string(index=False))
ct = pd.crosstab(man.camera, man.label_capture); ct.to_csv("results/crosstab_camera_x_class_rgb.csv"); print(ct)
from scipy.stats import chi2_contingency; print("chi2 p =", chi2_contingency(ct.values)[1])
