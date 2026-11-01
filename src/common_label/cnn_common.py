"""Sound CNN refits on CLEAN recordings (wrong-session Android recordings excluded from TRAINING).

Jobs: outer:<k> (train folds != k, predict fold k), pair:<ab> (train folds not in {a,b}, predict a,b),
repro:<k> (outer fit with NO exclusion; reproduction check against cnn_oof_knock__fold_fruit.npy).
Excluded recordings are still predicted (diagnostics) but never trained on.
Original docstring follows.

Inner base-model fits for nested stacking (knocking-sound CNN).

For every unordered pair of outer folds {a, b} the CNN is trained on the knocks of the three
remaining folds and predicts the knocks of folds a and b.  For outer test fold k, the inner
out-of-fold prediction of a training fold j is then taken from the fit that excluded {k, j},
so no knock of fold k ever influences the stacker's training inputs.

Architecture, normalisation, augmentation, optimiser and schedule are identical to cnn_train.py
(20 epochs).  Each fit is seeded with 42 so fits are individually reproducible.
Writes handoff/cnn_pair_<a><b>.npy  (n_knocks x 3, NaN outside folds a, b).
"""
import os, sys, json, time, itertools
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("CNN_THREADS", "6")))
EPOCHS = 20
MELS = np.load("handoff/mels.npy")
knock = pd.read_parquet("features_sound_knocks.parquet")
rec2 = pd.read_parquet("handoff/rec2_meta.parquet")
assert len(knock) == len(MELS)
kf_ = knock.file.map(rec2.set_index("file")["fold_fruit"]).values.astype(int)
yk = knock.y.values.astype(int)


class Net(nn.Module):
    def __init__(s, nc=3):
        super().__init__()
        ch = [1, 16, 32, 64, 96]
        s.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(),
                                                nn.Conv2d(ch[i+1], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(),
                                                nn.MaxPool2d(2)) for i in range(4)])
        s.drop = nn.Dropout(0.3); s.fc = nn.Linear(ch[-1], nc)

    def forward(s, x):
        for b in s.blocks: x = b(x)
        return s.fc(s.drop(x.mean((2, 3))))


def augment(xb):
    sh = np.random.randint(-3, 4); xb = torch.roll(xb, sh, dims=3)
    f0 = np.random.randint(0, 56); w = np.random.randint(0, 9)
    xb = xb.clone(); xb[:, :, f0:f0+w, :] = 0
    return xb


def run_fit(tr, te):
    torch.manual_seed(42); np.random.seed(42)
    mu, sd = MELS[tr].mean(), MELS[tr].std()
    Xtr = torch.tensor((MELS[tr] - mu) / sd).unsqueeze(1); ytr = torch.tensor(yk[tr])
    Xte = torch.tensor((MELS[te] - mu) / sd).unsqueeze(1)
    cw = torch.tensor(len(ytr) / (3 * np.bincount(yk[tr], minlength=3)), dtype=torch.float32)
    net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * ((len(tr) + 127) // 128))
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 128):
            idx = perm[i:i+128]
            loss = F.cross_entropy(net(augment(Xtr[idx])), ytr[idx], weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(te), 512): P.append(F.softmax(net(Xte[i:i+512]), 1).numpy())
    if os.environ.get("CNN_SAVE_WEIGHTS"):            # v1.8: optional export of the fitted network (does not touch the RNG)
        _d = os.environ["CNN_SAVE_WEIGHTS"]; os.makedirs(_d, exist_ok=True)
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "mel_mean": float(mu), "mel_std": float(sd),
                    "epochs": EPOCHS, "seed": 42, "job": JOB, "classes": ["Unripe", "Ripe", "Overripe"]}, f"{_d}/{JOB}.pt")
    return np.concatenate(P)



# v1.10: CNN_EXCLUDE_FILE / CNN_OUT_DIR (optional) retrain without further recordings, e.g. the label-policy sessions of
# src/label_sensitivity/retrain_without_questionable.py; the defaults reproduce the paper exactly and the RNG stream is untouched
EXCL = set(open(os.environ.get("CNN_EXCLUDE_FILE", "handoff/exclude_recordings.txt")).read().split())
OUTDIR = os.environ.get("CNN_OUT_DIR", "handoff"); os.makedirs(OUTDIR, exist_ok=True)
# v1.3: one common session label for every modality (data/session_labels_common.csv)
_fy = pd.read_csv("handoff/file_label_common.csv").set_index("file").iloc[:, 0]
_new = knock.file.map(_fy)
yk = np.where(_new.notna(), _new.fillna(-1).values, yk).astype(int)
print("knock labels changed:", int((yk != knock.y.values).sum()), flush=True)
keep = ~knock.file.isin(EXCL).values
kr_ = knock.file.map(rec2.set_index("file")["fold_random"]).values.astype(int)
# v1.10 (docs/ANALYSIS_PLAN_anchor_physics.md, M1/M2): optional per-batch, per-mel-band standardization of the inputs before the
# usual global normalization. CNN_NORM=anchor: statistics from the knocks of each batch's anchor (intake) recordings listed in
# CNN_ANCHOR_FILE; CNN_NORM=all: from all verified knocks of each batch. Labels are not used. Default: off (paper behaviour).
_norm = os.environ.get("CNN_NORM", "")
if _norm:
    _bk = knock.Maturity_Stage.values
    _ref = keep.copy() if _norm == "all" else (keep & knock.file.isin(set(open(os.environ["CNN_ANCHOR_FILE"]).read().split())).values)
    MELS = MELS.astype(np.float32, copy=True)
    for _b in np.unique(_bk):
        _r = _bk == _b; _m = MELS[_r & _ref]
        _mu = _m.mean(axis=(0, 2), keepdims=True); _sd = _m.std(axis=(0, 2), keepdims=True); _sd[_sd == 0] = 1.0
        MELS[_r] = (MELS[_r] - _mu) / _sd
    print("CNN_NORM", _norm, "reference knocks", int(_ref.sum()), flush=True)
dev_k = knock.device.values; batch_k = knock.Maturity_Stage.values
for job in sys.argv[1:]:
    kind, arg = job.split(":")
    out = f"{OUTDIR}/cnn_common_{kind}_{arg.replace('>','2')}.npy"
    if os.path.exists(out): print("skip", out, flush=True); continue
    t = time.time()
    if kind in ("outer", "pair", "repro"):
        folds = [int(c) for c in arg]; trm = ~np.isin(kf_, folds); tem = np.isin(kf_, folds)
    elif kind == "rand":
        trm = kr_ != int(arg); tem = kr_ == int(arg)
    elif kind == "xdev":
        d, k = arg.split("|") if "|" in arg else arg.split("@"); a, b = d.split(">"); k = int(k)
        trm = (kf_ != k) & (dev_k == a); tem = (kf_ == k) & (dev_k == b)
    elif kind == "lobo":
        trm = batch_k != arg; tem = batch_k == arg
    if kind != "repro": trm &= keep
    tr = np.flatnonzero(trm); te = np.flatnonzero(tem)
    JOB = f"cnn_common_{kind}_{arg.replace('>','2')}"
    P = np.full((len(yk), 3), np.nan, np.float32); P[te] = run_fit(tr, te)
    np.save(out, P)
    acc = (P[te].argmax(1) == yk[te]).mean()
    print(f"{job}: train {len(tr)} knocks, test {len(te)}, knock acc {acc:.3f}, {time.time()-t:.0f}s", flush=True)
print("DONE", flush=True)
