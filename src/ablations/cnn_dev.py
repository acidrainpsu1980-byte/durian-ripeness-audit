"""Knock CNN (same architecture, schedule and seed as cnn_common.py) on alternative fruit partitions, GPU if available.
usage: python cnn_dev.py OUTDIR P:kind:k ...   kind = iphone | android | iphoneM (iPhone recordings of sessions with a verified Android recording); outer fold k of partition P."""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
EPOCHS = 20
MELS = np.load("mels.npy")
knock = pd.read_parquet("features_sound_knocks.parquet")
rec2 = pd.read_parquet("rec2_meta.parquet")
part = pd.read_csv("partitions.csv").set_index("fruit_id")
assert len(knock) == len(MELS)
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
    Xtr = torch.tensor((MELS[tr] - mu) / sd).unsqueeze(1).to(DEV); ytr = torch.tensor(yk[tr]).to(DEV)
    Xte = torch.tensor((MELS[te] - mu) / sd).unsqueeze(1).to(DEV)
    cw = torch.tensor(len(ytr) / (3 * np.bincount(yk[tr], minlength=3)), dtype=torch.float32).to(DEV)
    net = Net().to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * ((len(tr) + 127) // 128))
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(tr)).to(DEV)
        for i in range(0, len(tr), 128):
            idx = perm[i:i+128]
            loss = F.cross_entropy(net(augment(Xtr[idx])), ytr[idx], weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(te), 512): P.append(F.softmax(net(Xte[i:i+512]), 1).cpu().numpy())
    return np.concatenate(P)


EXCL = set(open("exclude_recordings.txt").read().split())
_fy = pd.read_csv("file_label_common.csv").set_index("file").iloc[:, 0]
_new = knock.file.map(_fy)
yk = np.where(_new.notna(), _new.fillna(-1).values, yk).astype(int)
keep = ~knock.file.isin(EXCL).values
fid = knock.file.map(rec2.set_index("file")["fruit_id"]).astype(float).astype(int).values
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
_km = np.load("knock_meta.npz", allow_pickle=True); DEV_K = _km["device"].astype(str); BOTH_K = _km["both"].astype(bool)
assert len(DEV_K) == len(yk)
print("device", DEV, "knock labels changed", int((yk != knock.y.values).sum()), flush=True)
for job in sys.argv[2:]:
    P_, kind, arg = job.split(":")
    fn = f"{out}/cnn_{P_}_{kind}_{arg}.npy"
    if os.path.exists(fn): continue
    kf = part[P_].reindex(fid).values
    assert not np.isnan(kf[keep]).any()
    folds = [int(arg)]
    trm = ~np.isin(kf, folds) & keep; tem = np.isin(kf, folds)
    if kind == "iphone": trm &= DEV_K == "iphone"
    elif kind == "android": trm &= DEV_K == "android"
    elif kind == "iphoneM": trm &= (DEV_K == "iphone") & BOTH_K
    else: raise ValueError(kind)
    tr, te = np.flatnonzero(trm), np.flatnonzero(tem); t = time.time()
    Pp = np.full((len(yk), 3), np.nan, np.float32); Pp[te] = run_fit(tr, te)
    np.save(fn, Pp)
    print(f"{job}: train {len(tr)} test {len(te)} knock acc {(Pp[te].argmax(1) == yk[te]).mean():.3f} {time.time()-t:.0f}s", flush=True)
print("DONE", flush=True)
