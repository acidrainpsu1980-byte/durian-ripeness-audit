"""Inner base-model fits for nested stacking (knocking-sound CNN).

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
    return np.concatenate(P)


for a, b in itertools.combinations(range(5), 2):
    out = f"handoff/cnn_pair_{a}{b}.npy"
    if os.path.exists(out): print("skip", out, flush=True); continue
    t = time.time()
    tr = np.flatnonzero(~np.isin(kf_, [a, b])); te = np.flatnonzero(np.isin(kf_, [a, b]))
    P = np.full((len(yk), 3), np.nan, np.float32); P[te] = run_fit(tr, te)
    np.save(out, P)
    acc = (P[te].argmax(1) == yk[te]).mean()
    print(f"pair {a}{b}: train {len(tr)} knocks, test {len(te)}, knock acc {acc:.3f}, {time.time()-t:.0f}s", flush=True)
print("DONE")
