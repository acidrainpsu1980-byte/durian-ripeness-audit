"""Small 2D-CNN on per-knock log-mel (64x31). Usage: python cnn_train.py <split> [epochs]
Writes handoff/cnn_oof_knock__<split>.npy (n_knocks x 3) and handoff/cnn_log__<split>.json"""
import sys, os, json, time
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
split = sys.argv[1]; EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MELS = np.load("handoff/mels.npy")                       # (N,64,31)
knock = pd.read_parquet("features_sound_knocks.parquet")
rec2 = pd.read_parquet("handoff/rec2_meta.parquet")
assert len(knock) == len(MELS)
fold_of_file = rec2.set_index("file")[split]
kf_ = knock.file.map(fold_of_file).values.astype(int)
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
        x = x.mean((2, 3))
        return s.fc(s.drop(x))


def augment(xb):
    # random time roll (+-3 frames) and one frequency mask (<=8 bands)
    B = xb.shape[0]
    sh = np.random.randint(-3, 4)
    xb = torch.roll(xb, sh, dims=3)
    f0 = np.random.randint(0, 56); w = np.random.randint(0, 9)
    xb = xb.clone(); xb[:, :, f0:f0+w, :] = 0
    return xb


def run_fold(tr, te, log):
    mu, sd = MELS[tr].mean(), MELS[tr].std()
    Xtr = torch.tensor((MELS[tr] - mu) / sd).unsqueeze(1); ytr = torch.tensor(yk[tr])
    Xte = torch.tensor((MELS[te] - mu) / sd).unsqueeze(1)
    cw = torch.tensor(len(ytr) / (3 * np.bincount(yk[tr], minlength=3)), dtype=torch.float32)
    net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * ((len(tr) + 127) // 128))
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(tr)); tl = 0; t0 = time.time()
        for i in range(0, len(tr), 128):
            idx = perm[i:i+128]; xb = augment(Xtr[idx]); yb = ytr[idx]
            loss = F.cross_entropy(net(xb), yb, weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(idx)
        log.append(dict(epoch=ep, train_loss=tl / len(tr), sec=time.time() - t0)); print(log[-1], flush=True)
    net.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(te), 512):
            P.append(F.softmax(net(Xte[i:i+512]), 1).numpy())
    return np.concatenate(P)


oof = np.zeros((len(yk), 3), np.float32); log = []
for k in range(5):
    te = np.flatnonzero(kf_ == k); tr = np.flatnonzero(kf_ != k)
    print(f"fold {k}: train {len(tr)} test {len(te)}", flush=True)
    flog = []; oof[te] = run_fold(tr, te, flog); log.append(dict(fold=k, epochs=flog))
    acc = (oof[te].argmax(1) == yk[te]).mean(); print(f"fold {k} knock acc {acc:.3f}", flush=True)
np.save(f"handoff/cnn_oof_knock__{split}.npy", oof)
json.dump(log, open(f"handoff/cnn_log__{split}.json", "w"))
print("DONE")
