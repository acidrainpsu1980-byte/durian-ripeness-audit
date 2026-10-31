"""Small CNN on 128x128 fruit-centred z-scored thermal crops. 5-fold fold_fruit CV, 15 epochs, CPU."""
import os, sys, time, json
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
sys.path.append(os.getcwd())
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
import durian_common as dc

SEED = 42; EPOCHS = 15; BS = 32
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)
split = sys.argv[1] if len(sys.argv) > 1 else "fold_fruit"

feat = pd.read_parquet("features_thermal.parquet")
imgs = np.load("thermal_imgs128.npy")            # (N,128,128) z-scored per image
y = feat.y.values.astype(np.int64)
assert len(imgs) == len(feat)
if "device" not in feat.columns:
    feat["device"] = np.nan


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [1, 16, 32, 64, 128]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, x):
        x = self.features(x)
        return self.head(x.mean((2, 3)))


def augment(xb):
    # thermal crops are orientation-free: random flips / 90-degree rotations + small intensity jitter
    if np.random.rand() < 0.5: xb = torch.flip(xb, [3])
    if np.random.rand() < 0.5: xb = torch.flip(xb, [2])
    k = np.random.randint(4); xb = torch.rot90(xb, k, [2, 3])
    xb = xb * (1 + 0.1 * torch.randn(xb.shape[0], 1, 1, 1)) + 0.1 * torch.randn(xb.shape[0], 1, 1, 1)
    return xb


X = torch.tensor(imgs[:, None]).float()
oof = np.zeros((len(feat), 3), dtype=np.float32)
t0 = time.time()
for k, tr, te in dc.iter_folds(feat, split):
    torch.manual_seed(SEED + k); np.random.seed(SEED + k)
    net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * int(np.ceil(len(tr) / BS)))
    cw = torch.tensor(len(tr) / (3 * np.bincount(y[tr], minlength=3)), dtype=torch.float32)
    ytr = torch.tensor(y[tr])
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(len(tr)); tot = 0
        for i in range(0, len(tr), BS):
            idx = perm[i:i + BS]
            xb = augment(X[tr[idx]]); yb = ytr[idx]
            loss = Fn.cross_entropy(net(xb), yb, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(idx)
        print(f"fold {k} ep {ep} loss {tot / len(tr):.3f} {time.time() - t0:.0f}s", flush=True)
    net.eval()
    with torch.no_grad():
        # test-time augmentation over the 8 dihedral transforms
        P = torch.zeros(len(te), 3)
        for flip in [False, True]:
            for r in range(4):
                xb = X[te]
                if flip: xb = torch.flip(xb, [3])
                xb = torch.rot90(xb, r, [2, 3])
                P += Fn.softmax(net(xb), 1)
        oof[te] = (P / 8).numpy()
    m = dc.metrics_from_probs(y[te], oof[te]); print(f"fold {k} capture acc {m['accuracy']:.3f} f1 {m['macro_f1']:.3f}", flush=True)

np.save(f"results/oof_thermal__cnn__{split}.npy", oof)
p1, p2 = dc.save_predictions(feat, oof, split, "thermal", "cnn")
ev = dc.evaluate_oof(feat, oof, split)
print(json.dumps(ev["summary"], indent=1)); print("total", round(time.time() - t0), "s")
