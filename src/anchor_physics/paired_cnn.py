"""Round 4 (docs/ANALYSIS_PLAN_round4.md): fruit-specific baseline knock CNN, Ripe vs Overripe on endpoint knocks.

Shared encoder E = the paper's knock-CNN trunk (4 conv blocks, global mean pooling, 96-d); linear 2-class head on
  PAIRED_HEAD=current  h(z_t) | baseline  h(z_0) | concat  h([z_t, z_0]) | paired  h([z_t, z_t - z_0]) | change  h(z_t - z_0)
  PAIRED_SHUFFLE=1 (control K6): every fruit's baseline is replaced, in training and at test time, by the intake knocks of a donor
  fruit of the same harvest batch and the same primary fruit fold (random derangement among fruits with a baseline, numpy
  default_rng(42)); donors never cross the train/test boundary of either design.
Training: endpoint knocks (non-intake sessions labeled Ripe/Overripe, common label) of the training fruits that have a baseline; each
paired with one random intake knock of the same fruit per mini-batch; one forward pass (shared BatchNorm statistics). Schedule as in
src/common_label/cnn_common.py (20 epochs, batch 128, AdamW 2e-3, one-cycle 3e-3, label smoothing 0.05, augmentation, seed 42).
Test: z_0 = mean embedding (eval mode) of all baseline knocks of the fruit; knock probabilities of every endpoint knock of the test
fruits. Output columns (Unripe=0, Ripe, Overripe), NaN outside the test knocks.
Jobs: lobo:<batch> (train on the other two batches) | outer:<k> (primary fruit partition fold k).
usage: cd work/raw/sound && PAIRED_HEAD=paired python <repo>/src/anchor_physics/paired_cnn.py <repo> <OUTDIR> job [job ...]
"""
import os, sys, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("CNN_THREADS", "3")))
REPO = os.path.abspath(sys.argv[1]); OUTDIR = os.path.abspath(sys.argv[2]); JOBS = sys.argv[3:]; os.makedirs(OUTDIR, exist_ok=True)
HEAD = os.environ["PAIRED_HEAD"]; assert HEAD in ("current", "baseline", "concat", "paired", "change")
SHUFFLE = os.environ.get("PAIRED_SHUFFLE", "") == "1"
EPOCHS = 20

MELS = np.load("handoff/mels.npy"); knock = pd.read_parquet("features_sound_knocks.parquet"); assert len(knock) == len(MELS)
rec2 = pd.read_parquet("handoff/rec2_meta.parquet")
kf = knock.file.map(rec2.set_index("file")["fold_fruit"]).values.astype(int)
_fy = pd.read_csv("handoff/file_label_common.csv").set_index("file").iloc[:, 0]; _new = knock.file.map(_fy)
yk = np.where(_new.notna(), _new.fillna(-1).values, knock.y.values).astype(int)          # common label, as cnn_common.py
EXCL = set(open("handoff/exclude_recordings.txt").read().split()); keep = ~knock.file.isin(EXCL).values
man = pd.read_csv(f"{REPO}/data/sound_recording_manifest.csv").set_index("file")
sess = knock.file.map(man.session_id); assert sess.notna()[keep].all()
fruit = sess.str.split("_").str[0].astype(float).values
A = pd.read_csv(f"{REPO}/results/v21/anchor_sessions.csv"); anc_sess = set(A.session_id[A.anchor])
is_base = keep & sess.isin(anc_sess).values                                   # intake (baseline) knocks, label-free rule
fr_with_base = set(fruit[is_base])
is_ep = keep & ~sess.isin(anc_sess).values & np.isin(fruit, list(fr_with_base))  # endpoint knocks with a baseline
batch = knock.Maturity_Stage.values
DONOR = {f: f for f in fr_with_base}
if SHUFFLE:                                              # fixed derangement within (harvest batch, primary fruit fold)
    _r = np.random.default_rng(42)
    _g = pd.DataFrame({'f': fruit[is_base], 'b': batch[is_base], 'k': kf[is_base]}).drop_duplicates('f').sort_values('f')
    for (_b, _k), _grp in _g.groupby(['b', 'k']):
        _f = _grp.f.values
        if len(_f) < 2: continue
        while True:
            _p = _r.permutation(_f)
            if not (_p == _f).any(): break
        DONOR.update(dict(zip(_f, _p)))
    assert sum(DONOR[f] != f for f in DONOR) >= len(DONOR) - 6


class Enc(nn.Module):                                                          # trunk of cnn_common.Net
    def __init__(s):
        super().__init__(); ch = [1, 16, 32, 64, 96]
        s.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(),
                                                nn.Conv2d(ch[i+1], ch[i+1], 3, padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(),
                                                nn.MaxPool2d(2)) for i in range(4)])

    def forward(s, x):
        for b in s.blocks: x = b(x)
        return x.mean((2, 3))


class Model(nn.Module):
    def __init__(s):
        super().__init__(); s.enc = Enc(); s.drop = nn.Dropout(0.3); s.fc = nn.Linear(96 if HEAD in ("current", "baseline", "change") else 192, 2)

    def head(s, zt, z0):
        f = {"current": zt, "baseline": z0, "concat": torch.cat([zt, z0], 1), "paired": torch.cat([zt, zt - z0], 1), "change": zt - z0}[HEAD]
        return s.fc(s.drop(f))


def augment(xb):                                                               # as cnn_common.augment
    sh = np.random.randint(-3, 4); xb = torch.roll(xb, sh, dims=3)
    f0 = np.random.randint(0, 56); w = np.random.randint(0, 9)
    xb = xb.clone(); xb[:, :, f0:f0+w, :] = 0
    return xb


def run(tr_fruit_mask, te_fruit_mask, job):
    torch.manual_seed(42); np.random.seed(42); rng = np.random.default_rng(42)
    tr = np.flatnonzero(is_ep & tr_fruit_mask & np.isin(yk, [1, 2])); te = np.flatnonzero(is_ep & te_fruit_mask)
    base_tr = np.flatnonzero(is_base & tr_fruit_mask)
    pool = pd.Series(base_tr).groupby(fruit[base_tr]).apply(np.array).to_dict()
    assert all(DONOR[f] in pool for f in np.unique(fruit[tr]))
    ref = np.concatenate([tr, base_tr]); mu, sd = MELS[ref].mean(), MELS[ref].std()
    X = lambda idx: torch.tensor((MELS[idx] - mu) / sd).unsqueeze(1)
    ytr = torch.tensor(yk[tr] - 1); cw = torch.tensor(len(tr) / (2 * np.bincount(yk[tr] - 1, minlength=2)), dtype=torch.float32)
    net = Model(); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * ((len(tr) + 127) // 128))
    for ep in range(EPOCHS):
        net.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 128):
            idx = perm[i:i+128].numpy(); kt = tr[idx]
            don = np.array([DONOR[f] for f in fruit[kt]])
            k0 = np.array([rng.choice(pool[g]) for g in don])                   # same-fruit (K6: donor) intake knock
            assert (fruit[k0] == don).all() and is_base[k0].all()
            xb = augment(torch.cat([X(kt), X(k0)], 0)); z = net.enc(xb); n = len(kt)
            loss = F.cross_entropy(net.head(z[:n], z[n:]), ytr[idx], weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); P = np.full((len(knock), 3), np.nan, np.float32)
    with torch.no_grad():
        base_te = np.flatnonzero(is_base & te_fruit_mask)
        Z0 = {}
        for f in np.unique(fruit[te]):
            b = base_te[fruit[base_te] == DONOR[f]]; Z0[f] = torch.cat([net.enc(X(b[j:j+512])) for j in range(0, len(b), 512)]).mean(0)
        for j in range(0, len(te), 512):
            k = te[j:j+512]; zt = net.enc(X(k)); z0 = torch.stack([Z0[f] for f in fruit[k]])
            p = F.softmax(net.head(zt, z0), 1).numpy(); P[k, 0] = 0.0; P[k, 1:] = p
    if os.environ.get("PAIRED_SAVE_WEIGHTS"):
        torch.save({"state_dict": net.state_dict(), "mel_mean": float(mu), "mel_std": float(sd), "head": HEAD, "job": job}, f"{OUTDIR}/paired_{job}.pt")
    return P, len(tr), len(te), len(base_tr)


for job in JOBS:
    kind, arg = job.split(":"); name = f"{kind}_{arg}"; out = f"{OUTDIR}/paired_{name}.npy"
    if os.path.exists(out): print("skip", out, flush=True); continue
    t = time.time()
    if kind == "lobo": trm, tem = batch != arg, batch == arg
    elif kind == "outer": trm, tem = kf != int(arg), kf == int(arg)
    else: raise ValueError(job)
    P, ntr, nte, nb = run(trm, tem, name); np.save(out, P)
    print(f"{HEAD} {job}: train {ntr} endpoint knocks ({nb} baseline knocks), test {nte}, {time.time() - t:.0f}s", flush=True)
print("DONE", flush=True)
