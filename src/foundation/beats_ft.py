"""Round 5 (docs/ANALYSIS_PLAN_round5.md, tag prereg-round5), arm A1ft: BEATs iter3+ AS2M (self-supervised) LP-FT on knocks.

Stage 1 (LP): linear 2-class head (Ripe, Overripe) on the frozen masked-mean BEATs embedding, 20 epochs.
Stage 2 (FT): top 2 of 12 transformer layers + head unfrozen, 8 epochs; AdamW (encoder lr 2e-5, head lr 1e-3, weight decay 0.01),
batch 64 knocks, class-balanced cross-entropy, no early stopping, no augmentation; the model stays in eval mode (dropout off)
for determinism. Input: the W150 knock window at 16 kHz re-cut from the original-rate recording, zero-padded to 1.0 s with
padding_mask (identical to the frozen extraction). Seeds 0, 1, 2; the seed-averaged probabilities are the arm's output.
Training knocks: verified endpoint knocks (non-intake sessions labeled Ripe/Overripe, common label) of the training fruits that have
a baseline, exactly as src/anchor_physics/paired_cnn.py. No held-out knock enters any gradient step (asserted).
Jobs: lobo:<batch> | outer:<k>. Output: beats_ft_<kind>_<arg>.npy, (n_knocks, 3) columns (Unripe=NaN, Ripe, Overripe), NaN outside
the test knocks; row order = features_sound_knocks.parquet.
usage (GPU host): R5_EXTRACT=<path to extract_r5.py> python beats_ft.py <repo> <OUTDIR> job [job ...]
Prints only sizes and run times.
"""
import os, sys, time, importlib.util
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

REPO = os.path.abspath(sys.argv[1]); OUTDIR = os.path.abspath(sys.argv[2]); JOBS = sys.argv[3:]; os.makedirs(OUTDIR, exist_ok=True)
spec = importlib.util.spec_from_file_location("extract_r5", os.environ["R5_EXTRACT"]); X5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X5)
SEEDS = [0, 1, 2]; LP_EPOCHS = 20; FT_EPOCHS = 8; BS = 64; LR_ENC = 2e-5; LR_HEAD = 1e-3; WD = 0.01; N_TOP = 2; PAD = 16000

# ---------------------------------------------------------------- knocks, labels, splits (as paired_cnn.py)
knock = pd.read_parquet(f"{REPO}/work/raw/sound/features_sound_knocks.parquet")
rec2 = pd.read_parquet(f"{REPO}/work/raw/sound/rec2_meta.parquet")
kf = knock.file.map(rec2.set_index("file")["fold_fruit"]).values.astype(int)
_fy = pd.read_csv(f"{REPO}/work/raw/sound/file_label_common.csv").set_index("file").iloc[:, 0]; _new = knock.file.map(_fy)
yk = np.where(_new.notna(), _new.fillna(-1).values, knock.y.values).astype(int)
EXCL = set(open(f"{REPO}/work/raw/sound/exclude_recordings.txt").read().split()); keep = ~knock.file.isin(EXCL).values
man = pd.read_csv(f"{REPO}/data/sound_recording_manifest.csv").set_index("file")
sess = knock.file.map(man.session_id); assert sess.notna()[keep].all()
fruit = sess.str.split("_").str[0].astype(float).values
A = pd.read_csv(f"{REPO}/results/v21/anchor_sessions.csv"); anc_sess = set(A.session_id[A.anchor])
is_base = keep & sess.isin(anc_sess).values
fr_with_base = set(fruit[is_base])
is_ep = keep & ~sess.isin(anc_sess).values & np.isin(fruit, list(fr_with_base))
batch = knock.Maturity_Stage.values

# ---------------------------------------------------------------- waveforms (16 kHz, W150) from the frozen-extraction loader
t0 = time.time()
XW, idx, fr, NV = X5.load_audio_segments(None)
assert (idx.file.values == knock.file.values).all() and (idx.knock_idx.values == knock.knock_idx.values).all()
W = XW[(16000, "W150")]; nv = NV[(16000, "W150")]
Wp = np.zeros((len(W), PAD), np.float32); Wp[:, :W.shape[1]] = W
print(f"waveforms {Wp.shape} in {time.time()-t0:.0f}s", flush=True)


def build():
    X5.setup_torch(); E = X5.build_beats(); return E["fn"]._model


def emb(m, x, n):
    pm = torch.arange(x.shape[1], device=x.device)[None, :] >= n[:, None]
    rep, tm = m.extract_features(x, padding_mask=pm)
    k = (~tm).float() if tm is not None else torch.ones(rep.shape[:2], device=x.device)
    return (rep * k[..., None]).sum(1) / k.sum(1, keepdim=True).clamp_min(1.0)


def run(kind, arg, seed):
    if kind == "lobo": tr_f = batch != arg; te_f = batch == arg
    else: k = int(arg); tr_f = kf != k; te_f = kf == k
    tr = np.flatnonzero(is_ep & tr_f & np.isin(yk, [1, 2])); te = np.flatnonzero(is_ep & te_f)
    assert not set(fruit[tr]) & set(fruit[te])
    torch.manual_seed(seed); np.random.seed(seed); g = torch.Generator().manual_seed(seed)
    m = build(); dev = next(m.parameters()).device
    for p in m.parameters(): p.requires_grad_(False)
    m.eval(); head = nn.Linear(768, 2).to(dev)
    ytr = torch.as_tensor(yk[tr] - 1, device=dev)
    cw = torch.as_tensor(len(tr) / (2 * np.bincount(yk[tr] - 1, minlength=2)), dtype=torch.float32, device=dev)
    xs = lambda ii: (torch.from_numpy(Wp[ii]).to(dev), torch.as_tensor(nv[ii], device=dev))
    with torch.no_grad():                                                    # stage 1 on frozen embeddings
        Z = torch.cat([emb(m, *xs(tr[s:s + BS])) for s in range(0, len(tr), BS)])
    opt = torch.optim.AdamW(head.parameters(), lr=LR_HEAD, weight_decay=WD)
    for ep in range(LP_EPOCHS):
        for b in torch.randperm(len(tr), generator=g).split(BS):
            loss = F.cross_entropy(head(Z[b.to(dev)]), ytr[b.to(dev)], weight=cw); opt.zero_grad(); loss.backward(); opt.step()
    top = list(m.encoder.layers[-N_TOP:].parameters())
    for p in top: p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": top, "lr": LR_ENC}, {"params": head.parameters(), "lr": LR_HEAD}], weight_decay=WD)
    for ep in range(FT_EPOCHS):                                              # stage 2
        for b in torch.randperm(len(tr), generator=g).split(BS):
            ii = tr[b.numpy()]; loss = F.cross_entropy(head(emb(m, *xs(ii))), ytr[b.to(dev)], weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
    P = np.full((len(knock), 3), np.nan)
    with torch.no_grad():
        for s in range(0, len(te), BS):
            ii = te[s:s + BS]; P[ii, 1:] = F.softmax(head(emb(m, *xs(ii))), 1).cpu().numpy()
    del m; torch.cuda.empty_cache()
    return P, len(tr), len(te)


for job in JOBS:
    kind, arg = job.split(":"); t1 = time.time(); Ps = []
    for sd in SEEDS:
        P, ntr, nte = run(kind, arg, sd); Ps.append(P)
    P = np.mean(Ps, 0); np.save(f"{OUTDIR}/beats_ft_{kind}_{arg}.npy", P)
    print(f"A1ft {job}: train {ntr} knocks, test {nte}, {len(SEEDS)} seeds, {time.time()-t1:.0f}s", flush=True)
