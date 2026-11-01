"""Pre-registered analysis (docs/ANALYSIS_PLAN_anchor_physics.md), step 3: post-hoc adaptation of the M0 LOBO knock CNNs and
session probabilities of every knock-CNN variant.

From the saved M0 networks (results/v21/cnn/M0_weights/cnn_common_lobo_<batch>.pt, trained by cnn_common.py lobo:<batch>):
  M0r  reload control: the saved network re-applied to the held-out knocks (must equal cnn/M0/*.npy)
  M3   anchored logit shift: knock log-probabilities of the held-out batch shifted by mean(source anchor knocks) - mean(target anchor knocks)
  M4   prior-shift EM (Saerens et al. 2002) on the held-out knocks; source prior = class frequencies of the training knocks (common label)
  M5   AdaBN anchored: BatchNorm running statistics re-estimated (cumulative average, no augmentation, dropout off) on the held-out
       batch's anchor knocks
  M6   AdaBN unanchored: the same on all verified knocks of the held-out batch (unlabeled)
M1 and M2 are retrained networks (cnn_common.py with CNN_NORM=anchor|all). Labels are never used by any adaptation step.
Recording probability = mean over its knocks; session probability = mean over the session's verified recordings (as in the paper).
Control: M0 recording probabilities equal intermediate_common/sound/cnn_lobo_recprobs.json (the shipped LOBO CNN).
usage: cd work/raw/sound && python <repo>/src/anchor_physics/cnn_adapt.py <repo>
"""
import os, sys, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F


def saerens_em(P, prior_src, n_iter=100, tol=1e-8):          # identical to lobo_classical.saerens_em
    w = prior_src.copy()
    for _ in range(n_iter):
        Q = P * (w / prior_src); Q /= Q.sum(1, keepdims=True); w_new = Q.mean(0)
        if np.abs(w_new - w).max() < tol: w = w_new; break
        w = w_new
    Q = P * (w / prior_src); return Q / Q.sum(1, keepdims=True)

torch.set_num_threads(int(os.environ.get("CNN_THREADS", "3")))
REPO = os.path.abspath(sys.argv[1]); OUT = os.path.abspath(os.environ.get("ANCHOR_OUT", f"{REPO}/results/v21"))   # ANCHOR_OUT: run_all.sh reproduction
BATCHES = ["Immature", "Mature", "Overmature"]; PC = ["p_Unripe", "p_Ripe", "p_Overripe"]

MELS = np.load("handoff/mels.npy"); knock = pd.read_parquet("features_sound_knocks.parquet"); assert len(knock) == len(MELS)
EXCL = set(open("handoff/exclude_recordings.txt").read().split())
keep = ~knock.file.isin(EXCL).values; batch_k = knock.Maturity_Stage.values
anchor_files = set(open(f"{OUT}/anchor_recordings.txt").read().split())
anc = keep & knock.file.isin(anchor_files).values
_fy = pd.read_csv("handoff/file_label_common.csv").set_index("file").iloc[:, 0]; _new = knock.file.map(_fy)
yk = np.where(_new.notna(), _new.fillna(-1).values, knock.y.values).astype(int)          # as in cnn_common.py


class Net(nn.Module):                      # identical to src/common_label/cnn_common.py
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


def load(b):
    ck = torch.load(f"{OUT}/cnn/M0_weights/cnn_common_lobo_{b}.pt", map_location="cpu")
    net = Net(); net.load_state_dict(ck["state_dict"]); net.eval(); return net, ck["mel_mean"], ck["mel_std"]


def X(idx, mu, sd): return torch.tensor((MELS[idx] - mu) / sd).unsqueeze(1)


def predict(net, idx, mu, sd):
    net.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(idx), 512): P.append(F.softmax(net(X(idx[i:i+512], mu, sd)), 1).numpy())
    return np.concatenate(P)


def adabn(net, idx, mu, sd):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d): m.reset_running_stats(); m.momentum = None; m.train()
    with torch.no_grad():
        for i in range(0, len(idx), 512): net(X(idx[i:i+512], mu, sd))
    net.eval(); return net


def to_sessions(Pk, variant):
    k = pd.DataFrame(Pk, columns=PC); k["file"] = knock.file.values
    k = k[keep & np.isfinite(Pk).all(1)]
    r = k.groupby("file")[PC].mean()
    man = pd.read_csv(f"{REPO}/data/sound_recording_manifest.csv").set_index("file")
    r["session_id"] = r.index.map(man.session_id)
    assert r.session_id.notna().all()
    s = r.groupby("session_id")[PC].mean().reset_index(); s["model"] = "sound_cnn"; s["variant"] = variant
    return r, s


if __name__ == "__main__":
    V = {v: np.full((len(knock), 3), np.nan, np.float32) for v in ["M0r", "M3", "M4", "M5", "M6"]}
    for b in BATCHES:
        te = np.flatnonzero((batch_k == b)); src_anc = np.flatnonzero(anc & (batch_k != b)); tgt_anc = np.flatnonzero(anc & (batch_k == b))
        tgt_all = np.flatnonzero(keep & (batch_k == b))
        net, mu, sd = load(b)
        P0 = predict(net, te, mu, sd); V["M0r"][te] = P0
        Ps = predict(net, src_anc, mu, sd); Pt = predict(net, tgt_anc, mu, sd)
        shift = np.log(Ps + 1e-9).mean(0) - np.log(Pt + 1e-9).mean(0)
        L = np.log(P0 + 1e-9) + shift; L = np.exp(L - L.max(1, keepdims=True)); V["M3"][te] = L / L.sum(1, keepdims=True)
        prior = np.bincount(yk[keep & (batch_k != b)], minlength=3) / (keep & (batch_k != b)).sum()
        V["M4"][te] = saerens_em(P0.astype(float), prior)
        net5, _, _ = load(b); V["M5"][te] = predict(adabn(net5, tgt_anc, mu, sd), te, mu, sd)
        net6, _, _ = load(b); V["M6"][te] = predict(adabn(net6, tgt_all, mu, sd), te, mu, sd)
        print(b, "anchor knocks src/tgt", len(src_anc), len(tgt_anc), "all tgt", len(tgt_all), "shift", np.round(shift, 3), flush=True)
    # retrained variants
    for v in ["M0", "M1", "M2"]:
        P = np.full((len(knock), 3), np.nan, np.float32)
        for b in BATCHES:
            Q = np.load(f"{OUT}/cnn/{v}/cnn_common_lobo_{b}.npy"); m = batch_k == b; assert np.isfinite(Q[m]).all(); P[m] = Q[m]
        V[v] = P
    # controls
    d = np.nanmax(np.abs(V["M0r"] - V["M0"])); print("control reload M0r vs M0 max|diff|", float(d)); assert d < 1e-5
    ref = json.load(open(f"{REPO}/intermediate_common/sound/cnn_lobo_recprobs.json"))
    r0, _ = to_sessions(V["M0"], "M0"); dm = 0.0
    for b, t in ref.items():
        R = pd.DataFrame({c: np.asarray(t[c], float) for c in ["Unripe", "Ripe", "Overripe"]}, index=t["file"])
        dm = max(dm, float(np.abs(r0.loc[R.index, PC].values - R.values).max()))
    print("control M0 vs shipped cnn_lobo_recprobs max|diff|", dm, "recordings", len(r0))
    json.dump({"reload_max_abs_diff": float(d), "shipped_lobo_max_abs_diff": dm, "n_recordings": int(len(r0))},
              open(f"{OUT}/cnn_controls.json", "w"), indent=1)
    S = pd.concat([to_sessions(V[v], v)[1] for v in ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]], ignore_index=True)
    S.to_parquet(f"{OUT}/lobo_cnn_sessions.parquet", index=False); print(S.groupby("variant").size().to_dict())
