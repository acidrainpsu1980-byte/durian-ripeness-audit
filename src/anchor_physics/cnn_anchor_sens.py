"""Anchor sensitivity for the knock CNN M1 (docs/ANALYSIS_PLAN_anchor_physics.md §6; draws from OUT/anchor_draws.csv).
The M1 LOBO networks (cnn_common.py CNN_NORM=anchor, weights saved to OUT/cnn/M1_weights) were trained on the source batches only,
so a draw changes only the per-mel-band standardization of the held-out batch (statistics from the anchor knocks of the drawn fruits).
Fraction 1.0 must reproduce OUT/cnn/M1/*.npy. Writes OUT/anchor_sens_cnn.parquet (session probabilities).
usage: cd work/raw/sound && python <repo>/src/anchor_physics/cnn_anchor_sens.py <repo>
"""
import os, sys
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.argv = sys.argv[:2]
import cnn_adapt as ca                      # Net, predict, knock table, keep mask, anchor mask, to_sessions (no side effects on import)

OUT = ca.OUT; PC = ca.PC
man = pd.read_csv(f"{ca.REPO}/data/sound_recording_manifest.csv").set_index("file")
fruit_k = ca.knock.file.map(man.session_id).str.split("_").str[0].astype(float).values
D = pd.read_csv(f"{OUT}/anchor_draws.csv")
RAW = ca.MELS.astype(np.float32, copy=True)


def norm_batch(b, ref_mask):
    r = ca.batch_k == b; m = RAW[r & ref_mask]
    mu = m.mean(axis=(0, 2), keepdims=True); sd = m.std(axis=(0, 2), keepdims=True); sd[sd == 0] = 1.0
    return (RAW[r] - mu) / sd


rows = []; ctrl = 0.0
for b in ca.BATCHES:
    ck = torch.load(f"{OUT}/cnn/M1_weights/cnn_common_lobo_{b}.pt", map_location="cpu")
    net = ca.Net(); net.load_state_dict(ck["state_dict"]); net.eval()
    te = np.flatnonzero(ca.batch_k == b)
    cfgs = [(1.0, 0, None)] + [(f, k, set(D[(D.batch == b) & (D.fraction == f) & (D.draw == k)].fruit_id)) for f in [0.1, 0.2, 0.4] for k in range(5)]
    for f, k, fr in cfgs:
        ref = ca.anc if fr is None else (ca.anc & np.isin(fruit_k, list(fr)))
        Z = norm_batch(b, ref)
        ca.MELS = np.zeros_like(RAW); ca.MELS[te] = Z            # predict() reads ca.MELS
        P = np.full((len(RAW), 3), np.nan, np.float32); P[te] = ca.predict(net, te, ck["mel_mean"], ck["mel_std"])
        if fr is None:
            Q = np.load(f"{OUT}/cnn/M1/cnn_common_lobo_{b}.npy"); ctrl = max(ctrl, float(np.abs(P[te] - Q[te]).max()))
        _, s = ca.to_sessions(P, "M1"); s["fraction"] = f; s["draw"] = k; s["batch"] = b; rows.append(s)
    print(b, "done", flush=True)
print("control fraction 1.0 vs cnn/M1 max|diff|", ctrl); assert ctrl < 1e-5
pd.concat(rows, ignore_index=True).to_parquet(f"{OUT}/anchor_sens_cnn.parquet", index=False)
