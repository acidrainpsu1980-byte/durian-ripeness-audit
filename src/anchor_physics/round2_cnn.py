"""Round 2 (docs/ANALYSIS_PLAN_round2.md), knock-CNN part: AdaBN session probabilities for every network.

3-class networks (C1 base): outer = round-1 M0 LOBO networks (results/v21/cnn/M0_weights), inner = single-batch networks
(results/round2/cnn/tri_bt_weights); AdaBN on the test batch's anchor knocks.
2-class Ripe/Overripe networks (C3 base): outer = results/round2/cnn/bin_lobo_weights, inner = bin_bt_weights; AdaBN on the test
batch's unlabeled endpoint knocks (verified knocks of non-intake recordings). Labels are never used.
Writes results/round2/cnn_sessions.parquet: session_id, p_Unripe, p_Ripe, p_Overripe, base (tri|bin), train (lobo|<batch>), test.
usage: cd work/raw/sound && python <repo>/src/anchor_physics/round2_cnn.py <repo>
"""
import os, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
REPO = os.path.abspath(sys.argv[1]); os.environ.setdefault("ANCHOR_OUT", f"{REPO}/results/v21")
import cnn_adapt as ca                                  # Net, adabn, to_sessions, knock table, keep and anchor masks
R2 = f"{REPO}/results/round2"; B = ca.BATCHES


def load(path):
    ck = torch.load(path, map_location="cpu"); nc = ck["state_dict"]["fc.weight"].shape[0]
    net = ca.Net(nc); net.load_state_dict(ck["state_dict"]); net.eval(); return net, ck["mel_mean"], ck["mel_std"], nc


def predict(net, idx, mu, sd, nc):
    P = ca.predict(net, idx, mu, sd)
    return np.concatenate([np.zeros((len(P), 1), P.dtype), P], 1) if nc == 2 else P


rows = []; ctrl = None
for base in ["tri", "bin"]:
    for test in B:
        trains = ["lobo"] + [b for b in B if b != test]
        for train in trains:
            if base == "tri":
                path = f"{ca.OUT}/cnn/M0_weights/cnn_common_lobo_{test}.pt" if train == "lobo" else f"{R2}/cnn/tri_bt_weights/cnn_common_bt_{train}2{test}.pt"
                ref = ca.anc & (ca.batch_k == test)
            else:
                path = f"{R2}/cnn/bin_lobo_weights/cnn_common_lobo_{test}.pt" if train == "lobo" else f"{R2}/cnn/bin_bt_weights/cnn_common_bt_{train}2{test}.pt"
                ref = ca.keep & ~ca.anc & (ca.batch_k == test)
            net, mu, sd, nc = load(path); assert nc == (3 if base == "tri" else 2)
            te = np.flatnonzero(ca.batch_k == test); ref = np.flatnonzero(ref)
            P = np.full((len(ca.knock), 3), np.nan, np.float32); P[te] = predict(ca.adabn(net, ref, mu, sd), te, mu, sd, nc)
            _, s = ca.to_sessions(P, "r2"); s = s.drop(columns=["model", "variant"]); s["base"] = base; s["train"] = train; s["test"] = test
            rows.append(s); print(base, train, "->", test, "ref knocks", len(ref), flush=True)
S = pd.concat(rows, ignore_index=True)
# control: tri/lobo = round-1 M5 session probabilities
M5 = pd.read_parquet(f"{ca.OUT}/lobo_cnn_sessions.parquet"); M5 = M5[M5.variant == "M5"].set_index("session_id")[ca.PC]
t = S[(S.base == "tri") & (S.train == "lobo")].set_index("session_id")[ca.PC]
d = float(np.abs(t.values - M5.loc[t.index].values).max()); print("control C1 outer vs round-1 M5 max|diff|", d); assert d < 1e-6
S.to_parquet(f"{R2}/cnn_sessions.parquet", index=False)
