"""CNN leave-one-batch-out and cross-device runs. Usage: python cnn_extra.py <lobo|crossdevice> [epochs]"""
import sys, json, numpy as np, pandas as pd
mode = sys.argv[1]; EP = sys.argv[2] if len(sys.argv) > 2 else "20"
sys.argv = ["cnn_train.py", "fold_fruit", EP]
import os as _os
src = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "cnn_train.py")).read().split("oof = np.zeros")[0]
exec(src)
torch.set_num_threads(3)
dev_k = knock.device.values; batch_k = knock.Maturity_Stage.values
file_k = knock.file.values

def rec_probs(te, P):
    tmp = pd.DataFrame(P, columns=["Unripe", "Ripe", "Overripe"]); tmp["file"] = file_k[te]
    return tmp.groupby("file")[["Unripe", "Ripe", "Overripe"]].mean()

out = {}
if mode == "lobo":
    for b in sorted(set(batch_k)):
        te = np.flatnonzero(batch_k == b); tr = np.flatnonzero(batch_k != b)
        print(f"LOBO {b}: train {len(tr)} test {len(te)}", flush=True)
        P = run_fold(tr, te, [])
        out[b] = rec_probs(te, P).reset_index().to_dict(orient="list")
    json.dump(out, open("handoff/cnn_lobo_recprobs.json", "w"))
else:
    for trd, ted in [("iphone", "android"), ("android", "iphone")]:
        for k in range(5):
            te = np.flatnonzero((kf_ == k) & (dev_k == ted)); tr = np.flatnonzero((kf_ != k) & (dev_k == trd))
            print(f"XDEV {trd}->{ted} fold {k}: train {len(tr)} test {len(te)}", flush=True)
            P = run_fold(tr, te, [])
            out[f"{trd}->{ted}|{k}"] = rec_probs(te, P).reset_index().to_dict(orient="list")
    json.dump(out, open("handoff/cnn_crossdevice_recprobs.json", "w"))
print("DONE")
