"""v1.8: export every out-of-fold prediction behind the principal tables in long, documented form.

usage (repo root): DURIAN_INTERMEDIATE=intermediate_common python src/audit/export_predictions.py      (~5 min, CPU)
reads:  intermediate_v14/fits/<P>/<sensor>__<candidate>.npz, intermediate_v14/cnn*/, results/session_table.csv
writes: audit/predictions/capture_oof.parquet        capture / recording-level outer-fold probabilities, every base candidate,
                                                      every partition (p0cpu = primary partition with the CPU knock CNN; p0-p9 GPU CNN)
        audit/predictions/session_oof.parquet        session-level outer-fold probabilities on the 268 evaluation sessions for every
                                                      base candidate ("<sensor>:<candidate>") and every configuration / procedure
                                                      produced by nsel.evaluate (fixed families, nested family / configuration
                                                      selection, mean / product fusions, nested stacking)
        audit/predictions/session_oof_primary.csv    the same, primary partition only, wide and human-readable
        audit/predictions/selection_choices.csv      every inner score and every per-fold choice of the selection procedures
Check: fold-mean accuracy / macro-F1 recomputed from session_oof.parquet must equal results/v14/partition_per_fold.csv exactly.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(REPO); sys.path.insert(0, os.path.join(REPO, "src", "nested_selection"))
os.environ.setdefault("DURIAN_INTERMEDIATE", "intermediate_common")
os.environ.setdefault("EMB_MOBILENET", "intermediate_common/rgb/emb_mobilenetv2.npy")
import cands, nsel

IV = "intermediate_v14"; OUT = os.environ.get("DURIAN_AUDIT_OUT", "audit") + "/predictions"; os.makedirs(OUT, exist_ok=True)
st = pd.read_csv(os.environ.get("DURIAN_RESULTS", "results") + "/session_table.csv", low_memory=False)
ST = st[st.in_main_analysis.astype(str) == "True"][["session_id", "fruit_id", "y"]].reset_index(drop=True)
ST["fruit_id"] = ST.fruit_id.astype(int); y = ST.y.values.astype(int); sid = ST.session_id.astype(str).values
recf = cands.load("sound")[0].file.values; KN = f"{IV}/features_sound_knocks.parquet"
CL = nsel.CL; LABELS = np.array(["Unripe", "Ripe", "Overripe"])
PARTS = [("p0cpu", "p0", f"{IV}/cnn_cpu_p0", "cnn_common_{kind}_{arg}.npy")] + [(f"p{i}", f"p{i}", f"{IV}/cnn", "cnn_{P}_{kind}_{arg}.npy") for i in range(10)]
cap_rows, ses_rows, choices = [], [], []
for name, P, cdir, cfmt in PARTS:
    pr = nsel.Preds(P, f"{IV}/fits", cdir, cfmt, KN, recf)
    o, ch, fold = nsel.evaluate(pr, ST); choices.append(ch.assign(partition=name))
    for mod in nsel.MODS:
        m = pr.capture_meta(mod); fm = m.fruit_id.map(nsel.PARTS[P]).values.astype(int)
        for c in cands.CANDIDATES[mod]:
            a = pr._arrays(mod, c)["outer"]
            d = pd.DataFrame(a, columns=CL); d.insert(0, "partition", name); d.insert(1, "sensor", mod); d.insert(2, "candidate", c)
            d.insert(3, "unit_id", m.file.values); d.insert(4, "session_id", m.session_id.astype(str).values)
            d.insert(5, "fruit_id", m.fruit_id.values); d.insert(6, "outer_fold", fm); d.insert(7, "y", m.y.values.astype(int))
            cap_rows.append(d)
            s = pr.session(mod, c, "outer").reindex(sid)
            o[f"{mod}:{c}"] = s.values
    for k, v in o.items():
        d = pd.DataFrame(v, columns=CL); d.insert(0, "partition", name); d.insert(1, "configuration", k)
        d.insert(2, "session_id", sid); d.insert(3, "fruit_id", ST.fruit_id.values); d.insert(4, "outer_fold", fold); d.insert(5, "y", y)
        ses_rows.append(d)
    print(name, "done", flush=True)
CAP = pd.concat(cap_rows, ignore_index=True); SES = pd.concat(ses_rows, ignore_index=True)
for D_ in (CAP, SES):
    ok = D_[CL].notna().all(1)
    D_["pred"] = pd.Series(pd.NA, index=D_.index, dtype="object"); D_.loc[ok, "pred"] = LABELS[D_.loc[ok, CL].values.argmax(1)]
    D_["label"] = LABELS[D_.y.values]
CAP.to_parquet(f"{OUT}/capture_oof.parquet", index=False); SES.to_parquet(f"{OUT}/session_oof.parquet", index=False)
SES[SES.partition == "p0cpu"].to_csv(f"{OUT}/session_oof_primary.csv", index=False, float_format="%.6f")
pd.concat(choices, ignore_index=True).to_csv(f"{OUT}/selection_choices.csv", index=False)
# ---- check against the shipped per-fold table (Table 3 inputs)
PF = pd.read_csv(os.environ.get("DURIAN_RESULTS", "results") + "/v14/partition_per_fold.csv")
mx = 0.0
for r in PF.itertuples():
    t = SES[(SES.partition == r.partition) & (SES.configuration == r.config) & (SES.outer_fold == r.fold)]
    pred = t[CL].values.argmax(1)
    mx = max(mx, abs(accuracy_score(t.y, pred) - r.accuracy), abs(f1_score(t.y, pred, average="macro") - r.macro_f1))
print(f"capture rows {len(CAP)} | session rows {len(SES)} | configurations {SES.configuration.nunique()} | "
      f"max |diff| vs results/v14/partition_per_fold.csv over {len(PF)} rows = {mx}")
assert mx < 1e-12
