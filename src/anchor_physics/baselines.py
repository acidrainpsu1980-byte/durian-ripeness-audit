"""Pre-registered analysis (docs/ANALYSIS_PLAN_anchor_physics.md §2), secondary physics baselines under LOBO, in the same LR as the
physics-guided model (lobo_classical.lobo; variants M0-M4):
  phys_weight  harvest weight alone (kg; recorded at harvest, before storage)
  phys_fdom    dominant frequency alone
  phys_pcm     peak frequency + spectral centroid + mean MFCC 1-13 (recording means of the paper's sound features)
Writes OUT/lobo_baselines_sessions.parquet.  usage (repo root): python src/anchor_physics/baselines.py OUT
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import lobo_classical as lc

PCM = ["dom_freq__mean", "centroid__mean"] + [f"mfcc{i:02d}_mean__mean" for i in range(1, 14)]
_orig_load = lc.load


def load(mod):
    d, _, fam = _orig_load("physics")
    if mod == "phys_weight":
        p = pd.read_parquet(f"{lc.OUT}/physics_recordings.parquet").set_index("file"); d["harvest_weight_kg"] = d.file.map(p.harvest_weight_kg).astype(float)
        cols = ["harvest_weight_kg"]
    elif mod == "phys_fdom":
        cols = ["f_dom"]
    elif mod == "phys_pcm":
        s = pd.read_parquet(lc.CFG["sound"]["features"]).set_index("file")
        for c in PCM: d[c] = d.file.map(s[c]).astype(float)
        cols = PCM
    assert d[cols].notna().all().all(), mod
    return d, cols, fam


if __name__ == "__main__":
    OUT = sys.argv[1]; lc.OUT = OUT; lc.load = load
    A = pd.read_csv(f"{OUT}/anchor_sessions.csv"); anchor_sessions = set(A.session_id[A.anchor])
    R = pd.concat([lc.lobo(m, anchor_sessions) for m in ["phys_weight", "phys_fdom", "phys_pcm"]], ignore_index=True)
    R.to_parquet(f"{OUT}/lobo_baselines_sessions.parquet", index=False); print(R.groupby(["model", "variant"]).size().to_dict())
