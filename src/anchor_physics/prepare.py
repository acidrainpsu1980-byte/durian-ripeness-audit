"""Pre-registered analysis (docs/ANALYSIS_PLAN_anchor_physics.md), step 1: label-free intake anchors and physics-guided knock features.

usage (repo root, DURIAN_DATA_ROOT=<folder with dataset_clean_sound.zip>):
  python src/anchor_physics/prepare.py anchors OUT      # OUT/anchor_sessions.csv, OUT/anchor_recordings.txt
  python src/anchor_physics/prepare.py physics OUT      # OUT/physics_knocks.parquet, OUT/physics_recordings.parquet
Anchor rule (plan §1): the earliest capture session of each fruit, if it lies within 4 days of the fruit's recorded harvest date.
Labels are not read by the anchor rule (they are joined afterwards only to report the composition).
Physics features (plan §2), per knock from the same 150 ms window as the paper: dominant frequency (largest peak, 100-2000 Hz), first
prominent peak (lowest peak with prominence >= 10 dB, 100-2000 Hz), spectral centroid, 20 dB decay time, 1/Q = half-power bandwidth /
f_dom; per recording the median over knocks and the knock count; S = f_dom^2 * m^(2/3) with m = harvest weight (kg). End-of-storage
weight, weight loss and storage time are not read.
"""
import os, sys, zipfile
import numpy as np, pandas as pd
from scipy.signal import find_peaks
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "sound"))

FMIN, FMAX, PROM_DB, NFFT = 100.0, 2000.0, 10.0, 8192


def anchors(out):
    os.makedirs(out, exist_ok=True)
    S = pd.read_csv("audit/sessions.csv")[["session_id", "fruit_id", "date", "days_from_recorded_harvest", "in_evaluation_268"]]
    S["date"] = pd.to_datetime(S.date)
    first = S.groupby("fruit_id").date.transform("min")
    S["anchor"] = (S.date == first) & (S.days_from_recorded_harvest <= 4)
    # composition (labels joined only for reporting)
    lab = pd.read_csv("data/session_labels_common.csv").set_index("session_id").label_common
    S["label_common"] = S.session_id.map(lab)
    S.to_csv(f"{out}/anchor_sessions.csv", index=False)
    man = pd.read_csv("data/sound_recording_manifest.csv")
    files = sorted(man.file[man.session_id.isin(S.session_id[S.anchor])])
    open(f"{out}/anchor_recordings.txt", "w").write("\n".join(files) + "\n")
    E = S[S.in_evaluation_268.astype(bool)]
    print(pd.crosstab(E.anchor, E.label_common)); print("anchor recordings", len(files))


def _knock_physics(seg, sr):
    w = seg * np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(w)); fr = np.fft.rfftfreq(len(seg), 1 / sr)
    dom_full = fr[int(np.argmax(spec))]                         # = dom_freq of the paper (control)
    specz = np.abs(np.fft.rfft(w, n=NFFT)); frz = np.fft.rfftfreq(NFFT, 1 / sr)
    band = (frz >= FMIN) & (frz <= FMAX); sb = specz[band]; fb = frz[band]
    i = int(np.argmax(sb)); f_dom = float(fb[i])
    db = 20 * np.log10(sb + 1e-12)
    pk, pr = find_peaks(db, prominence=PROM_DB)
    f_first = float(fb[pk[0]]) if len(pk) else f_dom
    half = sb[i] / np.sqrt(2)                                     # -3 dB
    lo = i
    while lo > 0 and sb[lo] > half: lo -= 1
    hi = i
    while hi < len(sb) - 1 and sb[hi] > half: hi += 1
    inv_q = float((fb[hi] - fb[lo]) / f_dom)
    return dom_full, f_dom, f_first, inv_q


def physics(out):
    import knock_features as kf
    os.makedirs(out, exist_ok=True)
    K = pd.read_parquet("intermediate_v14/features_sound_knocks.parquet")[["file", "knock_idx", "onset_s", "dom_freq", "centroid", "decay20"]]
    zf = zipfile.ZipFile(kf.ZIP_PATH); members = {os.path.basename(n): n for n in zf.namelist() if not n.startswith("__MACOSX")}
    rows = []
    for j, (fname, g) in enumerate(K.groupby("file", sort=True)):
        b = zf.read(members[fname]); y, sr = kf.decode_bytes(b, os.path.splitext(fname)[1].lower()); y = kf.preprocess(y, sr)
        for _, r in g.iterrows():
            o = int(round(float(r.onset_s) * kf.SR)); seg = y[o:o + kf.WIN]
            dom_full, f_dom, f_first, inv_q = _knock_physics(seg, kf.SR)
            rows.append(dict(file=fname, knock_idx=int(r.knock_idx), dom_freq_check=dom_full, dom_freq_paper=float(r.dom_freq),
                             f_dom=f_dom, f_first=f_first, inv_q=inv_q, centroid=float(r.centroid), decay20=float(r.decay20)))
        if j % 50 == 0: print(j, fname, flush=True)
    P = pd.DataFrame(rows)
    d = np.abs(P.dom_freq_check - P.dom_freq_paper); print("segment control: dom_freq max|diff|", float(d.max()), "share exact", float((d < 1e-6).mean()))
    P.to_parquet(f"{out}/physics_knocks.parquet", index=False)
    R = P.groupby("file").agg(f_dom=("f_dom", "median"), f_first=("f_first", "median"), inv_q=("inv_q", "median"),
                              centroid=("centroid", "median"), decay20=("decay20", "median"), n_knocks=("f_dom", "size")).reset_index()
    log = pd.read_csv("data/zenodo/durian_characteristics_cleaned.csv"); log.insert(0, "fruit_id", np.arange(1, len(log) + 1))
    man = pd.read_csv("data/sound_recording_manifest.csv").set_index("file")
    R["fruit_id"] = R.file.map(man.fruit_id).astype(int); R["device"] = R.file.map(man.device)
    R["harvest_weight_kg"] = R.fruit_id.map(log.set_index("fruit_id").Harvest_Weight).astype(float)
    R["stiffness"] = R.f_dom ** 2 * R.harvest_weight_kg ** (2 / 3)
    assert R.harvest_weight_kg.notna().all()
    R.to_parquet(f"{out}/physics_recordings.parquet", index=False); print(R.describe().T.round(3).to_string())


if __name__ == "__main__":
    {"anchors": anchors, "physics": physics}[sys.argv[1]](sys.argv[2])
