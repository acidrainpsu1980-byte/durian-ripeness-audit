"""Per-file knock segmentation + feature extraction for the durian acoustic track."""
import os, io, sys, tempfile, subprocess, zipfile
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.abspath("numba_cache"))
import numpy as np
# v1.8: conda-forge soundfile has no bundled libsndfile and ctypes cannot see $CONDA_PREFIX/lib on macOS (DYLD_* variables are
# stripped by System Integrity Protection when scripts start through /usr/bin/env). Point the lookup at the environment's own
# library before importing soundfile; this changes only how the library is located.
import ctypes.util as _cu, glob as _glob
_fl = _cu.find_library
def _find_library(name, _fl=_fl):
    r = _fl(name)
    if r is None and name == "sndfile":
        for _p in dict.fromkeys([os.environ.get("CONDA_PREFIX", ""), sys.prefix, sys.base_prefix]):
            c = sorted(_glob.glob(os.path.join(_p, "lib", "libsndfile*.dylib")) + _glob.glob(os.path.join(_p, "lib", "libsndfile.so*"))) if _p else []
            if c:
                r = c[0]; break
    return r
_cu.find_library = _find_library
import soundfile
import librosa
from scipy.signal import butter, sosfiltfilt

SR = 22050
HOP = 256
WIN_S = 0.150
WIN = int(WIN_S * SR)              # 3307 samples
N_MELS = 64
MEL_NFFT, MEL_HOP = 512, 110       # -> 31 frames per 150 ms knock
N_MFCC = 20
DELTA = 0.12
MIN_GAP_S = 0.12
ZIP_PATH = os.path.join(os.environ.get("DURIAN_DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "raw")), "dataset_clean_sound.zip")
_SOS_HP = butter(4, 80, btype="highpass", fs=SR, output="sos")
_ZF = None


def _zip():
    global _ZF
    if _ZF is None:
        _ZF = zipfile.ZipFile(ZIP_PATH)
    return _ZF


def decode_bytes(b, suffix):
    """(y float32 mono, native sr). wav via libsndfile; m4a via ffmpeg -> f32le pipe."""
    if suffix == ".wav":
        y, sr = soundfile.read(io.BytesIO(b), dtype="float32", always_2d=True)
        return y.mean(1), sr
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(b); p = f.name
    try:
        pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                             "stream=sample_rate", "-of", "csv=p=0", p], capture_output=True, text=True)
        sr = int(pr.stdout.strip().split(",")[0])
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "f32le", "-ac", "1",
                              "-acodec", "pcm_f32le", "-"], capture_output=True)
        if out.returncode != 0 or len(out.stdout) == 0:
            raise RuntimeError("ffmpeg: " + out.stderr.decode()[:200])
        return np.frombuffer(out.stdout, dtype=np.float32).copy(), sr
    finally:
        os.unlink(p)


import threading as _threading
_RESAMPLE_LOCK = _threading.Lock()


def preprocess(y, sr):
    if sr != SR:
        with _RESAMPLE_LOCK:          # v1.8: libsoxr (OpenMP) is not safe to call from several Python threads at once (intermittent SIGSEGV)
            y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    y = sosfiltfilt(_SOS_HP, y).astype(np.float32)
    return y / (np.abs(y).max() + 1e-9)


def detect_onsets(y, delta=DELTA, min_gap_s=MIN_GAP_S):
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    env = np.log1p(rms * 100)
    oenv = np.maximum(0, np.diff(env, prepend=env[0]))
    oenv = oenv / (oenv.max() + 1e-9)
    wait = int(min_gap_s * SR / HOP)
    on = librosa.onset.onset_detect(onset_envelope=oenv, sr=SR, hop_length=HOP, backtrack=True,
                                    units="samples", pre_max=3, post_max=3, pre_avg=10, post_avg=10,
                                    delta=delta, wait=wait)
    on = np.asarray(on, dtype=int)
    on = on[on + WIN <= len(y)]
    pk = np.array([np.abs(y[o:o + WIN]).max() for o in on]) if len(on) else np.zeros(0)
    return on[pk > 0.05], oenv


_MEL_FB = librosa.filters.mel(sr=SR, n_fft=MEL_NFFT, n_mels=N_MELS, fmin=50, fmax=SR // 2)


def logmel(seg):
    S = np.abs(librosa.stft(seg, n_fft=MEL_NFFT, hop_length=MEL_HOP, center=True)) ** 2
    M = _MEL_FB @ S
    return librosa.power_to_db(M, ref=1.0, top_db=None).astype(np.float32)   # (64, 31)


def decay_time_20db(seg):
    fr = 64
    env = np.sqrt(np.convolve(seg.astype(np.float64) ** 2, np.ones(fr) / fr, mode="same")) + 1e-9
    env_db = 20 * np.log10(env)
    ip = int(np.argmax(env_db)); thr = env_db[ip] - 20
    below = np.flatnonzero(env_db[ip:] <= thr)
    return (below[0] / SR) if len(below) else WIN_S


def knock_features(seg):
    """Scalar feature dict + log-mel (64x31) for a single 150 ms knock segment."""
    M = logmel(seg)
    mf = librosa.feature.mfcc(S=M, n_mfcc=N_MFCC)              # (20, T)
    dmf = librosa.feature.delta(mf, width=5)
    S = np.abs(librosa.stft(seg, n_fft=MEL_NFFT, hop_length=MEL_HOP))
    cen = librosa.feature.spectral_centroid(S=S, sr=SR).mean()
    bw = librosa.feature.spectral_bandwidth(S=S, sr=SR).mean()
    ro = librosa.feature.spectral_rolloff(S=S, sr=SR, roll_percent=0.85).mean()
    fl = librosa.feature.spectral_flatness(S=S).mean()
    zcr = librosa.feature.zero_crossing_rate(seg, frame_length=512, hop_length=MEL_HOP).mean()
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / SR)
    domf = freqs[int(np.argmax(spec))]
    f = {}
    for i in range(N_MFCC):
        f[f"mfcc{i:02d}_mean"] = float(mf[i].mean())
    for i in range(N_MFCC):
        f[f"mfcc{i:02d}_std"] = float(mf[i].std())
    for i in range(N_MFCC):
        f[f"dmfcc{i:02d}_mean"] = float(dmf[i].mean())
    mm = M.mean(1)
    for i in range(N_MELS):
        f[f"logmel{i:02d}"] = float(mm[i])
    f.update(centroid=float(cen), bandwidth=float(bw), rolloff=float(ro), flatness=float(fl),
             zcr=float(zcr), dom_freq=float(domf), decay20=float(decay_time_20db(seg)),
             rms=float(np.sqrt(np.mean(seg ** 2))))
    return f, M


def process_file(zip_member, fname):
    """Returns dict(ok, error, duration_s, n_knocks, onsets_s, feats(list of dict), mels(array))."""
    try:
        b = _zip().read(zip_member)
        y, sr = decode_bytes(b, os.path.splitext(fname)[1].lower())
        y = preprocess(y, sr)
        on, _ = detect_onsets(y)
        feats, mels = [], []
        for o in on:
            seg = y[o:o + WIN]
            f, M = knock_features(seg)
            feats.append(f); mels.append(M)
        return dict(ok=True, error="", sr_native=int(sr), duration_s=float(len(y) / SR), n_knocks=int(len(on)),
                    onsets_s=(on / SR).astype(np.float32), feats=feats,
                    mels=np.stack(mels) if mels else np.zeros((0, N_MELS, 31), np.float32))
    except Exception as e:  # report, never drop silently
        return dict(ok=False, error=f"{type(e).__name__}: {e}"[:300], sr_native=0, duration_s=0.0, n_knocks=0,
                    onsets_s=np.zeros(0, np.float32), feats=[], mels=np.zeros((0, N_MELS, 31), np.float32))
