#!/usr/bin/env python
"""Round-5 LABEL-FREE frozen-embedding extraction for the durian ripeness project.

No label column is read into any model and no score is computed. File keys and the session keys
(fruit_id, date_captured) are the only metadata used.

Stages (python extract_r5.py --stage <s> [--limit N] --out <dir>):
  smoke     : load every encoder on CUDA, embed one dummy input
  manifest  : package versions, HF commit shas, checkpoint md5s
  audio     : per-knock embeddings (W150 / W500) for all knocks in features_sound_knocks.parquet
  images    : RGB (full / fruit crop), thermal (fixed range / per-image min-max), multispectral (8 bands)
  aggregate : recording- and session-level means (uses fruit_id/date_captured only)
  pack      : transfer copy (patch/time-mean features as float16)
"""
import os, sys, io, json, time, argparse, hashlib, glob, zipfile, traceback, shutil
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
R5 = os.path.expanduser("~/durian_r5")
os.environ.setdefault("DURIAN_DATA_ROOT", os.path.join(R5, "zips"))
os.environ.setdefault("HF_HOME", os.path.join(R5, "hf"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(R5, "numba_cache"))
REPO = os.path.join(R5, "repo", "durian_ripeness_repo")
SRC = os.path.join(REPO, "src")
MODELS = os.path.join(R5, "models")
for p in [os.path.join(MODELS, "beats_code"), os.path.join(SRC, "multispectral"), os.path.join(SRC, "thermal"),
          os.path.join(SRC, "rgb"), os.path.join(SRC, "sound"), SRC]:
    sys.path.insert(0, p)
import numpy as np, pandas as pd
import torch

DEV = "cuda"
GPU_BUDGET_GB = 7.0
WINDOWS = {"W150": 0.150, "W500": 0.500}
THERM_RANGE_C = (20.0, 40.0)   # fixed global window, pre-specified in novelty memo s3.3 (no labels, no data look)
HF_REPOS = {
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "clap": "laion/clap-htsat-unfused",
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
    "dinov2": "facebook/dinov2-base",
    "siglip": "google/siglip-base-patch16-224",
}
F16_SUFFIXES = ("_patchmean.npy", "_patchmean_signal.npy")   # downcast only in the transfer copy


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def setup_torch():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(0); np.random.seed(0)
    tot = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_BUDGET_GB * 1024 ** 3 / tot), 0)


def md5(path, bs=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def hf_path(key):
    from huggingface_hub import snapshot_download
    return snapshot_download(HF_REPOS[key], local_files_only=True)


def _eager(cls, path, **kw):
    try:
        return cls.from_pretrained(path, attn_implementation="eager", **kw)
    except (TypeError, ValueError):
        return cls.from_pretrained(path, **kw)


# ============================================================ audio encoders
# each builder returns dict(sr, min_len, bs, fn(x_np (B,n) float32, n_window) -> {feat: tensor (B,D)}, info)
def build_beats():
    from BEATs import BEATs, BEATsConfig
    ck = torch.load(os.path.join(MODELS, "BEATs_iter3_plus_AS2M.pt"), map_location="cpu", weights_only=False)
    cfg = BEATsConfig(ck["cfg"]); m = BEATs(cfg); m.load_state_dict(ck["model"]); m.eval().to(DEV)
    assert not getattr(cfg, "finetuned_model", False)

    def fn(x, n_window, nvalid=None):
        t = torch.from_numpy(np.ascontiguousarray(x)).to(DEV)
        nv = torch.as_tensor(np.full(len(x), n_window) if nvalid is None else np.minimum(nvalid, n_window), device=DEV)
        pm = torch.arange(t.shape[1], device=DEV)[None, :] >= nv[:, None]          # True = zero-padded sample
        rep, tm = m.extract_features(t, padding_mask=pm)
        keep = (~tm).float() if tm is not None else torch.ones(rep.shape[:2], device=DEV)
        masked = (rep * keep[..., None]).sum(1) / keep.sum(1, keepdim=True).clamp_min(1.0)
        return {"maskedmean": masked, "meanall": rep.mean(1)}
    fn._model = m
    return dict(sr=16000, min_len=16000, bs=64, fn=fn,
                info=dict(pooling="maskedmean = mean of last-layer tokens whose returned padding_mask is False (primary); meanall = mean of all tokens",
                          padding="zero-pad to 1.0 s (16000 samples) with padding_mask=True on padded samples (memo s3.3)",
                          note="BEATs maps the sample-level mask onto tokens by contiguous chunks (forward_padding_mask .all(-1)); used as released"))


def build_panns():
    import torch.nn.functional as F
    from panns_inference.models import Cnn14
    m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    ck = torch.load(os.path.join(MODELS, "Cnn14_mAP=0.431.pth"), map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"]); m.eval().to(DEV)

    def fn(x, n_window, nvalid=None):
        t = torch.from_numpy(np.ascontiguousarray(x)).to(DEV)
        h = m.logmel_extractor(m.spectrogram_extractor(t))
        h = m.bn0(h.transpose(1, 3)).transpose(1, 3)
        for blk, ps in [(m.conv_block1, (2, 2)), (m.conv_block2, (2, 2)), (m.conv_block3, (2, 2)),
                        (m.conv_block4, (2, 2)), (m.conv_block5, (2, 2)), (m.conv_block6, (1, 1))]:
            h = blk(h, pool_size=ps, pool_type="avg")
        h = torch.mean(h, dim=3)                       # (B, 2048, T')
        pooled = torch.max(h, dim=2)[0] + torch.mean(h, dim=2)
        return {"embedding": F.relu(m.fc1(pooled))}
    fn._model = m
    return dict(sr=32000, min_len=32000, bs=64, fn=fn,
                info=dict(pooling="embedding = built-in 2048-d embedding fc1(max_t + mean_t) (identical to Cnn14.forward()['embedding'] in eval mode)",
                          padding="zero-pad to 1.0 s (32000 samples) (memo s3.3)"))


def build_ast():
    from transformers import ASTFeatureExtractor, ASTModel
    p = hf_path("ast")
    fe = ASTFeatureExtractor.from_pretrained(p)
    m = _eager(ASTModel, p).eval().to(DEV)
    fdim, tdim = m.embeddings.get_shape(m.config)
    ts = m.config.time_stride

    def fn(x, n_window, nvalid=None):
        inp = fe(list(x), sampling_rate=16000, return_tensors="pt")["input_values"].to(DEV)
        o = m(input_values=inp); hs = o.last_hidden_state      # (B, 2 + fdim*tdim, 768)
        pt = hs[:, 2:].reshape(hs.shape[0], fdim, tdim, -1)
        nfr = 1 + (n_window - 400) // 160                # real log-mel frames of the window
        tsig = max(1, int(np.ceil(nfr / ts)))            # time-patch columns starting inside the real frames
        return {"pooler": o.pooler_output, "cls": hs[:, 0], "patchmean": pt.mean((1, 2)), "patchmean_signal": pt[:, :, :tsig].mean((1, 2))}
    return dict(sr=16000, min_len=0, bs=16, fn=fn,
                info=dict(pooling=f"pooler = pooler_output = mean(CLS, distill) (primary); cls = token 0; patchmean = mean of all {fdim}x{tdim} patch tokens; patchmean_signal = mean of patch columns whose first frame lies inside the real window",
                          note=f"feature extractor pads the log-mel to max_length={fe.max_length} frames (10.24 s), standard AST input",
                          fe_uses_torchaudio=str(getattr(fe, "use_torchaudio", "unknown"))))


def build_clap():
    from transformers import ClapModel, ClapFeatureExtractor
    p = hf_path("clap")
    fe = ClapFeatureExtractor.from_pretrained(p)
    m = ClapModel.from_pretrained(p).eval().to(DEV)

    import torch.nn.functional as F

    def fn(x, n_window, nvalid=None):
        out = {}
        for pad, tag in [("repeatpad", "repeatpad"), ("pad", "zeropad")]:
            f = fe(list(x), sampling_rate=48000, return_tensors="pt", padding=pad, truncation="rand_trunc")
            il = f.get("is_longer")
            ao = m.audio_model(input_features=f["input_features"].to(DEV), is_longer=None if il is None else il.to(DEV))
            pooled = ao.pooler_output
            out[f"{tag}_projl2"] = F.normalize(m.audio_projection(pooled), dim=-1)   # = get_audio_features, L2-normalised
            if tag == "repeatpad":
                out[f"{tag}_pooler"] = pooled
        return out
    return dict(sr=48000, min_len=0, bs=16, fn=fn,
                info=dict(pooling="projl2 = L2-normalised get_audio_features (audio_projection(pooler), 512-d); pooler = HTSAT pooled 768-d pre-projection",
                          note="feature extractor pads input to 10 s (480000 samples): repeatpad = model default padding='repeatpad' (clip tiled, then zeros; PRIMARY, memo s3.3); zeropad = padding='pad' (sensitivity); truncation='rand_trunc' never triggers (clips < 10 s)"))


def _build_ssl(key):
    from transformers import AutoFeatureExtractor, AutoModel
    p = hf_path(key)
    fe = AutoFeatureExtractor.from_pretrained(p)
    m = _eager(AutoModel, p).eval().to(DEV)
    mid = m.config.num_hidden_layers // 2

    def fn(x, n_window, nvalid=None):
        inp = fe(list(x), sampling_rate=16000, return_tensors="pt")["input_values"].to(DEV)
        o = m(inp, output_hidden_states=True)
        fm = torch.stack([h.mean(1) for h in o.hidden_states[1:]], 0)     # (12, B, D) frame-means of each transformer layer
        return {"alllayers_mean": fm.mean(0), "last_mean": o.last_hidden_state.mean(1), f"mid{mid}_mean": o.hidden_states[mid].mean(1)}
    return dict(sr=16000, min_len=0, bs=64, fn=fn,
                info=dict(pooling=f"alllayers_mean = mean over the {m.config.num_hidden_layers} transformer layers of each layer's frame-mean (primary, memo s3.3); last_mean = frame-mean of last layer; mid{mid}_mean = frame-mean of hidden_states[{mid}] (index 0 = pre-transformer)", padding="none",
                          do_normalize=bool(getattr(fe, "do_normalize", False))))


AUDIO_ENCODERS = {"beats": build_beats, "panns": build_panns, "ast": build_ast, "clap": build_clap,
                  "wav2vec2": lambda: _build_ssl("wav2vec2"), "hubert": lambda: _build_ssl("hubert")}


# ============================================================ vision encoders
# fn(x01 torch (B,3,224,224) float32 in [0,1] on DEV) -> {feat: (B,D)}
def build_dinov2():
    from transformers import AutoModel
    m = _eager(AutoModel, hf_path("dinov2")).eval().to(DEV)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

    def fn(x):
        o = m(pixel_values=(x - mean) / std); hs = o.last_hidden_state
        return {"pooler": o.pooler_output, "patchmean": hs[:, 1:].mean(1)}
    return dict(bs=64, fn=fn, info=dict(pooling="pooler = pooler_output = final-layernormed CLS (primary); patchmean = mean of 256 patch tokens", norm="ImageNet mean/std; input already 224x224 (pad-to-square + resize), no resize/centre-crop"))


def build_clip():
    import open_clip
    name = "ViT-B-16-quickgelu" if "ViT-B-16-quickgelu" in open_clip.list_models() else "ViT-B-16"
    m, _, _ = open_clip.create_model_and_transforms(name, pretrained="openai", cache_dir=os.path.join(R5, "hf", "open_clip"))
    m.eval().to(DEV); m.visual.output_tokens = True
    from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
    mean = torch.tensor(OPENAI_DATASET_MEAN, device=DEV).view(1, 3, 1, 1); std = torch.tensor(OPENAI_DATASET_STD, device=DEV).view(1, 3, 1, 1)

    def fn(x):
        pooled, tokens = m.visual((x - mean) / std)
        return {"proj": pooled, "patchmean": tokens.mean(1)}
    act = type(m.visual.transformer.resblocks[0].mlp.gelu).__name__
    return dict(bs=64, fn=fn, info=dict(open_clip_model=name, mlp_activation=act,
                                        pooling="proj = model-defined image embedding (512-d, un-normalised); patchmean = mean of 196 patch tokens (768-d)", norm="OpenAI CLIP mean/std"))


def build_siglip():
    from transformers import SiglipVisionModel
    m = _eager(SiglipVisionModel, hf_path("siglip")).eval().to(DEV)

    def fn(x):
        o = m(pixel_values=(x - 0.5) / 0.5)
        return {"pooler": o.pooler_output, "patchmean": o.last_hidden_state.mean(1)}
    return dict(bs=64, fn=fn, info=dict(pooling="pooler = MAP-head pooled output (model-defined image embedding); patchmean = mean of 196 post-layernorm patch tokens", norm="mean 0.5 / std 0.5"))


VISION_ENCODERS = {"dinov2": build_dinov2, "clip": build_clip, "siglip": build_siglip}


def free_gpu():
    import gc; gc.collect(); torch.cuda.empty_cache()


# ============================================================ run helpers
@torch.no_grad()
def run_batches(fn, X, bs, prep=None, n_window=None, rows=None, nvalid=None):
    rows = np.arange(len(X)) if rows is None else rows
    outs = {}
    for s in range(0, len(rows), bs):
        idx = rows[s:s + bs]
        xb = X[idx] if prep is None else prep(X[idx])
        o = fn(xb, n_window, None if nvalid is None else nvalid[idx]) if n_window is not None else fn(xb)
        for k, v in o.items():
            outs.setdefault(k, []).append(v.detach().float().cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in outs.items()}


def save_feats(outdir, prefix, feats, index_df):
    os.makedirs(outdir, exist_ok=True)
    shapes = {}
    for k, v in feats.items():
        assert np.isfinite(v).all(), f"non-finite values in {prefix}_{k}"
        assert len(v) == len(index_df)
        np.save(os.path.join(outdir, f"{prefix}_{k}.npy"), v.astype(np.float32))
        shapes[k] = list(v.shape)
    index_df.to_parquet(os.path.join(outdir, f"{prefix}_index.parquet"), index=False)
    return shapes


# ============================================================ audio data
def _audio_worker(args):
    file, member, onsets = args
    import librosa, knock_features as kf
    rec = dict(file=file, zip_member=member, ok=0, err="", sr_native=0, n_native=0, file_peak_abs=np.nan,
               n_knocks_parquet=len(onsets), redetect_n=-1, redetect_maxdiff_s=np.nan)
    try:
        b = kf._zip().read(member)
        y, sr = kf.decode_bytes(b, os.path.splitext(file)[1].lower())
        y = np.asarray(y, np.float32)
        rec.update(sr_native=int(sr), n_native=int(len(y)), file_peak_abs=float(np.abs(y).max()))
        # label-free consistency check: re-run the paper's own onset detector on this platform
        try:
            on, _ = kf.detect_onsets(kf.preprocess(y.copy(), sr))
            rec["redetect_n"] = int(len(on))
            if len(on) == len(onsets):
                rec["redetect_maxdiff_s"] = float(np.abs(on / kf.SR - onsets).max()) if len(on) else 0.0
        except Exception as e:
            rec["err"] = f"redetect: {e}"[:200]
        segs = {}
        for r in (16000, 32000, 48000):
            yr = y if sr == r else librosa.resample(y, orig_sr=sr, target_sr=r, res_type="soxr_hq").astype(np.float32)
            for wn, ws in WINDOWS.items():
                n = int(round(ws * r))
                S = np.zeros((len(onsets), n), np.float32); nv = np.zeros(len(onsets), np.int32); st = np.zeros(len(onsets), np.int64)
                for j, o in enumerate(onsets):
                    a = int(round(float(o) * r)); seg = yr[a:a + n]
                    S[j, :len(seg)] = seg; nv[j] = len(seg); st[j] = a
                segs[(r, wn)] = (S, nv, st)
        rec["ok"] = 1
        return rec, segs
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {e}"[:300]
        return rec, None


class _SerialPool:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def imap(self, f, it, chunksize=1): return map(f, it)


def _pool(workers):
    from multiprocessing import Pool
    return Pool(workers) if workers and workers > 1 else _SerialPool()


def load_audio_segments(limit=None, workers=6):
    kn = pd.read_parquet(os.path.join(REPO, "work/raw/sound/features_sound_knocks.parquet"), columns=["file", "knock_idx", "onset_s"])
    zf = zipfile.ZipFile(os.path.join(os.environ["DURIAN_DATA_ROOT"], "dataset_clean_sound.zip"))
    names = {os.path.basename(n): n for n in zf.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")}
    files = list(dict.fromkeys(kn.file))
    if limit:
        files = files[:limit]
    kn = pd.concat([kn[kn.file == f].sort_values("knock_idx", kind="stable") for f in files], ignore_index=True)  # parquet file order
    missing = [f for f in files if f not in names]
    assert not missing, f"{len(missing)} parquet files not in zip: {missing[:5]}"
    jobs = [(f, names[f], g.onset_s.values.astype(np.float64)) for f, g in kn.groupby("file", sort=False)]
    recs, SEG = {}, {}
    t0 = time.time()
    with _pool(workers) as pool:
        for n, (rec, segs) in enumerate(pool.imap(_audio_worker, jobs, chunksize=2)):
            recs[rec["file"]] = rec; SEG[rec["file"]] = segs
            if (n + 1) % 50 == 0:
                log(f"decoded {n+1}/{len(jobs)} files {time.time()-t0:.0f}s")
    fr = pd.DataFrame([recs[f] for f in files])
    X, NV, ST = {}, {}, {}
    for r in (16000, 32000, 48000):
        for w in WINDOWS:
            n = int(round(WINDOWS[w] * r)); parts, nvs, sts = [], [], []
            for f, g in kn.groupby("file", sort=False):
                if SEG[f] is None:
                    parts.append(np.zeros((len(g), n), np.float32)); nvs.append(np.zeros(len(g), np.int32)); sts.append(np.full(len(g), -1))
                else:
                    S, nv, st = SEG[f][(r, w)]
                    assert len(S) == len(g), f"knock count mismatch for {f}: {len(S)} vs parquet {len(g)}"
                    parts.append(S); nvs.append(nv); sts.append(st)
            X[(r, w)] = np.concatenate(parts); NV[(r, w)] = np.concatenate(nvs); ST[(r, w)] = np.concatenate(sts)
            assert len(X[(r, w)]) == len(kn)
    idx = kn[["file", "knock_idx", "onset_s"]].copy().reset_index(drop=True)
    idx.insert(0, "row", np.arange(len(idx)))
    idx = idx.merge(fr[["file", "zip_member", "ok", "sr_native"]], on="file", how="left")
    idx["device_ext"] = idx.file.str.rsplit(".", n=1).str[-1]
    for r in (16000, 32000, 48000):
        idx[f"start_{r}"] = ST[(r, "W150")]
        for w in WINDOWS:
            idx[f"nvalid_{w}_{r}"] = NV[(r, w)]
    for w in WINDOWS:
        idx[f"peak_abs_{w}_48k"] = np.abs(X[(48000, w)]).max(1)
    return X, idx, fr, NV


SIDE = os.path.join(R5, "emb_audio")      # separate audio-only run (started while the RGB zip was still downloading)


def wait_sidecar():
    """Main-job GPU stages wait while the audio sidecar runs, so only one <=7 GB process uses the GPU."""
    if os.environ.get("R5_AUDIO_SIDECAR") == "1":
        return
    t0 = time.time()
    while os.path.exists(os.path.join(SIDE, "RUNNING")):
        if int(time.time() - t0) % 600 < 60:
            log("waiting for audio sidecar to finish")
        time.sleep(60)


@torch.no_grad()
def run_audio(outdir, limit=None, encoders=None):
    wait_sidecar()
    if (limit is None and encoders is None and os.environ.get("R5_AUDIO_SIDECAR") != "1"
            and os.path.abspath(outdir) == os.path.join(R5, "emb") and os.path.exists(os.path.join(SIDE, "AUDIO_DONE"))):
        os.makedirs(outdir, exist_ok=True); n = 0
        for f in os.listdir(SIDE):
            if f.startswith("audio_"):
                shutil.copy2(os.path.join(SIDE, f), os.path.join(outdir, f)); n += 1
        log(f"audio: copied {n} files from completed sidecar run {SIDE}")
        return json.load(open(os.path.join(outdir, "audio_report.json")))
    setup_torch()
    X, idx, fr, NV = load_audio_segments(limit)
    os.makedirs(outdir, exist_ok=True)
    fr.to_csv(os.path.join(outdir, "audio_file_decode_log.csv"), index=False)
    idx.to_parquet(os.path.join(outdir, "audio_knocks_index.parquet"), index=False)
    log("audio knocks", len(idx), "files", len(fr), "decode failures", int((fr.ok == 0).sum()),
        "redetect count mismatches", int((fr.redetect_n != fr.n_knocks_parquet).sum()))
    report = {"n_knocks": len(idx), "n_files": len(fr), "decode_failures": int((fr.ok == 0).sum()),
              "redetect_count_mismatch_files": int((fr.redetect_n != fr.n_knocks_parquet).sum()),
              "redetect_max_onset_diff_s": float(np.nanmax(fr.redetect_maxdiff_s)) if fr.redetect_maxdiff_s.notna().any() else None,
              "sr_native_counts": fr.sr_native.value_counts().to_dict(),
              "n_knocks_truncated_at_file_end": {w: int((idx[f"nvalid_{w}_48000"] < int(WINDOWS[w] * 48000)).sum()) for w in WINDOWS},
              "encoders": {}}
    for name in (encoders or AUDIO_ENCODERS):
        t0 = time.time()
        try:
            E = AUDIO_ENCODERS[name]()
        except Exception as e:
            report["encoders"][name] = dict(error=f"build failed: {type(e).__name__}: {e}"[:500], tb=traceback.format_exc()[-1500:]); log(name, "BUILD FAILED", e); continue
        rep = report["encoders"][name] = dict(sr=E["sr"], min_len=E["min_len"], info=E["info"], windows={})
        for w in WINDOWS:
            Xw = X[(E["sr"], w)]; n = Xw.shape[1]
            L = max(n, E["min_len"])
            prep = (lambda a, L=L: np.pad(a, ((0, 0), (0, L - a.shape[1])))) if L > n else None
            try:
                nvw = NV[(E["sr"], w)]
                feats = run_batches(E["fn"], Xw, E["bs"], prep=prep, n_window=n, nvalid=nvw)
                k = min(100, len(Xw))
                rer = run_batches(E["fn"], Xw, E["bs"], prep=prep, n_window=n, rows=np.arange(k), nvalid=nvw)
                det = {f: float(np.abs(rer[f] - feats[f][:k]).max()) for f in feats}
                ix = idx.copy(); ix["encoder"] = name; ix["window"] = w; ix["encoder_input_samples"] = L
                shapes = save_feats(outdir, f"audio_{name}_{w}", feats, ix)
                rep["windows"][w] = dict(window_samples=n, zero_padded_to=L if L > n else None,
                                         shapes=shapes, rerun_first_k=k, rerun_max_abs_diff=det)
                log(name, w, shapes, "rerun maxdiff", max(det.values()), f"{time.time()-t0:.0f}s")
            except Exception as e:
                rep["windows"][w] = dict(error=f"{type(e).__name__}: {e}"[:500], tb=traceback.format_exc()[-1500:]); log(name, w, "FAILED", e)
        E = None; free_gpu()
        rep["seconds"] = round(time.time() - t0, 1)
        rep["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2); torch.cuda.reset_peak_memory_stats()
    json.dump(report, open(os.path.join(outdir, "audio_report.json"), "w"), indent=1, default=str)
    return report


# ============================================================ image data
def image_members(zipname, pred, limit=None):
    zf = zipfile.ZipFile(os.path.join(os.environ["DURIAN_DATA_ROOT"], zipname))
    ms = sorted(n for n in zf.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")
                and not os.path.basename(n).startswith("._") and pred(n.lower()))
    return ms[:limit] if limit else ms


def fruit_crop_224(canvas, mk, h, w, y0, x0, out=224, margin=0.12):
    """Verbatim geometry of src/rgb/rgb_models.py fruit_crop (512 canvas, mask after the 128-px round trip)."""
    import cv2
    ys, xs = np.where(mk); a, b_, c, d = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (a + b_) / 2, (c + d) / 2; side = max(b_ - a, d - c) * (1 + 2 * margin); side = max(side, 96)
    iy0, iy1, ix0, ix1 = y0, y0 + h, x0, x0 + w
    side = min(side, h, w)
    cy = np.clip(cy, iy0 + side / 2, iy1 - side / 2); cx = np.clip(cx, ix0 + side / 2, ix1 - side / 2)
    a0, b0 = int(round(cy - side / 2)), int(round(cx - side / 2)); s = int(round(side))
    return cv2.resize(np.ascontiguousarray(canvas[a0:a0 + s, b0:b0 + s]), (out, out), interpolation=cv2.INTER_AREA)


def _rgb_worker(args):
    i, member = args
    import cv2, build_cache, extract_features as ef
    rec, canvas = build_cache._work((i, member))
    rec = dict(rec); rec["seg_ok"] = 0; rec["crop_fallback_full"] = 0; rec["mask_frac"] = np.nan
    if canvas is None:
        return rec, None, None
    full = cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.ascontiguousarray(canvas[rec["y0"]:rec["y0"] + rec["h"], rec["x0"]:rec["x0"] + rec["w"]])
    mask = ef.seg_hsv(img)
    mc = np.zeros((512, 512), np.uint8); mc[rec["y0"]:rec["y0"] + rec["h"], rec["x0"]:rec["x0"] + rec["w"]] = mask
    mk = cv2.resize(cv2.resize(mc, (128, 128), interpolation=cv2.INTER_NEAREST), (512, 512), interpolation=cv2.INTER_NEAREST)
    rec["mask_frac"] = float(mask.mean())
    if mk.any():
        crop = fruit_crop_224(canvas, mk, rec["h"], rec["w"], rec["y0"], rec["x0"]); rec["seg_ok"] = 1
    else:
        crop = full.copy(); rec["crop_fallback_full"] = 1
    return rec, full, crop


def pad_to_square(a, value=0):
    h, w = a.shape[:2]; s = max(h, w); y0, x0 = (s - h) // 2, (s - w) // 2
    out = np.full((s, s) + a.shape[2:], value, a.dtype); out[y0:y0 + h, x0:x0 + w] = a
    return out


def _u8_fixed(T):
    lo, hi = THERM_RANGE_C
    return np.round(np.clip((T - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _u8_minmax(T):
    return np.round((T - T.min()) / max(float(T.max() - T.min()), 1e-6) * 255).astype(np.uint8)


THERM_VARIANTS = ["padsq_fixed20to40C", "padsq_minmax", "crop_fixed20to40C", "crop_minmax"]


def _thermal_worker(member):
    import cv2, thermal_features as tf
    rec = dict(zip_member=member, ok=0, err="")
    try:
        M = tf.load_matrix(member)                      # deg F -> deg C (repo f_to_c)
        lo, hi = THERM_RANGE_C
        out = {}
        # primary: full frame, fixed 20-40 C -> 0-255, pad to square (0), resize 224
        out["padsq_fixed20to40C"] = cv2.resize(pad_to_square(_u8_fixed(M)), (224, 224), interpolation=cv2.INTER_AREA)
        out["padsq_minmax"] = cv2.resize(pad_to_square(_u8_minmax(M)), (224, 224), interpolation=cv2.INTER_AREA)
        # secondary: repo fruit crop (crop_image geometry, square by construction), resized to 224 in deg C, no z-score
        mask, info = tf.segment(M)
        ys, xs = np.nonzero(mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        half = int(0.65 * max(ys.max() - ys.min(), xs.max() - xs.min(), 60))
        pad = np.pad(M, half, mode="edge")
        crop = cv2.resize(pad[cy:cy + 2 * half, cx:cx + 2 * half], (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
        out["crop_fixed20to40C"] = _u8_fixed(crop); out["crop_minmax"] = _u8_minmax(crop)
        rec.update(ok=1, shape_h=M.shape[0], shape_w=M.shape[1], frame_min_c=float(M.min()), frame_max_c=float(M.max()),
                   frac_below_20C=float((M < lo).mean()), frac_above_40C=float((M > hi).mean()),
                   crop_frac_below_20C=float((crop < lo).mean()), crop_frac_above_40C=float((crop > hi).mean()),
                   seg_fallback=int(info["seg_fallback"]), seg_quality=float(info["seg_quality"]), crop_side_px=2 * half)
        return rec, out
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {e}"[:300]
        return rec, None


def _ms_worker(member):
    import cv2, tifffile, ms_features as msf
    rec = dict(zip_member=member, ok=0, err="")
    try:
        with zipfile.ZipFile(os.path.join(os.environ["DURIAN_DATA_ROOT"], "dataset_clean_multispectral.zip")) as zf:
            raw = zf.read(member)
        arr = tifffile.imread(io.BytesIO(raw))
        cube = msf.read_montage(raw)                    # (8, 256, 256) float32, repo band order (B1_720 ... B8_980)
        scale = 255.0 if arr.dtype == np.uint8 else (float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else 1.0)
        out = np.stack([cv2.resize(b, (224, 224), interpolation=cv2.INTER_AREA) for b in cube]) / scale
        rec.update(ok=1, raw_dtype=str(arr.dtype), raw_shape="x".join(map(str, arr.shape)), scale=scale, dn_max=float(cube.max()), dn_min=float(cube.min()))
        return rec, np.clip(out, 0, 1).astype(np.float32)
    except Exception as e:
        rec["err"] = f"{type(e).__name__}: {e}"[:300]
        return rec, None


def load_images(limit=None, workers=6, modalities=("rgb", "thermal", "multispectral")):
    data = {}
    t0 = time.time()
    rgb_m = image_members("dataset_clean_rgb.zip", lambda n: n.endswith((".jpg", ".jpeg", ".png")), limit)
    th_m = image_members("dataset_clean_thermal.zip", lambda n: n.endswith(".csv") and "_thermal" in os.path.basename(n), limit)  # incl. *_thermal_03.csv
    ms_m = image_members("dataset_clean_multispectral.zip", lambda n: n.endswith((".tif", ".tiff")), limit)
    with _pool(workers) as pool:
      if "rgb" in modalities:
        recs, F, C = [], [], []
        for rec, full, crop in pool.imap(_rgb_worker, list(enumerate(rgb_m)), chunksize=2):
            recs.append(rec); F.append(full if full is not None else np.zeros((224, 224, 3), np.uint8)); C.append(crop if crop is not None else np.zeros((224, 224, 3), np.uint8))
        r = pd.DataFrame(recs).drop(columns=["row"], errors="ignore"); r["zip_member"] = rgb_m
        data["rgb"] = dict(index=r, variants={"padsq": np.stack(F), "crop": np.stack(C)})
        log("rgb loaded", len(r), "fail", int((r.ok == 0).sum()), "seg fail", int((r.seg_ok == 0).sum()), f"{time.time()-t0:.0f}s")
      if "thermal" in modalities:
        recs, TV = [], {v: [] for v in THERM_VARIANTS}
        for rec, out in pool.imap(_thermal_worker, th_m, chunksize=4):
            recs.append(rec)
            for v in THERM_VARIANTS:
                TV[v].append(out[v] if out is not None else np.zeros((224, 224), np.uint8))
        data["thermal"] = dict(index=pd.DataFrame(recs), variants={v: np.stack(TV[v]) for v in THERM_VARIANTS})
        log("thermal loaded", len(recs), "fail", int((data["thermal"]["index"].ok == 0).sum()), f"{time.time()-t0:.0f}s")
      if "multispectral" in modalities:
        recs, Mx = [], []
        for rec, cube in pool.imap(_ms_worker, ms_m, chunksize=4):
            recs.append(rec); Mx.append(cube if cube is not None else np.zeros((8, 224, 224), np.float32))
        data["multispectral"] = dict(index=pd.DataFrame(recs), variants={"band8": np.stack(Mx)})
        log("ms loaded", len(Mx), "fail", int((data["multispectral"]["index"].ok == 0).sum()), f"{time.time()-t0:.0f}s")
    return data


def _to_x01(a):
    t = torch.from_numpy(np.ascontiguousarray(a)).to(DEV).float()
    if a.dtype == np.uint8:
        t = t.div(255.0)
    if t.ndim == 4:                  # (B,224,224,3) RGB
        return t.permute(0, 3, 1, 2).contiguous()
    return t.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()     # (B,224,224) grey -> replicated to 3 channels


@torch.no_grad()
def run_images(outdir, limit=None, encoders=None, modalities=None):
    wait_sidecar()
    setup_torch()
    mods = tuple(modalities) if modalities else ("rgb", "thermal", "multispectral")
    data = load_images(limit, modalities=mods)
    folds = pd.read_csv(os.path.join(REPO, "data/durian_mm_capture_folds.csv"), usecols=["modality", "file", "zip_member", "fruit_id", "date_captured"])
    join = {}
    os.makedirs(outdir, exist_ok=True)
    for mod, d in data.items():
        ix = d["index"].copy()
        ix.insert(0, "row", np.arange(len(ix))); ix["file"] = ix.zip_member.map(os.path.basename)
        fm = folds[folds.modality == mod]
        ix["in_capture_folds"] = ix.zip_member.isin(set(fm.zip_member)).astype(int)
        join[mod] = dict(n_zip_files=len(ix), n_matched=int(ix.in_capture_folds.sum()), n_unmatched=int((ix.in_capture_folds == 0).sum()),
                         unmatched_examples=ix.loc[ix.in_capture_folds == 0, "zip_member"].head(10).tolist(),
                         n_fold_rows=len(fm), n_fold_rows_missing_in_zip=int((~fm.zip_member.isin(set(ix.zip_member))).sum()),
                         n_failed=int((ix.ok == 0).sum()))
        d["index"] = ix
        ix.to_parquet(os.path.join(outdir, f"{mod}_images_index.parquet"), index=False)
    log("join", json.dumps(join))
    report = {"join_capture_folds": join, "thermal_fixed_range_c": THERM_RANGE_C, "encoders": {}}
    for name, builder in VISION_ENCODERS.items():
        if encoders and name not in encoders:
            continue
        t0 = time.time()
        try:
            E = builder()
        except Exception as e:
            report["encoders"][name] = dict(error=f"build failed: {type(e).__name__}: {e}"[:500], tb=traceback.format_exc()[-1500:]); log(name, "BUILD FAILED", e); continue
        rep = report["encoders"][name] = dict(info=E["info"], runs={})
        for mod, d in data.items():
            for var, A in d["variants"].items():
                try:
                    flat = A.reshape(len(A) * 8, 224, 224) if mod == "multispectral" else A
                    feats = run_batches(E["fn"], flat, E["bs"], prep=_to_x01)
                    k = min(64, len(flat)); rer = run_batches(E["fn"], flat, E["bs"], prep=_to_x01, rows=np.arange(k))
                    det = {f: float(np.abs(rer[f] - feats[f][:k]).max()) for f in feats}
                    if mod == "multispectral":
                        feats = {f: v.reshape(len(A), 8, -1) for f, v in feats.items()}
                    ix = d["index"].copy(); ix["encoder"] = name; ix["variant"] = var
                    shapes = save_feats(outdir, f"{mod}_{name}_{var}", feats, ix)
                    rep["runs"][f"{mod}_{var}"] = dict(shapes=shapes, rerun_first_k=k, rerun_max_abs_diff=det)
                    log(name, mod, var, shapes, "rerun maxdiff", max(det.values()), f"{time.time()-t0:.0f}s")
                except Exception as e:
                    rep["runs"][f"{mod}_{var}"] = dict(error=f"{type(e).__name__}: {e}"[:500], tb=traceback.format_exc()[-1500:]); log(name, mod, var, "FAILED", e)
        E = None; free_gpu()
        rep["seconds"] = round(time.time() - t0, 1)
        rep["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2); torch.cuda.reset_peak_memory_stats()
    th = data["thermal"]["index"] if "thermal" in data else pd.DataFrame()
    if "frac_below_20C" in th:
        report["thermal_range_clipping"] = dict(window_c=THERM_RANGE_C,
            full_frame=dict(mean_frac_below=float(th.frac_below_20C.mean()), mean_frac_above=float(th.frac_above_40C.mean()),
                            n_images_clip_gt_1pct=int(((th.frac_below_20C + th.frac_above_40C) > 0.01).sum()),
                            frame_min_c_p01=float(th.frame_min_c.quantile(0.01)), frame_max_c_p99=float(th.frame_max_c.quantile(0.99))),
            fruit_crop=dict(mean_frac_below=float(th.crop_frac_below_20C.mean()), mean_frac_above=float(th.crop_frac_above_40C.mean()),
                            n_images_clip_gt_1pct=int(((th.crop_frac_below_20C + th.crop_frac_above_40C) > 0.01).sum())),
            n_seg_fallback=int(th.seg_fallback.sum()))
    rg = data["rgb"]["index"] if "rgb" in data else pd.DataFrame({"seg_ok": [], "crop_fallback_full": []})
    report["rgb_segmentation"] = dict(n_seg_ok=int(rg.seg_ok.sum()), n_crop_fallback_full=int(rg.crop_fallback_full.sum()),
                                      camera_counts=rg.camera.value_counts().to_dict() if "camera" in rg else None)
    msx = data["multispectral"]["index"] if "multispectral" in data else pd.DataFrame()
    if "raw_dtype" in msx:
        report["ms_raw"] = dict(dtypes=msx.raw_dtype.value_counts().to_dict(), shapes=msx.raw_shape.value_counts().to_dict(), dn_max=float(msx.dn_max.max()),
                                failed=msx.loc[msx.ok == 0, ["zip_member", "err"]].to_dict("records"))
    for mod, d in data.items():
        report.setdefault("failed_files", {})[mod] = d["index"].loc[d["index"].ok == 0, ["zip_member", "err"]].to_dict("records")
    name = "images_report.json" if set(mods) == {"rgb", "thermal", "multispectral"} else f"images_report_{'_'.join(mods)}.json"
    json.dump(report, open(os.path.join(outdir, name), "w"), indent=1, default=str)
    return report


# ============================================================ aggregation (keys only, no labels)
def aggregate(outdir):
    agg = os.path.join(outdir, "aggregated"); os.makedirs(agg, exist_ok=True)
    rec = pd.read_csv(os.path.join(REPO, "data/durian_mm_sound_recordings.csv"), usecols=["file", "fruit_id", "date_captured"])
    rec["session_id"] = rec.fruit_id.astype(int).astype(str) + "_" + rec.date_captured.astype(str)
    folds = pd.read_csv(os.path.join(REPO, "data/durian_mm_capture_folds.csv"), usecols=["modality", "zip_member", "fruit_id", "date_captured"])
    folds["session_id"] = folds.fruit_id.astype(int).astype(str) + "_" + folds.date_captured.astype(str)
    summary = {}
    for p in sorted(glob.glob(os.path.join(outdir, "*_index.parquet"))):
        prefix = os.path.basename(p)[:-len("_index.parquet")]
        if prefix.endswith("_images") or prefix == "audio_knocks":
            continue
        ix = pd.read_parquet(p)
        feats = {os.path.basename(f)[len(prefix) + 1:-4]: f for f in glob.glob(os.path.join(outdir, prefix + "_*.npy"))}
        if prefix.startswith("audio_"):
            ix = ix.merge(rec[["file", "session_id"]], on="file", how="left")
            levels = {"recording": ix.file.values, "session": ix.session_id.values}
        else:
            mod = prefix.split("_")[0]
            fm = folds[folds.modality == mod]
            levels = {"session": ix.zip_member.map(dict(zip(fm.zip_member, fm.session_id))).values}
        for lvl, keys in levels.items():
            keys = pd.Series(keys).fillna("UNMATCHED").astype(str).values
            uk, inv = np.unique(keys, return_inverse=True)
            cnt = np.bincount(inv)
            for fname, fpath in feats.items():
                V = np.load(fpath)
                S = np.zeros((len(uk),) + V.shape[1:], np.float64); np.add.at(S, inv, V)
                np.save(os.path.join(agg, f"{prefix}_{fname}_{lvl}.npy"), (S / cnt.reshape((-1,) + (1,) * (V.ndim - 1))).astype(np.float32))
            pd.DataFrame({"key": uk, "n_items": cnt}).to_parquet(os.path.join(agg, f"{prefix}_{lvl}_index.parquet"), index=False)
            summary[f"{prefix}_{lvl}"] = dict(n_groups=int(len(uk)), n_unmatched_items=int((keys == "UNMATCHED").sum()))
    json.dump(summary, open(os.path.join(agg, "aggregate_summary.json"), "w"), indent=1)
    log("aggregated", len(summary))


def pack(src, dst):
    """Transfer copy: identical files except patch/time-mean features (float16)."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    conv = []
    for root, _, files in os.walk(src):
        for f in files:
            s = os.path.join(root, f); d = os.path.join(dst, os.path.relpath(s, src)); os.makedirs(os.path.dirname(d), exist_ok=True)
            if f.endswith(".npy") and "aggregated" not in root and f.endswith(F16_SUFFIXES):
                np.save(d, np.load(s).astype(np.float16)); conv.append(os.path.relpath(s, src))
            else:
                shutil.copy2(s, d)
    json.dump(dict(float16_files=sorted(conv), note="all other .npy are float32; host copy ~/durian_r5/emb keeps float32 for every file"),
              open(os.path.join(dst, "pack_float16_list.json"), "w"), indent=1)
    log("packed", len(conv), "float16 files")


# ============================================================ smoke + manifest
@torch.no_grad()
def smoke(outdir):
    setup_torch()
    os.makedirs(outdir, exist_ok=True)
    res = {}
    for name, b in AUDIO_ENCODERS.items():
        try:
            E = b(); n = int(round(0.15 * E["sr"])); L = max(n, E["min_len"])
            x = (np.random.RandomState(0).randn(2, L) * 0.05).astype(np.float32)
            o = E["fn"](x, n)
            r = dict(ok=True, shapes={k: list(v.shape) for k, v in o.items()}, info=E["info"], min_len=E["min_len"])
            if name == "panns":   # hand-written forward must equal the library forward's embedding
                ref = E["fn"]._model(torch.from_numpy(x).to(DEV))["embedding"]
                r["max_abs_diff_vs_library_forward"] = float((ref - o["embedding"]).abs().max())
            if name in ("panns", "beats"):
                def works(L_):
                    try:
                        o_ = E["fn"](x[:, :L_] if L_ <= x.shape[1] else np.pad(x, ((0, 0), (0, L_ - x.shape[1]))), L_); return all(torch.isfinite(v).all() for v in o_.values())
                    except Exception:
                        return False
                r["unpadded_150ms_runs"] = works(n)
                r["padded_1s_150ms_mask_runs"] = True
                lo_, hi_ = 64, E["sr"]
                if works(lo_):
                    r["min_len_observed_samples"] = lo_
                else:
                    while hi_ - lo_ > 1:
                        mid_ = (lo_ + hi_) // 2
                        lo_, hi_ = (lo_, mid_) if works(mid_) else (mid_, hi_)
                    r["min_len_observed_samples"] = hi_
                r["min_len_observed_ms"] = round(1000 * r["min_len_observed_samples"] / E["sr"], 2)
            res[name] = r
        except Exception as e:
            res[name] = dict(ok=False, error=f"{type(e).__name__}: {e}"[:500], tb=traceback.format_exc()[-2000:])
        log(name, json.dumps({k: v for k, v in res[name].items() if k != "tb"}, default=str)[:500])
        E = None; free_gpu()
    for name, b in VISION_ENCODERS.items():
        try:
            E = b(); o = E["fn"](torch.rand(2, 3, 224, 224, device=DEV))
            res[name] = dict(ok=True, shapes={k: list(v.shape) for k, v in o.items()}, info=E["info"])
        except Exception as e:
            res[name] = dict(ok=False, error=f"{type(e).__name__}: {e}"[:500], tb=traceback.format_exc()[-2000:])
        log(name, json.dumps({k: v for k, v in res[name].items() if k != "tb"}, default=str)[:500])
        E = None; free_gpu()
    res["_peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    json.dump(res, open(os.path.join(outdir, "smoke_report.json"), "w"), indent=1, default=str)
    bad = [k for k, v in res.items() if isinstance(v, dict) and not v.get("ok")]
    log("SMOKE_FAILED" if bad else "SMOKE_OK", bad)
    return res


def manifest(outdir):
    import importlib.metadata as md, platform, subprocess
    pk = ["torch", "torchaudio", "torchvision", "transformers", "timm", "open_clip_torch", "librosa", "soundfile", "soxr", "numpy", "pandas",
          "pyarrow", "scipy", "pillow", "huggingface_hub", "opencv-python-headless", "tifffile", "scikit-image", "panns_inference", "torchlibrosa",
          "safetensors", "numba", "tokenizers", "sentencepiece", "imagecodecs"]
    vers = {}
    for p in pk:
        try: vers[p] = md.version(p)
        except Exception: vers[p] = None
    from huggingface_hub import snapshot_download
    hf = {}
    for k, repo in HF_REPOS.items():
        p = snapshot_download(repo, local_files_only=True)
        files = {os.path.relpath(f, p): dict(md5=md5(f), bytes=os.path.getsize(f)) for f in sorted(glob.glob(os.path.join(p, "**", "*"), recursive=True)) if os.path.isfile(f)}
        hf[k] = dict(repo=repo, commit_sha=os.path.basename(p), files=files)
    oc = sorted(set(glob.glob(os.path.join(R5, "hf", "open_clip", "**", "*.bin"), recursive=True) + glob.glob(os.path.join(R5, "hf", "open_clip", "**", "*.safetensors"), recursive=True)))
    ocinfo = [dict(path=os.path.relpath(f, R5), md5=md5(f), bytes=os.path.getsize(f)) for f in oc]
    side = json.load(open(os.path.join(MODELS, "model_sources.json"))) if os.path.exists(os.path.join(MODELS, "model_sources.json")) else {}
    ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[:1]
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    man = dict(env_name="durian-fm", env_prefix=sys.prefix, python=platform.python_version(), packages=vers, ffmpeg=ff, gpu=smi,
               torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
               hf_models=hf, open_clip_checkpoints=ocinfo, other_checkpoints=side,
               determinism=dict(cudnn_deterministic=True, cudnn_benchmark=False, tf32=False, use_deterministic_algorithms="True (warn_only)",
                                CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"), attn_implementation="eager where supported", dtype="float32"),
               gpu_memory_budget_gb=GPU_BUDGET_GB)
    json.dump(man, open(os.path.join(outdir, "manifest_env_models.json"), "w"), indent=1, default=str)
    log("manifest written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "manifest", "audio", "images", "aggregate", "pack"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--encoders", default=None, help="comma list (audio/images stage); default all")
    ap.add_argument("--modalities", default=None, help="comma list for the images stage; default rgb,thermal,multispectral")
    a = ap.parse_args()
    enc = a.encoders.split(",") if a.encoders else None
    {"smoke": lambda: smoke(a.out), "manifest": lambda: manifest(a.out), "audio": lambda: run_audio(a.out, a.limit, enc),
     "images": lambda: run_images(a.out, a.limit, enc, a.modalities.split(",") if a.modalities else None), "aggregate": lambda: aggregate(a.out), "pack": lambda: pack(a.src, a.out)}[a.stage]()
