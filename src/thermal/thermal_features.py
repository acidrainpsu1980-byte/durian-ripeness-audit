"""Thermal feature extraction for the durian ripeness project (thermal track).

Input: one 396x528 CSV of temperatures in degrees Fahrenheit (as shipped in dataset_clean_thermal.zip).
Output: dict of engineered features + 128x128 downsampled fruit-centred, z-scored temperature image.
"""
import io, os, sys
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi

sys.path.append(os.getcwd())
import durian_common as dc

HIST_EDGES = np.linspace(-6.0, 2.0, 17)   # bins of (T_fruit - T_background_median), degrees C
MIN_FRAC, MAX_FRAC = 0.03, 0.70            # plausible fruit area as fraction of frame
IMG = 128
QUAL_MIN = 1.5                             # min ring-contrast quality (in background SD units)

_zf = None
def _zip():
    global _zf
    if _zf is None:
        _zf = dc.open_zip("thermal")
    return _zf


def f_to_c(x):
    return (x - 32.0) * 5.0 / 9.0


def load_matrix(member):
    b = dc.read_member(_zip(), member)
    M = pd.read_csv(io.BytesIO(b), header=None).values.astype(np.float32)
    return f_to_c(M)


def otsu_thr(M):
    lo, hi = float(M.min()), float(M.max())
    u8 = np.clip((M - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8).reshape(-1, 1)
    t8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return lo + t8 / 255.0 * (hi - lo)


def _largest_central(mask):
    """Largest connected component weighted by closeness to the frame centre; holes filled."""
    lab, n = ndi.label(mask)
    if n == 0:
        return None
    H, W = mask.shape
    idx = np.arange(1, n + 1)
    sizes = ndi.sum(mask, lab, idx)
    cents = np.array(ndi.center_of_mass(mask, lab, idx))
    d = np.sqrt(((cents[:, 0] - H / 2) / H) ** 2 + ((cents[:, 1] - W / 2) / W) ** 2)
    score = sizes * np.exp(-(d / 0.25) ** 2)
    best = idx[np.argmax(score)]
    return ndi.binary_fill_holes(lab == best)


def _ring_quality(M, mask):
    """(ring_mean - fruit_mean) / ring_std, ring = band just outside the mask. Positive = fruit cooler."""
    m8 = mask.astype(np.uint8)
    outer = cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(bool)
    inner = cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    ring = outer & ~inner
    if ring.sum() < 200:
        return 0.0
    return float((M[ring].mean() - M[mask].mean()) / (M[ring].std() + 1e-6))


def _candidate(Ms, binary):
    mask = _largest_central(binary)
    if mask is None:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    mask = ndi.binary_fill_holes(mask)
    frac = mask.mean()
    if frac < MIN_FRAC or frac > MAX_FRAC:
        return None
    return mask


def segment(M):
    """Fruit = blob that differs from its surrounding background (usually cooler). Returns (mask, info)."""
    H, W = M.shape
    Ms = cv2.GaussianBlur(M, (5, 5), 0)
    # iterative Otsu: if the "hot" class is a small hot spot (lamp, hand) re-threshold the remaining pixels
    sub = Ms.ravel()
    t = otsu_thr(sub)
    for _ in range(3):
        hot_frac = (sub >= t).mean()
        if hot_frac < 0.15 or (sub < t).mean() > MAX_FRAC + 0.1:
            sub = sub[sub < t]
            if sub.size < 1000:
                break
            t = otsu_thr(sub)
        else:
            break
    # candidate thresholds: global iterative Otsu, Otsu on the central 60% window, two-level Otsu
    cy0, cy1, cx0, cx1 = int(0.2 * H), int(0.8 * H), int(0.2 * W), int(0.8 * W)
    thr_list = [t, otsu_thr(Ms[cy0:cy1, cx0:cx1])]
    below = Ms[Ms < t]
    if below.size > 1000:
        thr_list.append(otsu_thr(below))
    best, best_q, warmer = None, 0.0, 0
    for tt in thr_list:
        cand = _candidate(Ms, Ms < tt)
        if cand is None:
            continue
        cnts, _ = cv2.findContours(cand.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        solidity = cv2.contourArea(c) / (cv2.contourArea(cv2.convexHull(c)) + 1e-9)
        if solidity < 0.7:
            continue
        border = cand[0, :].sum() + cand[-1, :].sum() + cand[:, 0].sum() + cand[:, -1].sum()
        if border > 0.25 * cv2.arcLength(c, True):
            continue
        q = _ring_quality(Ms, cand)
        if q > best_q:
            best, best_q, t = cand, q, tt
    fallback = 0
    if best is None or best_q < QUAL_MIN:
        yy, xx = np.ogrid[:H, :W]
        best = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2) <= (0.3 * min(H, W)) ** 2
        fallback, warmer = 1, 0
        best_q = _ring_quality(Ms, best)
    return best, dict(otsu_thr_c=float(t), seg_quality=float(best_q), seg_fallback=fallback, fruit_warmer=warmer)


def features(M, mask, info):
    H, W = M.shape
    f = dict(info)
    fr = M[mask]
    # background: outside a dilated mask (exclude the boundary band)
    dil = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(bool)
    bg = M[~dil] if (~dil).sum() > 500 else M[~mask]
    bg_med = float(np.median(bg))
    f.update(
        fruit_mean=float(fr.mean()), fruit_median=float(np.median(fr)), fruit_std=float(fr.std()),
        fruit_min=float(fr.min()), fruit_max=float(fr.max()),
        fruit_p5=float(np.percentile(fr, 5)), fruit_p25=float(np.percentile(fr, 25)),
        fruit_p75=float(np.percentile(fr, 75)), fruit_p95=float(np.percentile(fr, 95)),
        fruit_skew=float(((fr - fr.mean()) ** 3).mean() / (fr.std() ** 3 + 1e-9)),
        fruit_kurt=float(((fr - fr.mean()) ** 4).mean() / (fr.std() ** 4 + 1e-9) - 3),
        bg_mean=float(bg.mean()), bg_median=bg_med, bg_std=float(bg.std()),
        bg_p5=float(np.percentile(bg, 5)), bg_p95=float(np.percentile(bg, 95)),
        frame_min=float(M.min()), frame_max=float(M.max()), frame_mean=float(M.mean()),
        contrast_mean=float(fr.mean() - bg.mean()), contrast_median=float(np.median(fr) - bg_med),
        contrast_min=float(fr.min() - bg_med), contrast_p5=float(np.percentile(fr, 5) - bg_med),
        contrast_p95=float(np.percentile(fr, 95) - bg_med),
        fruit_iqr=float(np.percentile(fr, 75) - np.percentile(fr, 25)),
        fruit_range=float(fr.max() - fr.min()),
    )
    # shape / area
    area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    cov = np.cov(np.vstack([ys, xs]))
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))) if cnts else area
    perim = cv2.arcLength(max(cnts, key=cv2.contourArea), True) if cnts else 0.0
    f.update(area_px=area, area_frac=area / (H * W),
             eccentricity=float(np.sqrt(1 - ev[1] / (ev[0] + 1e-9))),
             solidity=float(area / (hull_area + 1e-9)),
             circularity=float(4 * np.pi * area / (perim ** 2 + 1e-9)),
             centroid_y=float(cy / H), centroid_x=float(cx / W))
    # radial profile (normalised radius from centroid; R = equivalent radius)
    R = np.sqrt(area / np.pi)
    yy, xx = np.indices(M.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / R
    prof = []
    for lo_, hi_ in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0)]:
        sel = mask & (rr >= lo_) & (rr < hi_)
        prof.append(float(M[sel].mean()) if sel.sum() > 10 else np.nan)
    prof = np.array(prof)
    if np.isnan(prof).any():
        prof = pd.Series(prof).interpolate(limit_direction="both").values
    for i, v in enumerate(prof):
        f[f"radial_{i}"] = float(v - bg_med)
    f["centre_minus_rim"] = float(prof[0] - prof[-1])
    f["radial_slope"] = float(np.polyfit(np.arange(5), prof, 1)[0])
    # gradients / texture inside fruit (eroded to avoid the boundary)
    er = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    if er.sum() < 100:
        er = mask
    gx = cv2.Sobel(M, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(M, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx ** 2 + gy ** 2)[er]
    lap = cv2.Laplacian(M, cv2.CV_32F, ksize=3)[er]
    mu = cv2.blur(M, (7, 7)); mu2 = cv2.blur(M * M, (7, 7))
    lstd = np.sqrt(np.clip(mu2 - mu * mu, 0, None))[er]
    # boundary gradient (edge sharpness)
    edge = mask & ~er
    gedge = np.sqrt(gx ** 2 + gy ** 2)[edge] if edge.sum() > 10 else gm
    f.update(grad_mean=float(gm.mean()), grad_std=float(gm.std()), grad_p50=float(np.median(gm)),
             grad_p95=float(np.percentile(gm, 95)), lap_var=float(lap.var()), local_std_mean=float(lstd.mean()),
             local_std_p95=float(np.percentile(lstd, 95)), edge_grad_mean=float(gedge.mean()),
             n_unique_vals=int(np.unique(np.round(M, 3)).size))
    # 16-bin histogram of (T - bg_median) inside fruit, normalised
    h, _ = np.histogram(fr - bg_med, bins=HIST_EDGES)
    h = h / max(h.sum(), 1)
    for i, v in enumerate(h):
        f[f"hist_{i:02d}"] = float(v)
    f["hist_below"] = float((fr - bg_med < HIST_EDGES[0]).mean())
    f["hist_above"] = float((fr - bg_med >= HIST_EDGES[-1]).mean())
    return f


def crop_image(M, mask):
    """Fruit-centred square crop (1.3x bounding box), resized to IMG x IMG, per-image z-score."""
    ys, xs = np.nonzero(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    half = int(0.65 * max(ys.max() - ys.min(), xs.max() - xs.min(), 60))
    H, W = M.shape
    pad = np.pad(M, half, mode="edge")
    crop = pad[cy:cy + 2 * half, cx:cx + 2 * half]
    crop = cv2.resize(crop, (IMG, IMG), interpolation=cv2.INTER_AREA)
    z = (crop - crop.mean()) / (crop.std() + 1e-6)
    return z.astype(np.float32)


def process(member):
    try:
        M = load_matrix(member)
    except Exception as e:  # unreadable
        return dict(zip_member=member, read_error=str(e)[:200]), None
    mask, info = segment(M)
    f = features(M, mask, info)
    f["zip_member"] = member
    f["shape_h"], f["shape_w"] = M.shape
    f["read_error"] = ""
    return f, crop_image(M, mask)
