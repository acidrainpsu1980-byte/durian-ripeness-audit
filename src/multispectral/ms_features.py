"""Multispectral (Tetracam/Spectral Devices MSC2-NIR8-1-A) feature extraction.

Montage layout: 512 x 1024 uint8, 2 rows x 4 cols of 256 x 256 tiles, one tile per band.
Tile order (row-major) is ASSUMED to follow ascending band centre wavelength as listed in
the manufacturer datasheet: 720, 760, 800, 840, 860, 900, 940, 980 nm (FWHM ~20 nm).
The files carry no per-band wavelength tag, so this mapping cannot be verified from the data.
"""
import io
import numpy as np
import tifffile
from scipy import ndimage as ndi
import cv2


def threshold_otsu(img):
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)

WAVELENGTHS = np.array([720, 760, 800, 840, 860, 900, 940, 980], dtype=float)
BAND_NAMES = [f"B{i+1}_{int(w)}" for i, w in enumerate(WAVELENGTHS)]
PAIRS = [(i, j) for i in range(8) for j in range(i + 1, 8)]  # 28 pairs


def read_montage(raw_bytes):
    arr = tifffile.imread(io.BytesIO(raw_bytes))
    if arr.ndim == 3 and arr.shape[0] == 8:  # already band-stacked
        return arr.astype(np.float32)
    if arr.shape != (512, 1024):
        raise ValueError(f"unexpected montage shape {arr.shape}")
    return np.stack([arr[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256]
                     for r in range(2) for c in range(4)]).astype(np.float32)


def _poly2_design(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx - w / 2) / w; y = (yy - h / 2) / h
    return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y, x**3, y**3, x*x*y, x*y*y], -1).reshape(-1, 10)


def illum_field(img, bg_mask):
    """Fit a 3rd-order 2-D polynomial to background pixels -> smooth illumination field."""
    A = _poly2_design(*img.shape)
    sel = bg_mask.ravel()
    coef, *_ = np.linalg.lstsq(A[sel], img.ravel()[sel], rcond=None)
    return np.maximum(A @ coef, 1.0).reshape(img.shape)


def _dark_largest(score, open_r, erode_px):
    """Darker Otsu class -> opening (detaches stem) -> drop frame-like components
    (bounding box covering >60% of image area, i.e. vignetted border regions) -> largest -> closing (15 px) -> fill holes."""
    t = threshold_otsu(score)
    dark = score < t
    if open_r > 0:
        dark = ndi.binary_opening(dark, structure=np.ones((2 * open_r + 1, 2 * open_r + 1)))
    lab, n = ndi.label(dark)
    if n == 0:
        return np.zeros(score.shape, bool), t
    H, W = score.shape
    best, best_sz = None, 0
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        bh = sl[0].stop - sl[0].start; bw = sl[1].stop - sl[1].start
        if bh * bw > 0.6 * H * W:
            continue
        sz = int((lab[sl] == k).sum())
        if sz > best_sz:
            best, best_sz = k, sz
    if best is None:
        return np.zeros(score.shape, bool), t
    mask = lab == best
    mask = ndi.binary_closing(mask, structure=np.ones((15, 15)), border_value=0)  # bridge gaps between spines
    mask = ndi.binary_fill_holes(mask)
    if erode_px > 0:
        mask = ndi.binary_erosion(mask, iterations=erode_px)
    return mask, t


def segment_fruit(cube, open_r=6, erode_px=2):
    """Fruit is DARK on a bright, vignetted background.

    1) brightest band (highest mean); coarse illumination field = grey closing (61 px) + blur;
       Otsu on image/field -> darker class, drop border-touching components, opening (removes stem),
       largest component, fill holes.
    2) refit illumination as a 3rd-order polynomial surface on background pixels (outside the
       dilated coarse mask) for EVERY band -> per-pixel pseudo-reflectance R_b = DN_b / field_b.
    3) final mask: Otsu on R_brightest with the same component logic, 2 px erosion.
    Returns (mask, reflectance_cube, brightest_band_index, otsu_threshold).
    """
    b = int(np.argmax(cube.reshape(8, -1).mean(1)))
    img = ndi.gaussian_filter(cube[b], 1.0)
    coarse = ndi.gaussian_filter(ndi.grey_closing(img, size=(101, 101)), 15)
    m0, _ = _dark_largest(np.clip(img / np.maximum(coarse, 1) * 200, 0, 255), open_r, 0)
    if m0.sum() < 200:
        return np.zeros(img.shape, bool), None, b, np.nan
    bg = ~ndi.binary_dilation(m0, iterations=12)
    R = np.stack([cube[i] / illum_field(ndi.gaussian_filter(cube[i], 1.0), bg) for i in range(8)])
    mask, t = _dark_largest(np.clip(ndi.gaussian_filter(R[b], 1.0) * 200, 0, 255), open_r, erode_px)
    return mask, R, b, float(t) / 200.0


def features_from_cube(cube):
    mask, R, bidx, thr = segment_fruit(cube)
    area = int(mask.sum())
    f = {"fruit_area_px": area, "seg_band_idx": bidx, "otsu_thr": thr}
    if area < 200 or R is None:  # segmentation failure guard
        f["seg_ok"] = False
        return f
    f["seg_ok"] = True
    bg = ~ndi.binary_dilation(mask, iterations=15)
    refl = np.empty(8)
    means = np.empty(8); meds = np.empty(8); stds = np.empty(8); bgm = np.empty(8)
    for i, bn in enumerate(BAND_NAMES):
        v = cube[i][mask]
        means[i] = v.mean(); meds[i] = np.median(v); stds[i] = v.std()
        bgm[i] = cube[i][bg].mean() if bg.any() else np.nan
        f[f"mean_{bn}"] = means[i]; f[f"median_{bn}"] = meds[i]; f[f"std_{bn}"] = stds[i]
        f[f"bgmean_{bn}"] = bgm[i]
        refl[i] = R[i][mask].mean()
        f[f"refl_{bn}"] = refl[i]          # flat-fielded pseudo-reflectance (DN / fitted background field)
        f[f"reflstd_{bn}"] = R[i][mask].std()
        f[f"sat_frac_{bn}"] = float((cube[i][mask] >= 254).mean())
    for i, j in PAIRS:
        a, b = refl[i], refl[j]
        f[f"ND_{i+1}_{j+1}"] = (a - b) / (a + b) if (a + b) != 0 else np.nan
        f[f"SR_{i+1}_{j+1}"] = a / b if b != 0 else np.nan
    # spectral slope / curvature of band means over wavelength (nm)
    x = (WAVELENGTHS - WAVELENGTHS.mean()) / 100.0
    c1 = np.polyfit(x, refl, 1); c2 = np.polyfit(x, refl, 2)
    f["slope_per100nm"] = c1[0]; f["curvature"] = c2[0]
    c1r = np.polyfit(x, means, 1); f["slope_rawDN_per100nm"] = c1r[0]
    f["refl_mean"] = refl.mean(); f["spec_mean"] = means.mean()
    for i, bn in enumerate(BAND_NAMES):  # brightness-normalised spectrum (SNV-like, removes overall albedo)
        f[f"norm_{bn}"] = refl[i] / refl.mean()
    # shape of the mask
    ys, xs = np.nonzero(mask)
    f["mask_h"] = int(ys.max() - ys.min() + 1); f["mask_w"] = int(xs.max() - xs.min() + 1)
    f["mask_fill"] = area / (f["mask_h"] * f["mask_w"])
    return f


FEATURE_SETS = {
    "raw8": [f"mean_{bn}" for bn in BAND_NAMES],           # 8 raw band means (DN)
    "refl8": [f"refl_{bn}" for bn in BAND_NAMES],          # 8 flat-fielded band means
}
