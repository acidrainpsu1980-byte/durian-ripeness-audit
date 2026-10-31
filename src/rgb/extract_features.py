"""Hand-crafted colour + texture features from the 512-px cache. Writes features_rgb.parquet and masks npy."""
import os, sys, time
import numpy as np, pandas as pd, cv2
from multiprocessing import Pool
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
sys.path.insert(0, os.getcwd())

X = None; MAN = None

def seg_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    S_, V_ = hsv[..., 1], hsv[..., 2]
    m = ((S_ > 60) & (V_ > 30)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m_full = m.copy()
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((17, 17), np.uint8))  # cut stalks/tags so wood cannot attach
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    H, W = m.shape
    box = np.zeros_like(m); box[int(H * 0.2):int(H * 0.8), int(W * 0.25):int(W * 0.75)] = 1
    best, bs = 0, -1
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        inside = int(((lab == k) & (box == 1)).sum())
        if inside / max(1, a) < 0.25:   # wood frame / tags: mostly outside the central box
            continue
        if inside > bs:
            bs, best = inside, k
    if best == 0:
        return np.zeros_like(m)
    out = (lab == best).astype(np.uint8)
    for _ in range(10):  # limited geodesic reconstruction (~30 px): recover stalk base, not the wood frame
        out = cv2.dilate(out, np.ones((7, 7), np.uint8)) & m_full
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(out); cv2.drawContours(out, cnts, -1, 1, -1)
    return out

def feats(img, mask):
    f = {}
    px = mask.astype(bool)
    f["mask_frac"] = float(px.mean())
    if px.sum() < 200:
        return f, False
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    for name, arr in [("rgb", img), ("hsv", hsv), ("lab", lab)]:
        v = arr[px].astype(np.float32)
        for c in range(3):
            f[f"{name}{c}_mean"] = float(v[:, c].mean()); f[f"{name}{c}_std"] = float(v[:, c].std())
    # percentiles of hue/sat/value and a*/b*
    for name, arr, c in [("hue", hsv, 0), ("sat", hsv, 1), ("val", hsv, 2), ("a", lab, 1), ("b", lab, 2)]:
        v = arr[px][:, c].astype(np.float32)
        for q in (10, 50, 90):
            f[f"{name}_p{q}"] = float(np.percentile(v, q))
    # 16-bin hue histogram (OpenCV hue 0..179), weighted by saturation>40
    hue = hsv[px][:, 0]; sat = hsv[px][:, 1]
    hh, _ = np.histogram(hue[sat > 40], bins=16, range=(0, 180))
    hh = hh / max(1, hh.sum())
    for i in range(16):
        f[f"hue_hist{i}"] = float(hh[i])
    # colour-category fractions (OpenCV HSV: H 0-179, S,V 0-255)
    H_, S_, V_ = hsv[px][:, 0].astype(int), hsv[px][:, 1].astype(int), hsv[px][:, 2].astype(int)
    f["frac_green"] = float(((H_ >= 35) & (H_ <= 85) & (S_ > 50)).mean())
    f["frac_yellow"] = float(((H_ >= 20) & (H_ < 35) & (S_ > 60) & (V_ > 120)).mean())
    f["frac_brown"] = float(((H_ >= 5) & (H_ < 25) & (S_ > 60) & (V_ <= 150)).mean())
    f["frac_dark"] = float((V_ < 70).mean())
    # texture on grey fruit crop
    ys, xs = np.where(px)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1]
    mk = px[y0:y1, x0:x1]
    for P, R in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(g, P, R, method="uniform")
        hist, _ = np.histogram(lbp[mk], bins=P + 2, range=(0, P + 2))
        hist = hist / max(1, hist.sum())
        for i in range(P + 2):
            f[f"lbp{P}_{i}"] = float(hist[i])
    gq = (g // 8).astype(np.uint8)  # 32 grey levels
    gq[~mk] = 0
    glcm = graycomatrix(gq, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    glcm[0, :, :, :] = 0; glcm[:, 0, :, :] = 0  # drop background level
    s = glcm.sum(axis=(0, 1), keepdims=True); s[s == 0] = 1; glcm = glcm / s
    for prop in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]:
        v = graycoprops(glcm, prop)  # (dist, angle)
        for d in range(2):
            f[f"glcm_{prop}_d{[1,3][d]}"] = float(v[d].mean())
    f["grey_std_fruit"] = float(g[mk].std())
    lap = cv2.Laplacian(g, cv2.CV_32F)
    f["lap_var_fruit"] = float(lap[mk].var())
    # shape (camera-framing dependent; kept but flagged)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c); hull = cv2.convexHull(c)
    f["solidity"] = float(area / max(1, cv2.contourArea(hull)))
    (cx, cy), (w, h), ang = cv2.minAreaRect(c)
    f["aspect"] = float(max(w, h) / max(1e-3, min(w, h)))
    return f, True

def _work(i):
    global X, MAN
    if X is None:
        X = np.load("cache/rgb512.npy", mmap_mode="r"); MAN = pd.read_csv("cache/rgb512_manifest.csv")
    m = MAN.loc[i]
    img = np.ascontiguousarray(X[i][m.y0:m.y0 + m.h, m.x0:m.x0 + m.w])
    mask = seg_hsv(img)
    f, ok = feats(img, mask)
    f["seg_ok"] = int(ok)
    # store mask in 512 canvas at 1/4 resolution for later overlays
    mc = np.zeros((512, 512), np.uint8); mc[m.y0:m.y0 + m.h, m.x0:m.x0 + m.w] = mask
    return i, f, cv2.resize(mc, (128, 128), interpolation=cv2.INTER_NEAREST)

if __name__ == "__main__":
    man = pd.read_csv("cache/rgb512_manifest.csv")
    t0 = time.time(); rows = {}; masks = np.zeros((len(man), 128, 128), np.uint8)
    with Pool(4) as pool:
        for n, (i, f, mk) in enumerate(pool.imap_unordered(_work, range(len(man)), chunksize=8)):
            rows[i] = f; masks[i] = mk
            if (n + 1) % 400 == 0: print(n + 1, f"{time.time()-t0:.0f}s", flush=True)
    F = pd.DataFrame([rows[i] for i in range(len(man))])
    meta = man[["modality", "file", "fruit_id", "session_id", "orientation", "date_captured", "label_capture", "y",
                "Maturity_Stage", "Class", "fold_fruit", "fold_random", "camera"]]
    out = pd.concat([meta, F], axis=1)
    out.to_parquet("features_rgb.parquet", index=False)
    np.save("cache/rgb_masks128.npy", masks)
    print("done", time.time() - t0, out.shape, "seg failures", int((out.seg_ok == 0).sum()), flush=True)
