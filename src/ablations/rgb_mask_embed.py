"""Comment-6 RGB ablation: embed fruit-only and context-only crops with EfficientNet-B0 (ImageNet).
Same crop geometry and 5 deterministic views as src/rgb/rgb_models.py (stage 'embed').
fruit_only  : pixels outside the (eroded) fruit mask set to ImageNet-mean grey -> stalk, tags, background, frame borders removed
context_only: pixels inside the (dilated) fruit mask set to grey -> only stalk/tags/background/border remain
original    : unmodified crop (sanity check against stored emb_efficientnetb0.npy)
"""
import os, sys, time, numpy as np, pandas as pd, cv2, torch, torch.nn.functional as Fnn, torchvision.transforms.functional as TF
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
X = np.load(sys.argv[1], mmap_mode="r"); masks = np.load(sys.argv[2]); man = pd.read_csv(sys.argv[3]); OUT = sys.argv[4]; MODE = sys.argv[5] if len(sys.argv) > 5 else "all"
os.makedirs(OUT, exist_ok=True); torch.set_num_threads(8); torch.manual_seed(42)
GREY = np.array([124, 116, 104], np.uint8)
K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

def crop_box(i, margin=0.12):
    m = man.loc[i]; mk = cv2.resize(masks[i].astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mk); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2; side = max(y1 - y0, x1 - x0) * (1 + 2 * margin); side = max(side, 96)
    iy0, iy1, ix0, ix1 = m.y0, m.y0 + m.h, m.x0, m.x0 + m.w
    side = min(side, m.h, m.w)
    cy = np.clip(cy, iy0 + side / 2, iy1 - side / 2); cx = np.clip(cx, ix0 + side / 2, ix1 - side / 2)
    a0, b0 = int(round(cy - side / 2)), int(round(cx - side / 2)); s = int(round(side))
    return mk, a0, b0, s

def variants(i, out=224):
    mk, a0, b0, s = crop_box(i); img = np.array(X[i])
    inner = cv2.erode(mk, K) > 0; outer = cv2.dilate(mk, K) > 0
    if MODE == "body":   # stalk removed: opening with a 41-px disc (wider than the stalk), largest component, then the same erosion
        op = cv2.morphologyEx(mk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(op)
        if n > 1: op = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        inner = cv2.erode(op, K) > 0
    fo = img.copy(); fo[~inner] = GREY
    co = img.copy(); co[outer] = GREY
    r = lambda a: cv2.resize(np.ascontiguousarray(a[a0:a0 + s, b0:b0 + s]), (out, out), interpolation=cv2.INTER_AREA)
    return r(img), r(fo), r(co), inner[a0:a0 + s, b0:b0 + s].mean()

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def views(b):
    x = torch.from_numpy(b).permute(0, 3, 1, 2).float() / 255.
    vs = [x, TF.hflip(x), TF.vflip(x), TF.rotate(x, 12, interpolation=TF.InterpolationMode.BILINEAR), TF.rotate(x, -12, interpolation=TF.InterpolationMode.BILINEAR)]
    return [(v - MEAN) / STD for v in vs]

t = time.time(); V = [variants(i) for i in range(len(man))]
C = {nm: np.stack([v[j] for v in V]) for j, nm in enumerate(["original", "fruit_only", "context_only"])}
if MODE == "body": C = {"fruit_body": C["fruit_only"]}
np.save(f"{OUT}/mask_frac_in_crop_{MODE}.npy", np.array([v[3] for v in V]))
for nm in C: np.save(f"{OUT}/crops224_{nm}_examples.npy", C[nm][::60])
print("crops", len(man), f"{time.time()-t:.0f}s", flush=True)
net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).eval()
with torch.no_grad():
    for nm, A in C.items():
        t = time.time(); E = []
        for s in range(0, len(A), 32):
            E.append(torch.stack([Fnn.adaptive_avg_pool2d(net.features(v), 1).flatten(1) for v in views(A[s:s + 32])], 1))
        E = torch.cat(E).numpy(); np.save(f"{OUT}/emb_effb0_{nm}.npy", E); print(nm, E.shape, f"{time.time()-t:.0f}s", flush=True)
