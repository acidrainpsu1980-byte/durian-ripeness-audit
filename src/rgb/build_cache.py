"""Cache pass: decode every RGB JPEG from the 20 GB zip, downscale to long side 512 px,
store in a uint8 memmap (N, 512, 512, 3) with zero padding (image centred) + manifest CSV.
Workers each open their own ZipFile handle. Run: python build_cache.py
"""
import io, os, sys, time, json
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
import numpy as np, pandas as pd
from multiprocessing import Pool
from PIL import Image, ImageOps
sys.path.insert(0, os.getcwd())
import durian_common as dc

S = 512
OUT_NPY = "cache/rgb512.npy"
OUT_MAN = "cache/rgb512_manifest.csv"
_zf = None

def _work(args):
    i, member = args
    global _zf
    if _zf is None:
        _zf = dc.open_zip("rgb")
    rec = dict(row=i, zip_member=member, ok=0, err="", orig_w=0, orig_h=0, h=0, w=0, y0=0, x0=0,
               camera="", bytes=0, exif_orient=0)
    try:
        b = dc.read_member(_zf, member)
        rec["bytes"] = len(b)
        im = Image.open(io.BytesIO(b))
        ex = im.getexif()
        rec["camera"] = str(ex.get(272, "")).strip()
        rec["exif_orient"] = int(ex.get(274, 0) or 0)
        rec["orig_w"], rec["orig_h"] = im.size
        im.draft("RGB", (S + 256, S + 256))          # fast DCT-domain downscale
        im = im.convert("RGB")
        im = ImageOps.exif_transpose(im)
        w, h = im.size
        sc = S / max(w, h)
        nw, nh = max(1, round(w * sc)), max(1, round(h * sc))
        im = im.resize((nw, nh), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.uint8)
        y0 = (S - nh) // 2; x0 = (S - nw) // 2
        canvas = np.zeros((S, S, 3), np.uint8)
        canvas[y0:y0 + nh, x0:x0 + nw] = arr
        rec.update(ok=1, h=nh, w=nw, y0=y0, x0=x0)
        return rec, canvas
    except Exception as e:
        rec["err"] = repr(e)[:200]
        return rec, None

if __name__ == "__main__":
    df = dc.load_folds(dc.FOLDS_CSV, "rgb")
    N = len(df)
    os.makedirs("cache", exist_ok=True)
    mm = np.lib.format.open_memmap(OUT_NPY, mode="w+", dtype=np.uint8, shape=(N, S, S, 3))
    recs = []
    t0 = time.time()
    with Pool(4) as pool:
        for n, (rec, canvas) in enumerate(pool.imap(_work, list(enumerate(df.zip_member)), chunksize=4)):
            if canvas is not None:
                mm[rec["row"]] = canvas
            recs.append(rec)
            if (n + 1) % 200 == 0:
                print(f"{n+1}/{N} {time.time()-t0:.0f}s", flush=True)
    mm.flush(); del mm
    man = pd.DataFrame(recs).sort_values("row").reset_index(drop=True)
    man = pd.concat([df[["modality", "file", "fruit_id", "session_id", "orientation", "date_captured",
                         "label_capture", "y", "Maturity_Stage", "Class", "fold_fruit", "fold_random"]].reset_index(drop=True),
                     man.drop(columns=["zip_member"])], axis=1)
    man.to_csv(OUT_MAN, index=False)
    print("done", time.time() - t0, "failures", int((man.ok == 0).sum()), flush=True)
    print(man.camera.value_counts(dropna=False).to_dict())
