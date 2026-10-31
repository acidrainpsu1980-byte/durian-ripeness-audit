import os, sys, time
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
import numpy as np, pandas as pd
import durian_common as dc
from multiprocessing import get_context

def work(member):
    import thermal_features as tf   # each spawned worker opens its own zip handle lazily
    return tf.process(member)

if __name__ == "__main__":
    df = dc.load_folds(dc.FOLDS_CSV, "thermal")
    df = df[df.file.str.endswith(".csv") & ~df.zip_member.str.startswith("__MACOSX")].reset_index(drop=True)
    t = time.time()
    with get_context("spawn").Pool(4) as pool:
        out = []
        for i, o in enumerate(pool.imap(work, df.zip_member.tolist(), chunksize=10)):
            out.append(o)
            if i % 400 == 0:
                print(i, round(time.time() - t, 1), flush=True)
    F = pd.DataFrame([o[0] for o in out])
    imgs = np.stack([o[1] if o[1] is not None else np.zeros((128, 128), np.float32) for o in out])
    assert (F.zip_member.values == df.zip_member.values).all()
    F.to_parquet("thermal_feat_raw.parquet", index=False)
    np.save("thermal_imgs128.npy", imgs)
    print("done", round(time.time() - t, 1), F.shape, imgs.shape)
