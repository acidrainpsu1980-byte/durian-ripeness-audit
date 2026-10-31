import sys, os, time, numpy as np, pandas as pd
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
sys.path.insert(0, os.getcwd())
import durian_common as dc, ms_features as mf, io
df = dc.load_folds(dc.FOLDS_CSV, "multispectral")
zf = dc.open_zip("multispectral")
meta = pd.read_csv(io.BytesIO(dc.read_member(zf, "dataset_clean_multispectral/metadata.csv")))
cleaned = dict(zip(meta.new_filename, meta.cleaned))
recs = []; t0 = time.time()
for i, member in enumerate(df.zip_member.tolist()):
    try:
        cube = mf.read_montage(dc.read_member(zf, member))
        f = mf.features_from_cube(cube); f["read_ok"] = True; f["err"] = ""
    except Exception as e:
        f = {"read_ok": False, "seg_ok": False, "err": repr(e)[:200]}
    recs.append(f)
    if i % 400 == 0: print(i, round(time.time()-t0,1), flush=True)
F = pd.DataFrame(recs)
out = pd.concat([df[["modality","file","zip_member","fruit_id","session_id","orientation","device","date_captured",
                     "label_capture","y","actual","Maturity_Stage","Class","fold_fruit","fold_random"]].reset_index(drop=True), F], axis=1)
out["cleaned_flag"] = out.file.map(cleaned)
out.to_parquet("features_multispectral.parquet", index=False)
print("done", out.shape, out.read_ok.sum(), out.seg_ok.sum(), flush=True)
