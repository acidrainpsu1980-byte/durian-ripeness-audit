"""v1.8: inventory of every member of the four Zenodo archives, with its role in the analysis.

usage:  DURIAN_DATA_ROOT=/path/to/zips python src/audit/inventory_raw.py          (from the repository root)
writes: data/raw_file_inventory.csv  (one row per archive member; shipped, so the audit tables can be rebuilt without raw data)

Columns: modality, zip_member, size_bytes, crc32 (from the zip central directory), fruit_id (parsed from the file name;
empty for tree-code names), role.  role is one of
  model_input              capture used to train / evaluate models (thermal .csv, multispectral .tif, RGB .jpg, knock audio)
  paired_preview           thermal .jpg rendering of a used thermal .csv (indexed in the capture table, not a model input)
  macos_resource_fork      __MACOSX/ AppleDouble copy created by the archiver (no image data)
  archive_metadata         metadata.csv shipped inside the archive (read for provenance, not a capture)
  tree_code_unlinked       capture named with a field tree code (e.g. AT9B3-C5); the archive metadata gives no fruit id
  fruit_not_in_analysis    capture of a fruit outside the 164-fruit analysis set (reason in audit/fruits.csv)
"""
import os, re, sys, zipfile
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = os.environ.get("DURIAN_DATA_ROOT", os.path.join(REPO, "raw"))
cap = pd.read_csv(f"{REPO}/data/durian_mm_capture_folds.csv")
f164 = set(pd.read_csv(f"{REPO}/data/durian_mm_folds_fruit_level.csv").fruit_id.astype(int))
rows = []
for mod in ["thermal", "multispectral", "sound", "rgb"]:
    used = set(cap.loc[cap.modality == mod, "zip_member"])
    with zipfile.ZipFile(os.path.join(ROOT, f"dataset_clean_{mod}.zip")) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
    for i in infos:
        n = i.filename; b = os.path.basename(n); mm = re.match(r"durian_(\d{3})_", b)
        fid = int(mm.group(1)) if mm else None
        if n.startswith("__MACOSX"):
            role = "macos_resource_fork"
        elif b == "metadata.csv":
            role = "archive_metadata"
        elif n in used:
            role = "paired_preview" if (mod == "thermal" and b.lower().endswith(".jpg")) else "model_input"
        elif fid is None:
            role = "tree_code_unlinked"
        elif fid not in f164:
            role = "fruit_not_in_analysis"
        else:
            sys.exit(f"unclassified member {n}")
        rows.append(dict(modality=mod, zip_member=n, size_bytes=i.file_size, crc32=f"{i.CRC:08x}",
                         fruit_id=fid if not n.startswith("__MACOSX") else None, role=role))
inv = pd.DataFrame(rows)
inv["fruit_id"] = inv.fruit_id.astype("Int64")
inv.to_csv(f"{REPO}/data/raw_file_inventory.csv", index=False)
print(inv.groupby(["modality", "role"]).size().unstack(fill_value=0).to_string())
