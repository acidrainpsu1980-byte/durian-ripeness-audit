"""v1.8: assemble the inputs of the knock-CNN scripts (cnn_common.py, nested_selection/cnn_part.py, ablations/cnn_dev.py)
in a work directory, from the repository data files and the output of `sound_pipeline.py features`.

usage (from the work directory in which `python <repo>/src/sound/sound_pipeline.py features` was run):
    python <repo>/src/common_label/prepare_sound_cnn_inputs.py

Reads   ./features_sound_knocks.parquet, ./handoff/mels.npy, ./handoff/rec2_meta.parquet   (sound_pipeline.py features)
        <repo>/data/sound_recording_manifest.csv, session_labels_common.csv, sound_exclude_wrong_session.txt
        <repo>/src/nested_selection/partitions.csv
Writes  ./handoff/exclude_recordings.txt     the 106 wrong-session Android recordings (never trained on)
        ./handoff/file_label_common.csv      recording -> common session label (0 Unripe, 1 Ripe, 2 Overripe)
        ./mels.npy, ./rec2_meta.parquet, ./partitions.csv, ./exclude_recordings.txt, ./file_label_common.csv
                                             (flat copies; layout expected by cnn_part.py and cnn_dev.py)
"""
import os, shutil
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CI = {"Unripe": 0, "Ripe": 1, "Overripe": 2}
for f in ["features_sound_knocks.parquet", "handoff/mels.npy", "handoff/rec2_meta.parquet"]:
    assert os.path.exists(f), f"{f} missing: run sound_pipeline.py features in this directory first"

man = pd.read_csv(f"{REPO}/data/sound_recording_manifest.csv")
lab = pd.read_csv(f"{REPO}/data/session_labels_common.csv").set_index("session_id").label_common
rec2 = pd.read_parquet("handoff/rec2_meta.parquet")
m = man.set_index("file").reindex(rec2.file)
y = m.session_id.map(lab).map(CI)
assert y.notna().all(), f"{int(y.isna().sum())} recordings without a common session label"
# column name kept as in the original hand-off file (it holds the integer class, not a session id)
pd.DataFrame({"file": rec2.file.values, "session_id": y.astype(int).values}).to_csv("handoff/file_label_common.csv", index=False)
shutil.copy(f"{REPO}/data/sound_exclude_wrong_session.txt", "handoff/exclude_recordings.txt")
for src, dst in [("handoff/mels.npy", "mels.npy"), ("handoff/rec2_meta.parquet", "rec2_meta.parquet"),
                 ("handoff/file_label_common.csv", "file_label_common.csv"), ("handoff/exclude_recordings.txt", "exclude_recordings.txt"),
                 (f"{REPO}/src/nested_selection/partitions.csv", "partitions.csv")]:
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)
n_ex = len(open("handoff/exclude_recordings.txt").read().split())
print(f"recordings {len(rec2)} | excluded from training {n_ex} | label counts {y.value_counts().sort_index().to_dict()}")
