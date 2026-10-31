"""Shared harness for the durian multimodal ripeness project.

Every modality track must use these functions so results are comparable.

Conventions
-----------
* Target: `label_capture` (state of the fruit on the day of capture), 3 classes.
* Primary split: `fold_fruit` (fruit-grouped, 5 folds, frozen in durian_mm_capture_folds.csv).
* Control split: `fold_random` (capture-level, leaks fruit identity) - run ONLY for the leakage comparison.
* Report per-fold metrics at capture level AND fruit level (fruit = fruit_id x date_captured session,
  aggregated by mean class probability).
* Save predictions with `save_predictions` so fusion can be done later from out-of-fold probabilities.
"""
import json, os, zipfile
import numpy as np
import pandas as pd

CLASSES = ["Unripe", "Ripe", "Overripe"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
# ---------------------------------------------------------------------------------------------
# Paths.  Raw data (the four dataset_clean_<modality>.zip archives from Zenodo 10.5281/zenodo.18603796)
# are located through the environment variable DURIAN_DATA_ROOT; the frozen split tables ship with
# the repository under data/ and are resolved relative to this file.
# ---------------------------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
FOLDS_CSV = os.path.join(DATA_DIR, "durian_mm_capture_folds.csv")            # v2: 538 sound rows (one per recording)
FRUIT_TABLE_CSV = os.path.join(DATA_DIR, "durian_mm_fruit_table.csv")
FRUIT_FOLDS_CSV = os.path.join(DATA_DIR, "durian_mm_folds_fruit_level.csv")
SOUND_RECORDINGS_CSV = os.path.join(DATA_DIR, "durian_mm_sound_recordings.csv")  # 538 recordings, session dates matched to the image modalities
DATA_ROOT = os.environ.get("DURIAN_DATA_ROOT", os.path.join(REPO_ROOT, "raw"))
ZIPS = {m: os.path.join(DATA_ROOT, f"dataset_clean_{m}.zip") for m in ["sound", "thermal", "multispectral", "rgb"]}


def load_folds(capture_folds_csv=None, modality=None):
    """Return the frozen capture-level fold table, optionally filtered to one modality.

    capture_folds_csv defaults to data/durian_mm_capture_folds.csv (v2).  For the sound track use
    SOUND_RECORDINGS_CSV (one row per recording, session dates aligned with the image modalities)."""
    df = pd.read_csv(capture_folds_csv or FOLDS_CSV)
    if modality is not None:
        df = df[df.modality == modality].reset_index(drop=True)
    df["y"] = df.label_capture.map(CLASS_IDX)
    df["session_id"] = df.fruit_id.astype(int).astype(str) + "_" + df.date_captured.astype(str)
    assert df.y.notna().all(), "unknown label in capture table"
    return df


def open_zip(modality):
    p = ZIPS[modality]
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found. Download dataset_clean_{modality}.zip from Zenodo 10.5281/zenodo.18603796 "
                                "and point DURIAN_DATA_ROOT at the folder containing the four zip archives.")
    return zipfile.ZipFile(p)


def read_member(zf, member):
    """Bytes of one zip member (works without extracting the 20 GB RGB archive)."""
    with zf.open(member) as fh:
        return fh.read()


def iter_folds(df, split="fold_fruit", n_folds=5):
    """Yield (k, train_idx, test_idx) over the chosen split column."""
    for k in range(n_folds):
        te = np.flatnonzero(df[split].values == k)
        tr = np.flatnonzero(df[split].values != k)
        yield k, tr, te


def metrics_from_probs(y_true, proba):
    """accuracy, macro-F1, per-class recall, confusion matrix from class probabilities."""
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
    y_pred = np.asarray(proba).argmax(1)
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        recall_per_class=recall_score(y_true, y_pred, average=None, labels=[0, 1, 2]).round(4).tolist(),
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        n=int(len(y_true)),
    )


def aggregate_to_session(df_test, proba):
    """Mean probability per (fruit_id, date_captured) session; returns (y_session, proba_session, session_ids)."""
    tmp = pd.DataFrame(proba, columns=CLASSES)
    tmp["session_id"] = df_test.session_id.values
    tmp["y"] = df_test.y.values
    g = tmp.groupby("session_id")
    P = g[CLASSES].mean().values
    y = g.y.first().values
    return y, P, g.size().index.values


def evaluate_oof(df, oof_proba, split):
    """Per-fold + pooled metrics at capture level and session (fruit) level.

    oof_proba: array (len(df), 3) of out-of-fold probabilities for the given split.
    Returns a dict ready for json.dump.
    """
    rows = []
    for k, _, te in iter_folds(df, split):
        sub = df.iloc[te]
        cap_m = metrics_from_probs(sub.y.values, oof_proba[te])
        ys, Ps, _ = aggregate_to_session(sub, oof_proba[te])
        ses_m = metrics_from_probs(ys, Ps)
        rows.append(dict(fold=k, level="capture", **cap_m))
        rows.append(dict(fold=k, level="session", **ses_m))
    per_fold = pd.DataFrame(rows)
    summary = {}
    for lvl in ["capture", "session"]:
        d = per_fold[per_fold.level == lvl]
        summary[lvl] = dict(
            accuracy_mean=float(d.accuracy.mean()), accuracy_sd=float(d.accuracy.std(ddof=1)),
            macro_f1_mean=float(d.macro_f1.mean()), macro_f1_sd=float(d.macro_f1.std(ddof=1)),
            accuracy_ci95=[float(d.accuracy.mean() - 2.776 * d.accuracy.std(ddof=1) / np.sqrt(5)),
                           float(d.accuracy.mean() + 2.776 * d.accuracy.std(ddof=1) / np.sqrt(5))],
        )
    pooled = metrics_from_probs(df.y.values, oof_proba)
    return dict(split=split, per_fold=rows, summary=summary, pooled_capture=pooled)


def save_predictions(df, oof_proba, split, modality, model_name, out_dir="results"):
    """Write out-of-fold probabilities (one row per capture) + metrics json. Returns the two paths."""
    os.makedirs(out_dir, exist_ok=True)
    pred = df[["modality", "file", "fruit_id", "session_id", "orientation", "device", "date_captured",
               "label_capture", "y", "Maturity_Stage", "Class", split]].copy()
    for i, c in enumerate(CLASSES):
        pred[f"p_{c}"] = oof_proba[:, i]
    pred["model"] = model_name
    stem = f"{modality}__{model_name}__{split}"
    p1 = os.path.join(out_dir, f"oof_{stem}.csv"); pred.to_csv(p1, index=False)
    p2 = os.path.join(out_dir, f"metrics_{stem}.json")
    with open(p2, "w") as fh:
        json.dump(evaluate_oof(df, oof_proba, split), fh, indent=1)
    return p1, p2


def leave_one_batch_out(df, batch_col="Maturity_Stage"):
    """Yield (held_out_batch, train_idx, test_idx) for the harvest-batch confound check."""
    for b in sorted(df[batch_col].unique()):
        te = np.flatnonzero(df[batch_col].values == b)
        tr = np.flatnonzero(df[batch_col].values != b)
        yield b, tr, te
