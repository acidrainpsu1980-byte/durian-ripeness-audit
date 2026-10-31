"""v1.8: build the audit tables in audit/ from shipped repository files only (no raw data needed).

usage:  python src/audit/build_audit_tables.py        (from the repository root)
reads:  data/raw_file_inventory.csv, data/durian_mm_fruit_table.csv, data/durian_mm_capture_folds.csv,
        data/session_labels_common.csv, data/sound_recording_manifest.csv, data/sound_session_evidence.csv,
        src/nested_selection/partitions.csv, results/session_table.csv
writes: audit/fruits.csv              189 fruits: analysis status, reason, fruit-linked captures per sensor, partitions p0-p9
        audit/sessions.csv            315 sessions of the 164 analysis fruits: sensors present, per-sensor file-name labels,
                                      common label and its basis, evaluation flag (268), fold in each partition p0-p9
        audit/fruit_partitions.csv    164 fruits x 10 stratified fruit partitions (p0 = the frozen primary split)
        audit/exclusion_ledger.csv    one row per excluded unit (archive member, fruit, session, recording) with reason + evidence
"""
import os
import numpy as np, pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
D = lambda f: os.path.join(REPO, f)
AOUT = os.environ.get("DURIAN_AUDIT_OUT", D("audit")); os.makedirs(AOUT, exist_ok=True)
A = lambda f: os.path.join(AOUT, f)
SENS = ["thermal", "multispectral", "sound", "rgb"]

inv = pd.read_csv(D("data/raw_file_inventory.csv"))
ft = pd.read_csv(D("data/durian_mm_fruit_table.csv"))
cap = pd.read_csv(D("data/durian_mm_capture_folds.csv"))
lab = pd.read_csv(D("data/session_labels_common.csv"))
man = pd.read_csv(D("data/sound_recording_manifest.csv"))
sev = pd.read_csv(D("data/sound_session_evidence.csv"))
part = pd.read_csv(D("src/nested_selection/partitions.csv"))
st = pd.read_csv(os.path.join(os.environ.get("DURIAN_RESULTS", D("results")), "session_table.csv"), low_memory=False)
PCOL = [f"p{i}" for i in range(10)]

# ------------------------------------------------------------------ fruits
linked = inv[inv.role.isin(["model_input", "fruit_not_in_analysis"]) & inv.fruit_id.notna()].copy()
linked = linked[~((linked.modality == "thermal") & linked.zip_member.str.lower().str.endswith(".jpg"))]   # count thermal captures once
cnt = linked.groupby(["fruit_id", "modality"]).size().unstack(fill_value=0).reindex(columns=SENS, fill_value=0)
fr = ft[["fruit_id", "Code", "Maturity_Stage", "Class", "actual", "excluded"]].merge(
    cnt.add_prefix("n_linked_").reset_index().astype({"fruit_id": int}), on="fruit_id", how="left").fillna(
    {f"n_linked_{m}": 0 for m in SENS})
in_set = set(part.fruit_id)
def fruit_reason(r):
    if r.fruit_id in in_set:
        return "analysis_set", ""
    if bool(r.excluded):
        if str(r.actual) == "Disease":
            return "excluded", "endpoint recorded as 'Disease' (no ripeness endpoint)"
        return "excluded", "no endpoint recorded in the fruit log (actual missing)"
    miss = [m for m in SENS if r[f"n_linked_{m}"] == 0]
    if len(miss) == 4:
        tags = __import__("re").findall(r"\(Tagak (\d+)\)", str(r.Code))
        if tags and (inv.zip_member.str.contains(rf"(?i)durian_tagak{tags[0]}_")).any():
            return "excluded", (f"no numbered capture in any sensor; thermal/RGB files named TAGAK{tags[0]} match its field tag but carry no fruit"
                                f" number, and no multispectral or knock file exists for it (would not qualify even if linked)")
        return "excluded", "no fruit-linked capture in any sensor"
    return "excluded", "no fruit-linked capture for: " + ", ".join(miss)
fr[["status", "reason"]] = fr.apply(lambda r: pd.Series(fruit_reason(r)), axis=1)
fr = fr.merge(part[["fruit_id", "stratum"] + PCOL], on="fruit_id", how="left")
fr.to_csv(A("fruits.csv"), index=False)
assert (fr.status == "analysis_set").sum() == 164

# ------------------------------------------------------------------ sessions
s = lab.copy()
s["label_basis"] = np.where(s.unanimous, "unanimous file-name label across sensors", s.label_rule)
st_ = st.set_index("session_id")
for m in SENS:
    s[f"n_{m}"] = s.session_id.map(st_[f"n_captures__{m}"]).fillna(0).astype(int)
s["sensors_present"] = s[[f"n_{m}" for m in SENS]].gt(0).sum(axis=1)
s["in_evaluation_268"] = s.session_id.map(st_.in_main_analysis.astype(str) == "True").fillna(False)
s["sound_session_evidence"] = s.session_id.map(sev.set_index("session_id").session_evidence)
s = s.merge(part[["fruit_id"] + PCOL], on="fruit_id", how="left").rename(columns={p: f"fold_{p}" for p in PCOL})
keep = ["session_id", "fruit_id", "date", "days_from_recorded_harvest"] + [f"label_{m}" for m in SENS] + \
       ["label_common", "label_basis", "unanimous", "log_inconsistent", "Actual_Ripening_Status", "Actual_Ripening_Days",
        "Actual_Days_Capsule_Splitting"] + [f"n_{m}" for m in SENS] + ["sensors_present", "in_evaluation_268", "sound_session_evidence"] + \
       [f"fold_{p}" for p in PCOL]
s[keep].to_csv(A("sessions.csv"), index=False)
assert s.in_evaluation_268.sum() == 268 and len(s) == 315

part.to_csv(A("fruit_partitions.csv"), index=False)

# ------------------------------------------------------------------ exclusion ledger
L = []
fr_reason = fr.set_index("fruit_id").reason
# field tags: the fruit log gives e.g. "OM_CA_RI_135 (Tagak 3)"; archive files named durian_TAGAK3_* carry no fruit number
TAG = {int(n): (int(r.fruit_id), r.Code) for r in ft.itertuples() for n in __import__("re").findall(r"\(Tagak (\d+)\)", str(r.Code))}
def tag_note(member):
    mm = __import__("re").search(r"(?i)durian_tagak(\d*)_", member)
    if not mm:
        return ""
    if not mm.group(1):
        return " Name 'TAGAK' has no tag number, so it cannot be matched to a fruit-log entry."
    fid, code = TAG.get(int(mm.group(1)), (None, None))
    if fid is None:
        return f" Tag {mm.group(1)} has no fruit-log entry."
    if fid in in_set:
        return (f" Name matches field tag of fruit {fid} ({code}), which is in the analysis set through its numbered files; these"
                f" harvest-time captures were not linked (disclosed; they would add harvest-day thermal/RGB images only).")
    return (f" Name matches field tag of fruit {fid} ({code}); even if linked, the fruit has no multispectral or knock capture"
            f" and would not qualify for the four-sensor analysis set.")
for r in inv[inv.role != "model_input"].itertuples():
    if r.role == "paired_preview":
        continue                                     # not an exclusion: rendering of a used thermal CSV
    ev = {"macos_resource_fork": "path starts with __MACOSX/ (AppleDouble metadata written by the archiver; no pixel data)",
          "archive_metadata": "metadata table inside the archive; read for provenance",
          "tree_code_unlinked": "file name carries a field tree code instead of a fruit number; archive metadata has no fruit id for it.",
          "fruit_not_in_analysis": f"fruit {r.fruit_id}: " + (fr_reason.get(int(r.fruit_id), "") if pd.notna(r.fruit_id) else "")}[r.role]
    if r.role == "tree_code_unlinked":
        ev += tag_note(r.zip_member)
    L.append(dict(level="archive_member", unit_id=r.zip_member, modality=r.modality, fruit_id=r.fruit_id, session_id="",
                  reason_code=r.role, evidence=ev, evidence_source="data/raw_file_inventory.csv (size, crc32)", effect="not read"))
for r in fr[fr.status == "excluded"].itertuples():
    L.append(dict(level="fruit", unit_id=str(r.fruit_id), modality="all", fruit_id=r.fruit_id, session_id="",
                  reason_code="fruit_excluded", evidence=f"{r.Code}: {r.reason}; fruit-linked captures T/MS/S/RGB = "
                  f"{r.n_linked_thermal}/{r.n_linked_multispectral}/{r.n_linked_sound}/{r.n_linked_rgb}",
                  evidence_source="audit/fruits.csv; data/durian_mm_fruit_table.csv", effect="fruit not in any split"))
for r in s[~s.in_evaluation_268].itertuples():
    miss = [m for m in SENS if getattr(r, f"n_{m}") == 0]
    L.append(dict(level="session", unit_id=r.session_id, modality=",".join(miss), fruit_id=r.fruit_id, session_id=r.session_id,
                  reason_code="session_missing_sensor", evidence="no capture for: " + ", ".join(miss),
                  evidence_source="audit/sessions.csv (n_<sensor>)",
                  effect="excluded from the 268-session evaluation set; its captures remain in training folds of other test folds"))
mw = man[man.recording_status == "wrong_session"]
for r in mw.itertuples():
    L.append(dict(level="recording", unit_id=r.file, modality="sound", fruit_id=r.fruit_id, session_id=r.session_id,
                  reason_code="recording_from_other_session",
                  evidence=f"{r.matching_evidence} | md5 {r.md5} | timestamp {r.timestamp_local} | identical_to {r.identical_to}",
                  evidence_source="data/sound_recording_manifest.csv (Table S7); data/sound_session_evidence.csv (Table S8)",
                  effect="never trained on or evaluated; listed in data/sound_exclude_wrong_session.txt"))
led = pd.DataFrame(L)
led["fruit_id"] = led.fruit_id.astype("Int64")
led.to_csv(A("exclusion_ledger.csv"), index=False)
excl_txt = set(open(D("data/sound_exclude_wrong_session.txt")).read().split())
assert set(mw.file) == excl_txt and len(excl_txt) == 106
print(led.groupby(["level", "reason_code"]).size().to_string())
print("fruits:", fr.status.value_counts().to_dict(), "| sessions:", len(s), "evaluation:", int(s.in_evaluation_268.sum()))
