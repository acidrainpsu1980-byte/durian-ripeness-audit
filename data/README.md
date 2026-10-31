# data/ - frozen split tables (use everywhere; never re-split)

All modelling in the paper used the fruit-grouped 5-fold split frozen here (seed 42).  A *session* is one
fruit on one capture date; every fruit's sessions share a fold, so no fruit appears in both training and test.

| file | size | sha256 |
|---|---|---|
| `durian_mm_capture_folds.csv` | 1371.2 kB | `aa9677e10583a14b073bb06269da73bdc354902c195d09ef2c66f4bc3fb5086b` |
| `durian_mm_folds_fruit_level.csv` | 10.0 kB | `5651100d5dfca09e478480d1597cedb5cbb353930554b200eb7dc9573e9555db` |
| `durian_mm_fruit_table.csv` | 15.8 kB | `0ac3b69ab1dbcfbd01cd39a7c179d638160c9512188cc8eeb18008698c7504a8` |
| `durian_mm_sound_recordings.csv` | 90.3 kB | `2ff818f9e04323a75be880697e6e9e57dfcf707587067796430cebd8231f8e9e` |

- `durian_mm_capture_folds.csv` (v2) - one row per capture (thermal CSV/JPG, multispectral TIFF, RGB JPEG) or per
  sound recording, with `zip_member` (path inside the Zenodo archive), `fruit_id`, `orientation`, `device`,
  `date_captured`, `label_capture` (state of the fruit on the day of capture: Unripe / Ripe / Overripe),
  `actual` (fruit's endpoint), `Maturity_Stage` (= harvest batch), `Class` (size grade), `fold_fruit`
  (primary, fruit-grouped) and `fold_random` (capture-level control split; use only for the leakage comparison).
  Rows per modality:

```
               rows  files  fruits
modality                          
multispectral  2222   2222     164
rgb            1999   1999     164
sound           538    538     164
thermal        3342   3342     164
```

- `durian_mm_sound_recordings.csv` - the 538 knock recordings with session dates matched to the image modalities
  (`date_src`: 524 matched by label, 14 endpoint fallback).  THIS is the table the sound track and the fusion
  stage used; the sound rows of the capture table carry the earlier date assignment (see CHANGELOG).
- `durian_mm_fruit_table.csv` - one row per fruit (189 rows): design cells, harvest / endpoint dates, destructive
  references (dry matter, TSS, firmness), coverage per sensor, exclusion flag and `fold_fruit`.
- `durian_mm_folds_fruit_level.csv` - fruit -> fold lookup (164 rows).

Files added after v1.0 (see CHANGELOG):

- `session_labels_common.csv` (Table S10, v1.3) - the common session label with the rule applied; `label_v12_majority` keeps the
  earlier majority file-name label.
- `sound_recording_manifest.csv` (Table S7), `sound_session_evidence.csv` (Table S8), `sound_exclude_wrong_session.txt` (v1.2) -
  knock-recording provenance and the 106 excluded Android recordings.
- `raw_file_inventory.csv` (v1.8) - every member of the four Zenodo archives (size, CRC-32, parsed fruit id, role); built by
  `src/audit/inventory_raw.py`.
- `zenodo_checksums.md5` (v1.8) - MD5 of the Zenodo files, as distributed with the record.
- `zenodo/durian_characteristics_cleaned.csv` (v1.8) - the fruit log of the Zenodo record, copied unchanged (CC BY 4.0, attribution:
  Mesa-Satina et al. 2026, doi:10.5281/zenodo.18603796); fruit_id = row number (1-189).
- `literature/TableS18_literature_design_extraction.csv` (v1.7) - curated design extraction of the 21 cited durian studies.

The fruit partitions p0-p9 are in `../src/nested_selection/partitions.csv` (copy: `../audit/fruit_partitions.csv`).

Raw sensor data are not included (CC BY 4.0, University of the Philippines Mindanao); see the top-level README
for download instructions and checksums.
