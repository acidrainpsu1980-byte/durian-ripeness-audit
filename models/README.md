# Fitted models (primary partition p0)

One fitted model per outer fold (k = 0..4) for each of the four development pipelines of the paper (families fixed in v1.3):

| Files | Model | Input |
|---|---|---|
| `thermal__svm_rbf__outer{k}.joblib` | StandardScaler > SVC (RBF, probability, balanced), C and gamma chosen by the inner search stored in the object | 67 thermal features (`intermediate_common/features/features_thermal.parquet`, columns as in `src/nested_selection/cands.py`) |
| `multispectral__logreg__outer{k}.joblib` | median imputer > StandardScaler > multinomial logistic regression (balanced) | 121 multispectral features |
| `rgb__efficientnetb0_probe__outer{k}.joblib` | StandardScaler > logistic regression (balanced) | view-averaged EfficientNet-B0 embedding (1,280) of the fruit-centred crop |
| `cnn_common_outer_{k}.pt` | knock CNN `state_dict` (architecture in `src/common_label/cnn_common.py`, class `Net`), with `mel_mean`, `mel_std`, epochs, seed | per-knock 64 x 31 log-mel spectrogram; session probability = mean over knocks, then over recordings |

Each `.joblib` is a dict: `model` (the fitted `GridSearchCV`; `best_params_`, `cv_results_` available), `sensor`, `candidate`,
`feature_set`, `partition`, `outer_fold`, `train_files` (the captures it was trained on) and `classes` (Unripe, Ripe, Overripe).
`model_check.csv` records, for every file, the max |diff| between its predictions on the held-out fold and the shipped
out-of-fold predictions (0 for all 20) and its SHA-256.

These are the models behind the out-of-fold predictions; they are not a deployable classifier. Every model was trained on four
of five folds of one farm, one season and one cultivar, and the paper shows that none transfers across harvest batches.

Regenerate: `python src/audit/export_models.py fit` (classical models) and `CNN=1 ./run_all.sh` (CNNs); check:
`DURIAN_MELS=<work>/sound/handoff/mels.npy python src/audit/export_models.py check`.
