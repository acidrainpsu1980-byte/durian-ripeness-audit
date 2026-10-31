#!/usr/bin/env python
"""Multispectral track entry point: sets up the namespace that run_all.py expects and executes it.

Run from work/multispectral/ after extract_all.py has written features_multispectral.parquet:

    python ../../src/multispectral/extract_all.py
    python ../../src/multispectral/ms_run_models.py

run_all.py (unchanged analysis code from the session) trains logreg / rf / lightgbm / svm_rbf / PLS-DA with
fruit-grouped 5-fold CV, writes oof_multispectral__<model>__fold_<split>.csv (+ metrics json) into ./results,
runs the fold_random leakage control for the two best models, leave-one-harvest-batch-out for the best model
(lobo_multispectral__logreg.csv), feature importances, PLS VIP scores and the PLSR dry-matter / TSS regressions.
"""
import os, sys, io, json, time, glob
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE, os.path.dirname(_HERE)]
import numpy as np, pandas as pd
import durian_common as dc
import ms_features as mf

os.makedirs("results", exist_ok=True)
df = dc.load_folds(dc.FOLDS_CSV, "multispectral")
print(df.shape, df.fruit_id.nunique(), "fruits", df.session_id.nunique(), "sessions")
for p in glob.glob("results/*"):           # run_all.py appends; start from a clean results folder as in the session
    os.remove(p)
exec(open(os.path.join(_HERE, "run_all.py")).read())
