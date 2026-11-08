"""Round-3 comment 2: align the primary metric (session macro-F1) across estimates and paired comparisons.

Input : results/v17/matched_space_session_oof.parquet (session-level outer-fold probabilities of every estimate of
        matched_space.py, 11 partitions: p0cpu = primary partition with the CPU knock CNN, p0-p9 with the GPU refits).
Output: $DURIAN_V19_OUT (default results/v19)
  estimates_by_partition.csv  per estimate x partition: fold-mean accuracy / macro-F1 (and SD over the 5 outer folds),
                              pooled out-of-fold accuracy / macro-F1, pooled class recall (U/R/O)
  estimates_summary.csv       per estimate: primary partition (fold-mean +- SD; pooled; recall) and ten partitions
                              (mean, SD, min, max of the partition-level values; partitions p0-p9)
  comparisons_macro_f1.csv    paired comparisons: primary-partition pooled differences in macro-F1 and accuracy with a
                              fruit-cluster bootstrap 95% CI and a fruit-level permutation test; the 10 x 5-fold
                              corrected resampled test (Nadeau-Bengio) on fold differences; partition-level differences
                              (mean, SD, min, max, partitions favouring the candidate); Holm across H1-H3, the primary
                              comparisons designated retrospectively during revision (role 'primary_retrospective'; all
                              comparisons are interpreted as exploratory, and Holm controls multiplicity within H1-H3 only)
Four kinds of summary are kept apart in every column name:
  fold-mean (mean over one partition's five outer folds), pooled (one value from all 268 out-of-fold predictions of one
  partition), across-partition (mean / SD / min-max of ten partition-level values; a range, not a confidence interval),
  and CI (confidence interval of a paired difference).
usage (repo root): python src/nested_selection/primary_metric.py
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats4

IN = os.environ.get("DURIAN_V17_IN", "results/v17") + "/matched_space_session_oof.parquet"
OUT = os.environ.get("DURIAN_V19_OUT", "results/v19"); os.makedirs(OUT, exist_ok=True)
# v2.2 (single manifest): the ten partitions are the primary CPU run of partition 0 (p0cpu) and the GPU refits of partitions 1-9;
# the GPU refit of partition 0 ("p0") is kept only as a hardware-reproducibility check.
# DURIAN_P0_PLATFORM=gpu recomputes every summary with the GPU refit of partition 0 (hardware check, results/v19/p0_gpu_check).
TEN = (["p0"] if os.environ.get("DURIAN_P0_PLATFORM", "cpu") == "gpu" else ["p0cpu"]) + [f"p{i}" for i in range(1, 10)]
CLS = ["Unripe", "Ripe", "Overripe"]
N_PERM, N_BOOT, SEED = 20000, 2000, 42

S = pd.read_parquet(IN)
S["pred"] = S[["p_Unripe", "p_Ripe", "p_Overripe"]].values.argmax(1)
G = {k: g.sort_values("session_id").reset_index(drop=True) for k, g in S.groupby(["partition", "estimate"], sort=False)}
ref = G[("p0cpu", "fixed_sound")]
for g in G.values(): assert (g.session_id.values == ref.session_id.values).all() and (g.y.values == ref.y.values).all()
y = ref.y.values.astype(int); fruit = ref.fruit_id.values.astype(int); n = len(y); assert n == 268

# ---------------------------------------------------------------- per estimate x partition
rows = []
for (p, e), g in G.items():
    pr, f = g.pred.values, g.outer_fold.values
    fa = [accuracy_score(y[f == k], pr[f == k]) for k in range(5)]
    ff = [f1_score(y[f == k], pr[f == k], average="macro") for k in range(5)]
    rc = recall_score(y, pr, average=None, labels=[0, 1, 2])
    rows.append(dict(partition=p, estimate=e, foldmean_acc=np.mean(fa), foldsd_acc=np.std(fa, ddof=1), foldmean_f1=np.mean(ff),
                     foldsd_f1=np.std(ff, ddof=1), pooled_acc=accuracy_score(y, pr), pooled_f1=f1_score(y, pr, average="macro"),
                     recall_U=rc[0], recall_R=rc[1], recall_O=rc[2], n_errors=int((pr != y).sum())))
B = pd.DataFrame(rows); B.to_csv(f"{OUT}/estimates_by_partition.csv", index=False)

# consistency with the shipped v17 summary (fold-mean accuracy / macro-F1 of the primary partition; ten-partition accuracy)
ms = pd.read_csv(os.path.dirname(IN) + "/matched_space_summary.csv", index_col=0)
P0 = B[B.partition == "p0cpu"].set_index("estimate"); T = B[B.partition.isin(TEN)]
for e in ms.index:
    assert abs(round(P0.loc[e, "foldmean_acc"], 3) - ms.loc[e, "acc"]) < 1e-9 and abs(round(P0.loc[e, "foldmean_f1"], 3) - ms.loc[e, "f1"]) < 1e-9, e
    t = T[T.estimate == e].foldmean_acc
    if os.environ.get("DURIAN_P0_PLATFORM", "cpu") == "cpu":   # matched_space_summary.csv is the CPU-partition-0 manifest
        assert abs(round(t.mean(), 3) - ms.loc[e, "mean"]) < 1e-9 and abs(round(t.min(), 3) - ms.loc[e, "min"]) < 1e-9 and abs(round(t.max(), 3) - ms.loc[e, "max"]) < 1e-9, e

# ---------------------------------------------------------------- per estimate summary
MET = ["foldmean_acc", "foldmean_f1", "pooled_acc", "pooled_f1", "recall_U", "recall_R", "recall_O"]
srow = []
for e, g in T.groupby("estimate", sort=False):
    r = dict(estimate=e)
    for m in MET + ["foldsd_acc", "foldsd_f1", "n_errors"]: r[f"primary_{m}"] = P0.loc[e, m]
    for m in MET:
        v = g[m].values; assert len(v) == 10
        r.update({f"ten_{m}_mean": v.mean(), f"ten_{m}_sd": v.std(ddof=1), f"ten_{m}_min": v.min(), f"ten_{m}_max": v.max()})
    srow.append(r)
SUM = pd.DataFrame(srow); SUM.to_csv(f"{OUT}/estimates_summary.csv", index=False)


# ---------------------------------------------------------------- paired comparisons
def macro_f1_rows(yt, P):
    """Macro-F1 of each row of the prediction matrix P (r x n); classes 0-2."""
    out = np.zeros(P.shape[0])
    for c in range(3):
        tp = ((P == c) & (yt == c)).sum(1); pp = (P == c).sum(1); tt = (yt == c).sum()
        out += np.where(pp + tt > 0, 2 * tp / np.maximum(pp + tt, 1), 0.0)
    return out / 3


def fruit_perm(yt, pr_ref, pr_cand, cl, stat, n_perm=N_PERM, seed=SEED):
    """Paired permutation test: the two arms' predictions are exchanged for all sessions of a fruit with probability 1/2."""
    T0 = stat(yt, pr_cand[None]) - stat(yt, pr_ref[None])
    u, inv = np.unique(cl, return_inverse=True); rng = np.random.default_rng(seed)
    sw = rng.integers(0, 2, size=(n_perm, len(u))).astype(bool)[:, inv]
    A = np.where(sw, pr_ref[None], pr_cand[None]); Bm = np.where(sw, pr_cand[None], pr_ref[None])
    Tp = stat(yt, A) - stat(yt, Bm)
    return float(T0[0]), float((np.sum(np.abs(Tp) >= abs(T0[0]) - 1e-12) + 1) / (n_perm + 1))


acc_rows = lambda yt, P: (P == yt).mean(1)
COMP = [("H1", "primary_retrospective", "fixed_thermal+sound+rgb|mean", "fixed_sound", "Thermal + sound + RGB mean vs sound (development pipelines)"),
        ("H2", "primary_retrospective", "fixed_thermal+multispectral+sound+rgb|stacking", "fixed_sound", "All four, nested stacking vs sound (development pipelines)"),
        ("H3", "primary_retrospective", "nested_cfg_nestedfam", "nested_single_nestedfam", "Nested-selected configuration vs nested-selected single sensor"),
        ("P1", "exploratory", "nestedfam_thermal+sound+rgb|mean", "nestedfam_sound", "Thermal + sound + RGB mean vs sound (selection procedures)"),
        ("P2", "exploratory", "nestedfam_thermal+multispectral+sound+rgb|stacking", "nestedfam_sound", "All four, nested stacking vs sound (selection procedures)"),
        ("P4", "exploratory", "nested_cfg_fixedfam", "fixed_sound", "Nested-selected configuration (families fixed) vs sound"),
        ("E1", "exploratory", "posthoc_C37", "fixed_sound", "Sound + RGB stacking (post hoc best) vs sound"),
        ("E2", "exploratory", "fixed_thermal+sound+rgb|stacking", "fixed_sound", "Thermal + sound + RGB stacking vs sound"),
        ("E3", "exploratory", "fixed_thermal+multispectral+sound+rgb|mean", "fixed_sound", "All four mean vs sound"),
        ("E4", "exploratory", "fixed_thermal+multispectral+sound+rgb|stacking", "fixed_thermal+sound+rgb|stacking", "All four vs thermal + sound + RGB (stacking)"),
        ("E5", "exploratory", "fixed_thermal", "fixed_sound", "Thermal vs sound"),
        ("E6", "exploratory", "fixed_rgb", "fixed_sound", "RGB vs sound"),
        ("E7", "exploratory", "fixed_multispectral", "fixed_sound", "Multispectral vs sound")]
TEST_FRAC = 0.25     # n_test / n_train for 5-fold CV, as in v1.4
crow = []
for cid, role, cand, refe, lab in COMP:
    a, b = G[("p0cpu", cand)], G[("p0cpu", refe)]
    pc, pr_ = a.pred.values, b.pred.values
    df1, p_f1 = fruit_perm(y, pr_, pc, fruit, macro_f1_rows)
    dacc, p_acc = fruit_perm(y, pr_, pc, fruit, acc_rows)
    bt = stats4.fruit_bootstrap(y, pr_, pc, fruit, n_boot=N_BOOT, seed=SEED)
    mc = stats4.clustered_mcnemar(pr_ == y, pc == y, fruit)
    fa = B[(B.partition == "p0cpu") & (B.estimate == cand)].iloc[0]; fb = B[(B.partition == "p0cpu") & (B.estimate == refe)].iloc[0]
    r = dict(id=cid, role=role, label=lab, candidate=cand, reference=refe,
             primary_foldmean_d_f1_pp=100 * (fa.foldmean_f1 - fb.foldmean_f1), primary_foldmean_d_acc_pp=100 * (fa.foldmean_acc - fb.foldmean_acc),
             primary_pooled_d_f1_pp=100 * df1, boot_d_f1_lo_pp=100 * bt["f1_ci"][0], boot_d_f1_hi_pp=100 * bt["f1_ci"][1], p_perm_fruit_f1=p_f1,
             primary_pooled_d_acc_pp=100 * dacc, boot_d_acc_lo_pp=100 * bt["acc_ci"][0], boot_d_acc_hi_pp=100 * bt["acc_ci"][1], p_perm_fruit_acc=p_acc,
             ref_only=mc["ref_only"], cand_only=mc["cand_only"], p_mcnemar_fruit_clustered=mc["p"])
    # corrected resampled test over the 10 partitions x 5 folds, and partition-level differences
    for m, key in [("f1", "f1"), ("acc", "acc")]:
        d50, dpart = [], []
        for p in TEN:
            A_, B_ = G[(p, cand)], G[(p, refe)]; f = A_.outer_fold.values
            fun = (lambda t, q: f1_score(t, q, average="macro")) if m == "f1" else accuracy_score
            dk = [fun(y[f == k], A_.pred.values[f == k]) - fun(y[f == k], B_.pred.values[f == k]) for k in range(5)]
            d50 += dk; dpart.append(np.mean(dk))
        nb = stats4.nb_corrected_t(np.array(d50), TEST_FRAC); dpart = np.array(dpart)
        r.update({f"nb_d_{key}_pp": 100 * nb["mean"], f"nb_{key}_ci_lo_pp": 100 * nb["ci"][0], f"nb_{key}_ci_hi_pp": 100 * nb["ci"][1], f"nb_p_{key}": nb["p"],
                  f"part_d_{key}_sd_pp": 100 * dpart.std(ddof=1), f"part_d_{key}_min_pp": 100 * dpart.min(), f"part_d_{key}_max_pp": 100 * dpart.max(),
                  f"partitions_favouring_cand_{key}": int((dpart > 0).sum()), f"partitions_tied_{key}": int((dpart == 0).sum())})
    crow.append(r)
C = pd.DataFrame(crow)
h = C.role == "primary_retrospective"
for col in ["p_perm_fruit_f1", "nb_p_f1", "p_perm_fruit_acc", "p_mcnemar_fruit_clustered", "nb_p_acc"]:
    C.loc[h, col + "_holm"] = stats4.holm(C.loc[h, col].values)
C.to_csv(f"{OUT}/comparisons_macro_f1.csv", index=False)
print(C[["id", "primary_pooled_d_f1_pp", "boot_d_f1_lo_pp", "boot_d_f1_hi_pp", "p_perm_fruit_f1", "nb_d_f1_pp", "nb_f1_ci_lo_pp", "nb_f1_ci_hi_pp", "nb_p_f1",
         "partitions_favouring_cand_f1"]].round(3).to_string())
