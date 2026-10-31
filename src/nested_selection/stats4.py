"""Dependence-aware comparisons of two classifiers' session predictions (v1.4, reviewer comment 4).

clustered_mcnemar : McNemar test for clustered matched pairs (Durkalski et al., Stat Med 2003;22:2417), clusters = fruits.
fruit_signflip    : paired permutation test exchanging the two configurations within each fruit (sign flip of the
                    per-fruit difference in correctly classified sessions); exact when <= 20 informative fruits.
fruit_bootstrap   : fruit-cluster bootstrap of the pooled accuracy / macro-F1 difference (conditional on the fitted
                    out-of-fold predictions: it does not include model-fitting or model-selection variability).
nb_corrected_t    : corrected resampled t-test of Nadeau & Bengio (Mach Learn 2003;52:239) over r repetitions x k folds,
                    variance inflated by (1/J + n_test/n_train); an approximation that does not remove CV dependence exactly.
"""
import itertools
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def _bc(ok_ref, ok_cand, cluster):
    cl = np.unique(cluster)
    b = np.array([np.sum(ok_ref[cluster == c] & ~ok_cand[cluster == c]) for c in cl])
    c_ = np.array([np.sum(~ok_ref[cluster == c] & ok_cand[cluster == c]) for c in cl])
    return b, c_


def clustered_mcnemar(ok_ref, ok_cand, cluster):
    b, c = _bc(ok_ref, ok_cand, cluster)
    B, C = b.sum(), c.sum(); den = np.sum((b - c) ** 2)
    x2 = (B - C) ** 2 / den if den > 0 else 0.0
    return dict(ref_only=int(B), cand_only=int(C), fruits_discordant=int(np.sum((b + c) > 0)), chi2=float(x2), p=float(stats.chi2.sf(x2, 1)) if den > 0 else 1.0)


def fruit_signflip(ok_ref, ok_cand, cluster, n_perm=200000, seed=42):
    b, c = _bc(ok_ref, ok_cand, cluster); d = (c - b).astype(float); d = d[d != 0]
    T = abs(d.sum())
    if len(d) == 0: return dict(p=1.0, n_informative=0)
    if len(d) <= 20:
        tot = [abs(np.dot(s, d)) for s in itertools.product([-1, 1], repeat=len(d))]
        p = np.mean(np.array(tot) >= T - 1e-12)
    else:
        rng = np.random.default_rng(seed); S = rng.choice([-1, 1], size=(n_perm, len(d)))
        p = (np.sum(np.abs(S @ d) >= T - 1e-12) + 1) / (n_perm + 1)
    return dict(p=float(p), n_informative=int(len(d)))


def fruit_bootstrap(y, pred_ref, pred_cand, cluster, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed); cl = np.unique(cluster); idx = {c: np.flatnonzero(cluster == c) for c in cl}
    da, df = [], []
    for _ in range(n_boot):
        s = np.concatenate([idx[c] for c in rng.choice(cl, len(cl), replace=True)])
        da.append(np.mean(pred_cand[s] == y[s]) - np.mean(pred_ref[s] == y[s]))
        df.append(f1_score(y[s], pred_cand[s], average="macro") - f1_score(y[s], pred_ref[s], average="macro"))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(acc_ci=q(da), f1_ci=q(df))


def nb_corrected_t(diffs, test_frac):
    """diffs: per-fold differences over r repetitions x k folds (length J); test_frac = n_test / n_train."""
    d = np.asarray(diffs, float); J = len(d); m = d.mean(); s2 = d.var(ddof=1)
    se = np.sqrt((1.0 / J + test_frac) * s2); t = m / se if se > 0 else 0.0
    tq = stats.t.ppf(0.975, J - 1)
    return dict(mean=float(m), se_corrected=float(se), t=float(t), p=float(2 * stats.t.sf(abs(t), J - 1)), ci=(float(m - tq * se), float(m + tq * se)), J=J)


def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj
