"""Conditional planning scenarios with between-harvest variation (main-text Section 4.3; Supplementary Table D5).
Analytic normal approximation of the decision rule (paired macro-F1 gain, two-sided 95% CI excluding zero, 80% power):
SE^2 = tau^2 / H + s_f^2 / N, with H independent harvests, N endpoint sessions in total (one per fruit), s_f = 97.6 / 1.96 pp the
per-fruit SD implied by the observed CI half-width scaling (docs/protocol/power_halfwidth_scaling.csv), and tau the SD of the true
gain between harvests. tau = 0 reproduces the fruit-resampling estimates of docs/protocol/power_analysis.csv (197 and 401 sessions for 10 and 7 pp).
The minimum number of harvests is the smallest H for which 80% power is attainable with a finite number of fruits. The three observed batches are reported only as an illustration (results/round6/per_batch_ci.csv);
three batches cannot estimate tau. A t-based or hierarchical analysis with few harvests would require more.
usage (repo root): python src/protocol/harvest_scenarios.py -> docs/protocol/harvest_variance_scenarios.csv
"""
import os
import numpy as np, pandas as pd
from scipy.stats import norm
# One convention with Figure 6 / power_analysis.py: two-sided 95% CI excluding zero with 80% power, Z = z_0.975 + z_0.80 = 2.8016;
# fruits converted to endpoint sessions with the observed 104 sessions / 103 fruits, so tau = 0 reproduces 401 and 197 sessions.
S_F = 97.6 / 1.96; Z = norm.ppf(0.975) + norm.ppf(0.80); SESS_PER_FRUIT = 104 / 103
def n_needed(delta, tau, H):
    lim = (delta / Z) ** 2 - tau ** 2 / H                     # sampling variance left for fruits after the between-harvest term
    return np.inf if lim <= 0 else int(np.round(S_F ** 2 / lim * SESS_PER_FRUIT))   # nearest session, as in Figure 6
def min_harvests(delta, tau):
    """Smallest H for which 80% power is attainable with a FINITE number of fruits (tau^2/H strictly below (delta/Z)^2)."""
    return 1 if tau == 0 else int(np.floor(tau ** 2 / (delta / Z) ** 2)) + 1
rows = []
for delta in [7, 10]:
    for tau in [0, 5, 10]:
        for H in [3, 4, 5, 6, 8, 12, 17]:
            rows.append(dict(true_gain_pp=delta, tau_pp=tau, harvests=H, endpoint_sessions_needed=n_needed(delta, tau, H),
                             min_harvests_finite_fruits=min_harvests(delta, tau)))
for d_ in [7, 10]:
    for t_ in [5, 10]:
        h = min_harvests(d_, t_); assert np.isfinite(n_needed(d_, t_, h)) and (h == 1 or not np.isfinite(n_needed(d_, t_, h - 1)))
OUT = os.environ.get("PROTOCOL_OUT", "docs/protocol"); os.makedirs(OUT, exist_ok=True)
pd.DataFrame(rows).to_csv(f"{OUT}/harvest_variance_scenarios.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
