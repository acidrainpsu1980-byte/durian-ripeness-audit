"""Fruit-level power illustration for a confirmatory harvest (manuscript Section 4.3, Figure 6; docs/protocol/).
Resamples fruits of the round-6 population (paired macro-F1 of the foundation fusion vs CDR-RO), fits the CI half-width scaling
(97.6/sqrt(fruits)) and simulates power of the decision rule. Between-harvest variation is NOT modelled (see harvest_scenarios.py).
usage (repo root): python src/protocol/power_analysis.py   -> docs/protocol/power_analysis.csv, power_halfwidth_scaling.csv, power_curve.{png,pdf}
(recovered from the analysis cell that produced the v2.0 protocol files; inputs results/round5/probe_sessions.parquet,
results/session_table.csv, results/v21/lobo_cnn_sessions.parquet)
"""
import os
OUT = os.environ.get("PROTOCOL_OUT", "docs/protocol"); R5 = os.environ.get("ROUND5_DIR", "results/round5"); V21 = os.environ.get("ROUND_V21", "results/v21")
os.makedirs(OUT, exist_ok=True)
import os, sys, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import f1_score

pr = pd.read_parquet(f'{R5}/probe_sessions.parquet')
st = pd.read_csv('results/session_table.csv')
cnn = pd.read_parquet(f'{V21}/lobo_cnn_sessions.parquet')

lobo = pr[pr.split=='lobo']
sid = sorted(lobo[lobo.model=='F3'].session_id)
S = st.set_index('session_id').loc[sid]
y = S.y.values.astype(int); fruit = S.fruit_id.values.astype(int); batch = S.Maturity_Stage.values

def probs(model):
    g = lobo[lobo.model==model].set_index('session_id').loc[sid][['p_Ripe','p_Overripe']].values
    return g

F1p = np.mean([probs(m) for m in ['A1','V-thermal-DINO','V-ms-DINO','V-rgb-DINO']], axis=0)
m0 = cnn[cnn.variant=='M0']
m0 = m0.set_index('session_id').loc[sid][['p_Ripe','p_Overripe']].values
pred = lambda P: 1 + P.argmax(1)
p_cdr = pred(m0); p_f1 = pred(F1p); p_f3 = pred(probs('F3'))
f = lambda p: f1_score(y, p, labels=[1,2], average='macro')

def ro_counts(y, p, fruit, fr_index):
    C = np.zeros((len(fr_index), 8)); fi = np.searchsorted(fr_index, fruit)
    for j, c in enumerate([1, 2]):
        np.add.at(C[:, 3*j], fi, ((p == c) & (y == c)).astype(float))
        np.add.at(C[:, 3*j+1], fi, (p == c).astype(float)); np.add.at(C[:, 3*j+2], fi, (y == c).astype(float))
    np.add.at(C[:, 6], fi, (p == y).astype(float)); np.add.at(C[:, 7], fi, (p == 0).astype(float))
    return C

def ro_f1(S):
    f = []
    for j in range(2):
        den = S[..., 3*j+1] + S[..., 3*j+2]; f.append(np.where(den > 0, 2*S[..., 3*j]/np.maximum(den, 1e-12), 0.0))
    return (f[0]+f[1])/2

def boot_diff(Cr, Cc, n_boot=2000, seed=42, strata=None):
    F = len(Cr); rng = np.random.default_rng(seed)
    if strata is None:
        W = np.stack([np.bincount(rng.integers(0, F, F), minlength=F) for _ in range(n_boot)]).astype(float)
    else:
        W = np.zeros((n_boot, F))
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]
            for b in range(n_boot):
                W[b, idx] += np.bincount(rng.choice(idx, len(idx)), minlength=F)[idx]
    d = ro_f1(W @ Cc) - ro_f1(W @ Cr)
    return d

fr = np.unique(fruit)
Cr = ro_counts(y, p_cdr, fruit, fr); Cc = ro_counts(y, p_f1, fruit, fr)

fr_batch = pd.Series(batch, index=fruit).groupby(level=0).first().loc[fr].values
Cr_f, Cc_f = Cr, Cc
n_sess_per_fruit = np.bincount(np.searchsorted(fr, fruit), minlength=len(fr))
F = len(fr)

rng0 = np.random.default_rng(2026)

def resample_fruits(frac, rng):
    idx = []
    for b in np.unique(fr_batch):
        ib = np.where(fr_batch == b)[0]; k = int(round(frac * len(ib)))
        idx.append(rng.choice(ib, k, replace=frac > 1))
    return np.concatenate(idx)

def halfwidth(Cr_, Cc_, n_boot=2000, seed=0):
    d = 100*boot_diff(Cr_, Cc_, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(d, [2.5, 97.5]); return (hi-lo)/2, d.std(), lo, hi

fracs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0]
R = 60; rows = []
for frac in fracs:
    for r in range(R):
        idx = resample_fruits(frac, rng0)
        hw, sd, lo, hi = halfwidth(Cr_f[idx], Cc_f[idx], n_boot=1000, seed=r)
        rows.append(dict(frac=frac, rep=r, n_fruits=len(idx), n_sessions=int(n_sess_per_fruit[idx].sum()), halfwidth_pp=hw, boot_sd_pp=sd,
                         d_pp=100*(ro_f1(Cc_f[idx].sum(0))-ro_f1(Cr_f[idx].sum(0)))))
scal = pd.DataFrame(rows)
X = np.log(scal.n_fruits.values); Y = np.log(scal.halfwidth_pp.values)
a, b = np.polyfit(X, Y, 1)
c_sqrt = np.exp(np.mean(Y + 0.5*X))
c_sess = np.exp(np.mean(np.log(scal.halfwidth_pp.values) + 0.5*np.log(scal.n_sessions.values)))

c_fruit = c_sqrt; sess_per_fruit = 104/103

def hw_of_fruits(nf): return c_fruit/np.sqrt(nf)

def power_analytic(delta, nf, z_crit=1.96):
    sigma = hw_of_fruits(nf)/1.96; return norm.cdf(delta/sigma - z_crit)

def n_fruits_needed(delta, power=0.8, z_crit=1.96):
    z = z_crit + norm.ppf(power); return (c_fruit*z/(1.96*delta))**2

deltas = [5, 7, 10]
need = {d: n_fruits_needed(d) for d in deltas}
need_holm = {d: n_fruits_needed(d, z_crit=norm.ppf(1-0.025/2)) for d in deltas}

def make_candidate(target, rng):
    p = p_f1.copy(); d = 100*(f(p)-f(p_cdr)); other = np.where(y==1, 2, 1)
    pool = list(rng.permutation(np.where((p != y) if target > d else (p == y))[0]))
    while pool:
        i = pool.pop(); q = p.copy(); q[i] = y[i] if target > d else other[i]; dq = 100*(f(q)-f(p_cdr))
        if abs(dq-target) < abs(d-target): p, d = q, dq
        else: break
    return p, d

rngc = np.random.default_rng(7); cands = {dl: [make_candidate(dl, rngc) for _ in range(40)] for dl in deltas}

def resample_fruits_wr(frac, rng):
    idx = []
    for b in np.unique(fr_batch):
        ib = np.where(fr_batch == b)[0]; idx.append(rng.choice(ib, int(round(frac*len(ib))), replace=True))
    return np.concatenate(idx)

def boot_W(F, n_boot, rng):
    idx = rng.integers(0, F, (n_boot, F)); W = np.zeros((n_boot, F)); np.add.at(W, (np.repeat(np.arange(n_boot), F), idx.ravel()), 1.0); return W

def sim_power(delta, frac, R=200, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed); hits = 0; nf = []
    for r in range(R):
        pc, _ = cands[delta][r % len(cands[delta])]; Cc_ = ro_counts(y, pc, fruit, fr)
        idx = resample_fruits_wr(frac, rng); W = boot_W(len(idx), n_boot, rng)
        d = ro_f1(W @ Cc_[idx]) - ro_f1(W @ Cr_f[idx]); lo = np.percentile(d, 2.5)
        d_point = ro_f1(Cc_[idx].sum(0)) - ro_f1(Cr_f[idx].sum(0))
        hits += (lo > 0) and (d_point >= 0.03); nf.append(len(idx))
    return hits/R, np.mean(nf)

sim_rows = []
for frac in [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
    for dl in deltas:
        pw, nf = sim_power(dl, frac, seed=int(frac*10)+dl)
        sim_rows.append(dict(frac=frac, n_fruits=nf, n_sessions=nf*sess_per_fruit, delta_pp=dl, power_sim=pw, power_analytic=power_analytic(dl, nf)))
sim = pd.DataFrame(sim_rows)

z_holm = norm.ppf(1-0.025/2)
grid = sorted(set(list(range(50, 1001, 25)) + [104, 156, 208, 312, 416, 624, 832]))
rows = []
for ns in grid:
    nf = ns/sess_per_fruit; r = dict(n_sessions=ns, n_fruits=round(nf,1), halfwidth_pp=round(hw_of_fruits(nf),2), boot_se_pp=round(hw_of_fruits(nf)/1.96,2))
    for dl in deltas:
        r[f'power_{dl}pp'] = round(power_analytic(dl, nf),3); r[f'power_{dl}pp_holm2'] = round(power_analytic(dl, nf, z_holm),3)
        s = sim[(sim.delta_pp==dl) & (np.abs(sim.n_sessions-ns) < 2)]
        r[f'power_{dl}pp_sim'] = round(float(s.power_sim.iloc[0]),3) if len(s) else np.nan
    rows.append(r)
pa = pd.DataFrame(rows)


pa.to_csv(f'{OUT}/power_analysis.csv', index=False)
scal.to_csv(f'{OUT}/power_halfwidth_scaling.csv', index=False)

import matplotlib as mpl
import matplotlib.pyplot as plt
META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


# Load data
pr = pd.read_parquet('/Users/sakonchankhachon/.claude-science/orgs/f0d94ed0-c874-4eec-a1f3-96afbe8be396/artifacts/proj_9e1556dba6c0/0317533f-7ccb-4c33-8632-c1aae63b1bb0/vfff66702_probe_sessions.parquet')
st = pd.read_csv('/Users/sakonchankhachon/.claude-science/orgs/f0d94ed0-c874-4eec-a1f3-96afbe8be396/artifacts/proj_9e1556dba6c0/4c8ba1c8-7b7d-44dc-b9a8-8f25517ac392/vb7aed25d_session_table.csv')
cnn = pd.read_parquet('/Users/sakonchankhachon/.claude-science/orgs/f0d94ed0-c874-4eec-a1f3-96afbe8be396/artifacts/proj_9e1556dba6c0/cfe9cd99-46d6-4232-a9dc-0f73f5d80140/v1084c10e_lobo_cnn_sessions.parquet')

# Figure
apply_figure_style()


fig, ax = plt.subplots(figsize=(3.5, 2.9), dpi=300)
ns = np.linspace(50, 1000, 400)
pal = {5: '#9ecae1', 7: '#3182bd', 10: '#08306b'}
for dl in deltas:
    ax.plot(ns, power_analytic(dl, ns/sess_per_fruit), color=pal[dl], lw=1.6 if dl==7 else 1.1, label=f'+{dl} pp')
    s = sim[sim.delta_pp==dl]
    ax.scatter(s.n_sessions, s.power_sim, s=12, color=pal[dl], edgecolor='white', lw=0.4, zorder=3)
ax.axhline(0.8, color=META_GREY, lw=0.7, ls=':')
ax.axvline(104, color=META_GREY, lw=0.7, ls='--')
ax.text(104+12, 0.03, 'current study\n(104 sessions)', fontsize=6, color=META_GREY, va='bottom')
ax.text(990, 0.815, '80% power', fontsize=6, color=META_GREY, ha='right', va='bottom')
for dl in deltas:
    nn = need[dl]*sess_per_fruit
    ax.plot([nn, nn], [0, 0.8], color=pal[dl], lw=0.6, ls='--')
    ax.text(nn, 0.84, f'{nn:.0f}', fontsize=6, color=pal[dl], ha='center')
ax.set_xlabel('Endpoint sessions in the new harvest')
ax.set_ylabel('Power (95% CI excludes 0)')
ax.set_title('Fruit-level illustration; between-harvest variation not modelled', loc='left')
ax.set_xlim(40, 1010)
ax.set_ylim(0, 1.0)
ax.legend(title='true gain, macro-F1', frameon=False, loc='center right', fontsize=6, title_fontsize=6)
fig.tight_layout()
fig.savefig(f'{OUT}/power_curve.png')

for t in ax.texts:
    if t.get_text()=='80% power':
        t.set_position((600, 0.815))
        t.set_ha('center')
fig.savefig(f'{OUT}/power_curve.png')
fig.savefig(f'{OUT}/power_curve.pdf')
print('need', {k: round(v*sess_per_fruit) for k, v in need.items()})
