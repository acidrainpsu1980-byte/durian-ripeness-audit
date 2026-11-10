"""Figure 1c and its statistics: pulp dry matter, total soluble solids and firmness by actual endpoint class.
Population as in the paper: every fruit of the depositors' log with a ripeness endpoint (excluded == False and actual in
Unripe/Ripe/Overripe; 185 fruits, including 21 outside the 164-fruit analysis set), each panel using the fruits with that measurement.
Statistics (both for the full log and for the 164 analysis fruits): n, mean, SD and median per class; Kruskal-Wallis p over the three
classes; Mann-Whitney p and AUC for Ripe versus Overripe, taking lower values as Overripe (ties count one half).
The panel is drawn as in the composed Figure 1 (points coloured by harvest batch, bar = median; jitter seed 0); the composition of
panels (a) and (b) is not regenerated.
usage (repo root): python src/audit/fig1c_reference.py [OUT_DIR]   -> OUT_DIR/fig1c_reference_stats.csv, fig1c_reference.{png,pdf}
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
OUT = sys.argv[1] if len(sys.argv) > 1 else "results/fig1c"; os.makedirs(OUT, exist_ok=True)
CLASSES = ["Unripe", "Ripe", "Overripe"]; BATCHES = ["Immature", "Mature", "Overmature"]
REFS = [("Firmness_AVE", "Firmness (dataset units)"), ("TSS_AVE", "Total soluble solids (°Brix)"), ("Dry_Matter_AVE", "Dry matter (%)")]
ft = pd.read_csv("data/durian_mm_fruit_table.csv")
fr = pd.read_csv("audit/fruits.csv"); A = set(fr.fruit_id[fr.status == "analysis_set"])
log = ft[(ft.excluded.astype(str) == "False") & ft.actual.isin(CLASSES)].copy(); assert len(log) == 185, len(log)
ana = log[log.fruit_id.isin(A)]; assert len(ana) == 164, len(ana)

rows = []
for pop, d in [("fruit log with endpoint (185)", log), ("analysis set (164)", ana)]:
    for col, _ in REFS:
        g = {c: d.loc[d.actual == c, col].dropna().values for c in CLASSES}
        r, o = g["Ripe"], g["Overripe"]
        auc = (np.sum(r[:, None] > o[None, :]) + 0.5 * np.sum(r[:, None] == o[None, :])) / (len(r) * len(o))
        row = dict(population=pop, reference=col, kruskal_p=stats.kruskal(*g.values()).pvalue,
                   ripe_vs_overripe_mannwhitney_p=stats.mannwhitneyu(r, o).pvalue, ripe_vs_overripe_auc_lower_is_overripe=auc)
        for c in CLASSES:
            row.update({f"n_{c}": len(g[c]), f"mean_{c}": g[c].mean(), f"sd_{c}": g[c].std(ddof=1), f"median_{c}": np.median(g[c])})
        rows.append(row)
pd.DataFrame(rows).to_csv(f"{OUT}/fig1c_reference_stats.csv", index=False)

import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})
PAL = {"Immature": "#A8DDB5", "Mature": "#43A2CA", "Overmature": "#0868AC"}
CM = 1 / 2.54; fig, axs = plt.subplots(1, 3, figsize=(17.5 * CM, 5.8 * CM), gridspec_kw={"wspace": 0.4})
rng = np.random.default_rng(0)
for k, (col, lab) in enumerate(REFS):
    a = axs[k]
    for ci, cls in enumerate(CLASSES):
        d = log[log.actual == cls].dropna(subset=[col])
        for b in BATCHES:
            db = d[d.Maturity_Stage == b]
            a.scatter(ci + rng.uniform(-0.22, 0.22, len(db)), db[col], s=9, color=PAL[b], alpha=0.85, linewidth=0, zorder=3)
        med = d[col].median(); a.hlines(med, ci - 0.3, ci + 0.3, color="black", lw=1.4, zorder=4)
        a.text(ci + 0.33, med, f"{med:.2g}" if col != "Dry_Matter_AVE" else f"{med:.0f}", fontsize=6, va="center", ha="left")
    a.set_xticks(range(3)); a.set_xticklabels(CLASSES); a.set_xlim(-0.5, 2.7); a.set_ylabel(lab); a.margins(y=0.08)
    a.set_xlabel("Actual ripeness at end-point")
    if k == 0:
        for b in BATCHES: a.scatter([], [], s=12, color=PAL[b], label=f"{b} batch")
        a.legend(loc="upper right", fontsize=6, handletextpad=0.2, borderaxespad=0.2)
    if k == 1: a.text(0.02, 0.98, "bar = median", transform=a.transAxes, fontsize=6, va="top", color="#555555")
fig.savefig(f"{OUT}/fig1c_reference.png"); fig.savefig(f"{OUT}/fig1c_reference.pdf")
print("written:", f"{OUT}/fig1c_reference_stats.csv, fig1c_reference.png, fig1c_reference.pdf")
