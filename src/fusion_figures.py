"""Figures for the fusion / leakage / LOBO analysis.  Reads results/*.csv written by fusion_pipeline.py."""
import json, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

import os
RES = os.environ.get("DURIAN_RESULTS", "results")
FIG = os.environ.get("DURIAN_FIGURES", "figures/diagnostic"); os.makedirs(FIG, exist_ok=True)
RULE_LABEL = {"single": "single modality", "mean": "late: mean of probabilities", "product": "late: product rule",
              "stacking": "late: stacked logistic regression", "early_lgbm": "early: LightGBM on features",
              "early_logreg": "early: logistic regression on features"}
RULE_COL = {"single": "#4d4d4d", "mean": "#1b7837", "product": "#5aae61", "stacking": "#a6dba0",
            "early_lgbm": "#762a83", "early_logreg": "#c2a5cf"}
MOD_SHORT = {"thermal": "T", "multispectral": "M", "sound": "S", "rgb": "R"}
MOD_COL = {"thermal": "#d95f02", "multispectral": "#7570b3", "sound": "#1b9e77", "rgb": "#e7298a"}
MOD_ORDER = ["thermal", "multispectral", "sound", "rgb"]


def subset_order(subsets):
    """Group by number of modalities (1..k), then by modality order."""
    key = lambda s: (len(s.split("+")), [MOD_ORDER.index(m) for m in s.split("+")])
    return sorted(set(subsets), key=key)


def subset_label(s):
    parts = s.split("+")
    if len(parts) == 1:
        return {"rgb": "RGB"}.get(parts[0], parts[0].capitalize())
    return " + ".join(MOD_SHORT[p] for p in parts)


def fig_ablation(out=os.path.join(FIG, "fig_fusion_ablation.png")):
    fr = pd.read_csv(f"{RES}/fusion_results.csv")
    sm = fr[(fr.record == "summary") & (fr.task == "3class")].copy()
    order = subset_order(sm.subset)
    rules = ["single", "mean", "product", "stacking", "early_lgbm", "early_logreg", "early_lgbm_embpca", "early_logreg_embpca"]
    rl = dict(RULE_LABEL, early_lgbm_embpca="early: LightGBM (RGB = PCA-50 of CNN embedding)", early_logreg_embpca="early: logistic regression (RGB = PCA-50 of CNN embedding)")
    rc = dict(RULE_COL, early_lgbm_embpca="#e7298a", early_logreg_embpca="#f1b6da")
    nmod = [len(s.split("+")) for s in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.42 * len(order) + 1.6), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, met, xlabel in zip(axes, ["macro_f1", "accuracy"], ["Session macro-F1 (5-fold mean, 95% CI)", "Session accuracy (5-fold mean, 95% CI)"]):
        for i, sub in enumerate(order):
            d = sm[sm.subset == sub]
            rs = [r for r in rules if r in set(d.rule)]
            offs = np.linspace(-0.33, 0.33, len(rs)) if len(rs) > 1 else [0]
            for o, r in zip(offs, rs):
                row = d[d.rule == r].iloc[0]
                ax.errorbar(row[f"{met}_mean"], i + o, xerr=[[row[f"{met}_mean"] - row[f"{met}_ci_lo"]], [row[f"{met}_ci_hi"] - row[f"{met}_mean"]]],
                            fmt="o", ms=3.5, color=rc[r], ecolor=rc[r], elinewidth=0.9, capsize=0, label=rl[r])
        ax.set_yticks(range(len(order))); ax.set_yticklabels([subset_label(s) for s in order])
        ax.invert_yaxis(); ax.set_xlabel(xlabel); ax.grid(axis="x", lw=0.4, alpha=0.5)
        for i in range(1, len(order)):
            if nmod[i] != nmod[i - 1]: ax.axhline(i - 0.5, color="0.75", lw=0.6)
        for s_ in ["top", "right"]: ax.spines[s_].set_visible(False)
    # group headers in the left margin
    for k in sorted(set(nmod)):
        rows = [i for i, n in enumerate(nmod) if n == k]
        axes[0].text(-0.36, np.mean(rows), f"{k} modalit{'y' if k == 1 else 'ies'}", transform=axes[0].get_yaxis_transform(), rotation=90, va="center", ha="center", fontsize=6.5, color="0.35")
    maj = fr[(fr.record == "summary") & (fr.task == "3class") & (fr.rule == "single")].iloc[0]
    _st = pd.read_csv(os.path.join(RES, "session_table.csv")); _st = _st[_st.in_main_analysis.astype(str) == "True"]
    _maj = _st.label_capture.value_counts(normalize=True).max()
    axes[1].axvline(_maj, color="0.5", lw=0.8, ls="--"); axes[1].text(_maj + 0.006, len(order) - 0.6, "majority\nclass", ha="left", va="bottom", fontsize=6, color="0.4")
    axes[0].axvline(1 / 3, color="0.5", lw=0.8, ls="--"); axes[0].text(0.345, len(order) - 0.6, "chance", ha="left", va="bottom", fontsize=6, color="0.4")
    axes[0].set_xlim(0.25, 0.97); axes[1].set_xlim(0.55, 0.98)
    h, l = axes[0].get_legend_handles_labels(); uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    axes[0].text(-0.02, 1.01, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=10, ha="right"); axes[1].text(-0.02, 1.01, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=10, ha="right")
    fig.savefig(out, dpi=300, bbox_inches="tight"); return fig


def fig_leakage(out=os.path.join(FIG, "fig_leakage.png")):
    lk = pd.read_csv(f"{RES}/leakage_table.csv")
    cap = lk[lk.level == "capture"].copy()
    ses = lk[(lk.level == "session")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.6))
    def dumbbell(ax, d, labels, title):
        for i, (_, r) in enumerate(d.iterrows()):
            col = MOD_COL.get(r.modality, "#333333")
            ax.plot([r.fruit_pooled_f1, r.random_pooled_f1], [i, i], color=col, lw=1.2, alpha=0.7)
            ax.plot(r.fruit_pooled_f1, i, "o", color=col, ms=5, mfc="white", mew=1.4)
            ax.plot(r.random_pooled_f1, i, "o", color=col, ms=5)
            ax.text(max(r.fruit_pooled_f1, r.random_pooled_f1) + 0.015, i, f"{r.delta_f1_pp:+.1f} pp", va="center", fontsize=6, color="0.3")
        ax.set_yticks(range(len(d))); ax.set_yticklabels(labels); ax.invert_yaxis()
        ax.set_xlabel("Macro-F1 (pooled out-of-fold)"); ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="x", lw=0.4, alpha=0.5); ax.set_xlim(0.4, 1.0)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
    cap_lab = [f"{r.modality} / {r.model}" for _, r in cap.iterrows()]
    dumbbell(axes[0], cap, cap_lab, "Capture level: random split inflates\nimage models, not sound")
    ses_fus = ses[ses.model.str.contains("fusion")]
    ses_single = ses[~ses.model.str.contains("fusion") & ses.modality.isin(MOD_ORDER)]
    ses_single = ses_single[ses_single.model.isin(["svm_rbf", "logreg", "cnn", "efficientnetb0_probe"])]
    keep = {"thermal+sound", "thermal+multispectral", "thermal+multispectral+sound", "thermal+sound+rgb", "thermal+multispectral+sound+rgb"}
    ses_fus = ses_fus[ses_fus.modality.isin(keep) | ses_fus.model.str.contains("gradient")]
    d2 = pd.concat([ses_single, ses_fus])
    lab2 = [f"{r.modality} / {r.model}" if "fusion" not in r.model else f"{' + '.join(MOD_SHORT[m] for m in r.modality.split('+'))} fusion ({'GBM' if 'gradient' in r.model else 'best'} models)" for _, r in d2.iterrows()]
    dumbbell(axes[1], d2, lab2, "Session level: fusion inherits the\ninflation of its image modalities")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", color="0.3", mfc="white", mew=1.4, ls="", ms=5, label="fruit-grouped split"),
          Line2D([], [], marker="o", color="0.3", ls="", ms=5, label="random (capture-level) split")]
    hs += [Line2D([], [], color=MOD_COL[m], lw=2, label=m) for m in MOD_COL]
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.05))
    axes[0].text(-0.55, 1.08, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=10); axes[1].text(-0.55, 1.08, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=10)
    fig.savefig(out, dpi=300, bbox_inches="tight"); return fig


def fig_lobo(out=os.path.join(FIG, "fig_lobo.png")):
    lb = pd.read_csv(f"{RES}/lobo_table.csv")
    lb = lb[lb.source.str.contains("refit|json|mean of modality")]
    series = ["thermal", "multispectral", "sound", "rgb", "thermal+sound", "thermal+multispectral+sound", "thermal+multispectral+sound+rgb"]
    labels = {"thermal": "Thermal (SVM-RBF)", "multispectral": "Multispectral (LR)", "sound": "Knock sound (CNN)", "rgb": "RGB (EfficientNet-B0 probe)",
              "thermal+sound": "T + S fusion (mean)", "thermal+multispectral+sound": "T + M + S fusion (mean)", "thermal+multispectral+sound+rgb": "T + M + S + R fusion (mean)"}
    cols = {m: MOD_COL[m] for m in MOD_ORDER}; cols.update({"thermal+sound": "#5aae61", "thermal+multispectral+sound": "#1b7837", "thermal+multispectral+sound+rgb": "#00441b"})
    markers = {"thermal": "o", "multispectral": "o", "sound": "o", "rgb": "o", "thermal+sound": "D", "thermal+multispectral+sound": "s", "thermal+multispectral+sound+rgb": "^"}
    series = [s_ for s_ in series if s_ in set(lb.modality)]
    batches = ["Immature", "Mature", "Overmature"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(wspace=0.3))
    for ax, met, yl in zip(axes, ["session_acc", "session_macro_f1"], ["Session accuracy", "Session macro-F1"]):
        for bi, b in enumerate(batches):
            d = lb[lb.held_out_batch == b]
            maj = d.majority_rate.iloc[0]
            if met == "session_acc":
                ax.hlines(maj, bi - 0.38, bi + 0.38, color="0.35", lw=1.2, ls="--")
            else:
                ax.hlines(1 / 3, bi - 0.38, bi + 0.38, color="0.35", lw=1.2, ls="--")
            offs = np.linspace(-0.32, 0.32, len(series))
            for o, s in zip(offs, series):
                r = d[d.modality == s].iloc[0]
                ax.plot(bi + o, r[met], marker=markers[s], color=cols[s], ms=6, ls="", label=labels[s], mec="white", mew=0.5)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{b}\n(n = {int(lb[lb.held_out_batch == b].n_sessions.iloc[0])} sessions)" for b in batches]); ax.set_xlabel("Held-out harvest batch (trained on the other two)")
        ax.set_ylabel(yl); ax.set_ylim(0.15, 0.85); ax.grid(axis="y", lw=0.4, alpha=0.5)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
    axes[0].text(2.42, lb[lb.held_out_batch == "Overmature"].majority_rate.iloc[0], "majority\nclass", fontsize=6, va="center", color="0.35")
    axes[1].text(2.42, 1 / 3, "chance", fontsize=6, va="center", color="0.35")
    h, l = axes[0].get_legend_handles_labels(); uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Diagnostic: leave-one-harvest-batch-out session accuracy and macro-F1 versus the held-out batch's majority-class rate (paper: Figure 5)", x=0.02, ha="left", fontsize=8)
    axes[0].text(-0.18, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=10); axes[1].text(-0.18, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=10)
    fig.savefig(out, dpi=300, bbox_inches="tight"); return fig


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6})
    fig_ablation(); fig_leakage(); fig_lobo()
