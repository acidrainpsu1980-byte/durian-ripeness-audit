"""Nested model-family selection, nested configuration selection and nested stacking on one fruit partition.

For outer test fold k, every choice is made from inner out-of-fold predictions of the four training folds only:
the inner prediction of training fold j comes from the fit that excluded folds {k, j} (fold-pair refits, run_fits.py /
cnn_common.py / cnn_part.py). The outer-test prediction of fold k comes from the fit on the four folds other than k.
Selection criterion everywhere: pooled session macro-F1 on the inner predictions (ties: higher accuracy, then the
earlier candidate in the pre-specified list).
"""
import os, itertools
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
import cands

CL = ["p_Unripe", "p_Ripe", "p_Overripe"]
MODS = ["thermal", "multispectral", "sound", "rgb"]
HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = pd.read_csv(f"{HERE}/partitions.csv").set_index("fruit_id")
PAIRS = list(itertools.combinations(range(5), 2))


def _pk(a, b):
    return f"pair_{min(a, b)}{max(a, b)}"


class Preds:
    """Capture-level outer and pair predictions for one partition, aggregated to sessions on demand."""
    def __init__(self, P, fits_dir, cnn_dir, cnn_fmt, knock_parquet, rec_files):
        self.P = P; self.fits_dir = fits_dir; self.cnn_dir = cnn_dir; self.cnn_fmt = cnn_fmt
        self.knock = pd.read_parquet(knock_parquet)[["file"]]
        self.rec_files = rec_files; self.meta = {}; self._cache = {}

    def capture_meta(self, mod):
        if mod not in self.meta:
            m, _ = cands.load(mod); self.meta[mod] = m
        return self.meta[mod]

    def _arrays(self, mod, cand):
        key = (mod, cand)
        if key in self._cache: return self._cache[key]
        if cand == "cnn":
            out = {}
            for k in range(5): out["outer"] = out.get("outer")
            ok = np.full((len(self.knock), 3), np.nan)
            for k in range(5):
                a = np.load(f"{self.cnn_dir}/{self.cnn_fmt.format(P=self.P, kind='outer', arg=k)}"); ok[~np.isnan(a[:, 0])] = a[~np.isnan(a[:, 0])]
            out = {"outer": ok}
            for a, b in PAIRS:
                out[f"pair_{a}{b}"] = np.load(f"{self.cnn_dir}/{self.cnn_fmt.format(P=self.P, kind='pair', arg=f'{a}{b}')}")
            # knock -> recording mean, restricted to the analysed recordings
            res = {}
            for k_, arr in out.items():
                t = pd.DataFrame(arr, columns=CL); t["file"] = self.knock.file.values
                res[k_] = t.groupby("file")[CL].mean().reindex(self.rec_files).values
            files = np.asarray(self.rec_files)
        else:
            z = np.load(f"{self.fits_dir}/{self.P}/{mod}__{cand}.npz", allow_pickle=True)
            res = {k_: z[k_] for k_ in z.files if k_ != "files"}; files = z["files"]
        m = self.capture_meta(mod)
        if mod == "sound" and cand == "cnn":
            assert set(files) == set(m.file.values)
            idx = pd.Series(range(len(files)), index=files).reindex(m.file.values).values
            res = {k_: v[idx] for k_, v in res.items()}
        else:
            assert (np.asarray(files) == m.file.values.astype(str)).all(), (mod, cand)
        self._cache[key] = res
        return res

    def session(self, mod, cand, which):
        """which = 'outer' or (a, b) pair -> DataFrame session_id x CL (only sessions with predictions)."""
        arr = self._arrays(mod, cand)["outer" if which == "outer" else _pk(*which)]
        m = self.capture_meta(mod)
        t = pd.DataFrame(arr, columns=CL); t["session_id"] = m.session_id.astype(str).values
        return t.dropna().groupby("session_id")[CL].mean()

    def inner(self, mod, cand, k):
        """Session probabilities for every session outside fold k, from fits excluding {k, j}."""
        m = self.capture_meta(mod); fold = m.fruit_id.map(PARTS[self.P]).values
        parts = []
        for j in range(5):
            if j == k: continue
            s = self.session(mod, cand, (k, j))
            sj = pd.Index(m.session_id.astype(str).values[fold == j]).unique()
            parts.append(s.reindex(sj))
        return pd.concat(parts)


def mf1(y, P):
    return f1_score(y, P.argmax(1), average="macro")


def score(y, P):
    p = P.argmax(1); return (f1_score(y, p, average="macro"), accuracy_score(y, p))


def combine(Ps, rule):
    if rule == "mean": return np.mean(Ps, 0)
    if rule == "product":
        L = np.sum([np.log(np.clip(p, 1e-6, 1)) for p in Ps], 0); E = np.exp(L - L.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)
    raise ValueError(rule)


def fit_stacker(X, y, groups):
    gs = GridSearchCV(LogisticRegression(max_iter=5000, class_weight="balanced"), {"C": [0.03, 0.1, 0.3, 1, 3, 10]},
                      cv=GroupKFold(3), scoring="f1_macro", n_jobs=1)
    gs.fit(X, y, groups=groups); return gs


def evaluate(pr, ST, families=None, stack_subsets=None):
    """ST: 268 common sessions (session_id, fruit_id, y). Returns dict of session-level OOF probability arrays
    (aligned with ST) for every configuration, plus a table of per-fold choices."""
    P = pr.P; sid = ST.session_id.values; y = ST.y.values.astype(int)
    fold = ST.fruit_id.map(PARTS[P]).values.astype(int)
    families = families or cands.CANDIDATES
    out, choices = {}, []
    # ---- per-modality outer and inner session probs for every candidate
    O, I, LAB = {}, {}, {}
    for mod in MODS:
        m = pr.capture_meta(mod)
        LAB[mod] = m.groupby(m.session_id.astype(str)).y.first()
        for c in families[mod]:
            O[(mod, c)] = pr.session(mod, c, "outer")
            for k in range(5): I[(mod, c, k)] = pr.inner(mod, c, k)
    # ---- fixed families (v1.3 selection) on the 268 sessions
    for mod in MODS:
        out[f"{mod}|fixed"] = O[(mod, cands.SELECTED_V13[mod])].reindex(sid).values
    # ---- nested family selection per modality (criterion on the modality's own training sessions)
    chosen = {}
    for mod in MODS:
        Pm = np.zeros((len(sid), 3))
        for k in range(5):
            best = None
            for c in families[mod]:
                s = I[(mod, c, k)].dropna(); f1, acc = score(LAB[mod].reindex(s.index).values.astype(int), s.values)
                choices.append(dict(partition=P, level="family", modality=mod, fold=k, candidate=c, inner_f1=f1, inner_acc=acc))
                if best is None or (f1, acc) > best[0]: best = ((f1, acc), c)
            chosen[(mod, k)] = best[1]
            Pm[fold == k] = O[(mod, best[1])].reindex(sid[fold == k]).values
        out[f"{mod}|nested_family"] = Pm
    # ---- nested configuration selection over singles + mean/product fusions of the nested-chosen families
    configs = [((mod,), "single") for mod in MODS]
    for r in range(2, 5):
        for sub in itertools.combinations(MODS, r):
            configs += [(sub, "mean"), (sub, "product")]
    Psel = np.zeros((len(sid), 3)); Pss = np.zeros((len(sid), 3))
    for k in range(5):
        tr_sid = sid[fold != k]; ytr = y[fold != k]
        inner_k = {mod: I[(mod, chosen[(mod, k)], k)].reindex(tr_sid).values for mod in MODS}
        outer_k = {mod: O[(mod, chosen[(mod, k)])].reindex(sid[fold == k]).values for mod in MODS}
        best, best1 = None, None
        for sub, rule in configs:
            Pi = inner_k[sub[0]] if rule == "single" else combine([inner_k[m] for m in sub], rule)
            f1, acc = score(ytr, Pi); name = "+".join(sub) + "|" + rule
            choices.append(dict(partition=P, level="configuration", modality=name, fold=k, candidate=name, inner_f1=f1, inner_acc=acc))
            if best is None or (f1, acc) > best[0]: best = ((f1, acc), sub, rule)
            if rule == "single" and (best1 is None or (f1, acc) > best1[0]): best1 = ((f1, acc), sub, rule)
        for tgt, (_, sub, rule) in [(Psel, best), (Pss, best1)]:
            tgt[fold == k] = outer_k[sub[0]] if rule == "single" else combine([outer_k[m] for m in sub], rule)
        choices.append(dict(partition=P, level="selected_configuration", fold=k, candidate="+".join(best[1]) + "|" + best[2],
                            modality="families: " + ", ".join(f"{m}={chosen[(m, k)]}" for m in MODS)))
        choices.append(dict(partition=P, level="selected_single", fold=k, candidate=best1[1][0]))
    out["nested_selected_configuration"] = Psel; out["nested_selected_single"] = Pss
    # ---- fixed-family fusions: mean / product rules and nested stacking
    F = {mod: out[f"{mod}|fixed"] for mod in MODS}
    for r in range(2, 5):
        for sub in itertools.combinations(MODS, r):
            nm = "+".join(sub)
            out[f"{nm}|mean"] = combine([F[m] for m in sub], "mean"); out[f"{nm}|product"] = combine([F[m] for m in sub], "product")
    for sub in (stack_subsets or [s for r in range(2, 5) for s in itertools.combinations(MODS, r)]):
        nm = "+".join(sub); Pn = np.zeros((len(sid), 3))
        for k in range(5):
            tr = fold != k; te = fold == k
            Xtr = np.hstack([I[(m, cands.SELECTED_V13[m], k)].reindex(sid[tr]).values for m in sub])
            assert not np.isnan(Xtr).any(), (nm, k)
            gs = fit_stacker(Xtr, y[tr], ST.fruit_id.values[tr])
            Pn[te] = gs.predict_proba(np.hstack([F[m][te] for m in sub]))
        out[f"{nm}|stacking"] = Pn
    return out, pd.DataFrame(choices), fold
