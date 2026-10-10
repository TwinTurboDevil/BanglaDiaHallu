#!/usr/bin/env python3
"""
BanglaDiaHallu - statistical analysis of the expert ratings.

Reproduces every number, table and result figure reported in the manuscript
from the physicians' scoring sheet (data/Scoring_Table_BanglaDiaHallu.xlsx).

Design of the data
------------------
40 Bangla diabetes questions x 5 model configurations x 2 independent generations
("iterations") = 400 rated responses. Medical accuracy, completeness, safety and
readability were scored for every response (0.00-1.00). Reliability was scored
once per question-configuration pair by comparing the two generations (200 scores).

Because every configuration answered the same 40 questions, the question is the
unit of analysis (a blocking factor). The two generations of a question are not
independent, so they are averaged before the paired analyses.

Analyses
--------
1. Descriptive statistics (response level).
2. Omnibus comparison of the five configurations: Friedman test on question-level
   scores, with Kendall's W as the effect size.
3. Planned comparisons of BanglaDiaRAG with each baseline: two-sided Wilcoxon
   signed-rank tests on question-level scores, Holm-adjusted across the
   5 criteria x 4 baselines = 20 tests; effect sizes are the mean paired
   difference with a 95% percentile bootstrap CI (10,000 resamples of questions)
   and the matched-pairs rank-biserial correlation.
4. Hallucination rate: share of responses with medical accuracy < 0.70
   (the rubric's "clinically compromised" band), with Wilson 95% CIs and
   question-level cluster-bootstrap CIs; paired sign-flip permutation tests on
   per-question counts (Holm-adjusted across 4 comparisons).
5. Sensitivity analyses: linear mixed model with a random intercept per question;
   one-way ANOVA + Tukey HSD treating responses as independent; alternative
   hallucination thresholds; all pairwise Wilcoxon comparisons.
6. Clinical sub-domain breakdown and run-to-run agreement.
7. Exploratory: responses containing words in scripts other than Bangla and
   Latin (read from data/model_responses.csv).

Usage (from the repository root)
--------------------------------
    python analysis/Statistical_Analysis.py \
        --xlsx data/Scoring_Table_BanglaDiaHallu.xlsx \
        --responses data/model_responses.csv --out results --figdir figures

Requirements: numpy, pandas, scipy, statsmodels, openpyxl, matplotlib.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

SEED = 20260929
N_BOOT = 10_000
N_PERM = 200_000
THRESHOLD = 0.70  # rubric boundary: 0.00-0.69 clinically compromised
DECIMALS = 6      # rounding of question-level means and paired differences (removes float noise)

# Order used everywhere (tables and figures); BanglaDiaRAG last.
CONFIGS = ["Qwen3.5-9B", "TigerLLM-9B", "Qwen3.6-35B-A3B", "Gemma-4-26B-A4B", "BanglaDiaRAG"]
BASELINES = CONFIGS[:-1]
RAG = "BanglaDiaRAG"
CONFIG_LABELS = {
    "Qwen3.5-9B": "Qwen3.5-9B",
    "TigerLLM-9B": "TigerLLM-9B",
    "Qwen3.6-35B-A3B": "Qwen3.6-35B-A3B",
    "Gemma-4-26B-A4B": "Gemma-4-26B-A4B",
    "BanglaDiaRAG": "BanglaDiaRAG",
}
# Column block (0-based) where each configuration starts in the scoring sheet.
SHEET_BLOCK_START = {"Qwen3.5-9B": 2, "TigerLLM-9B": 9, "Qwen3.6-35B-A3B": 16,
                     "Gemma-4-26B-A4B": 23, "BanglaDiaRAG": 30}
CRITERIA = ["MA", "COMP", "SAFE", "READ", "REL"]
RESPONSE_CRITERIA = ["MA", "COMP", "SAFE", "READ"]
CRITERION_LABELS = {"MA": "Medical accuracy", "COMP": "Completeness", "SAFE": "Safety",
                    "READ": "Readability", "REL": "Reliability"}
DOMAINS = [  # clinical sub-domains of the question bank (question numbers inclusive)
    ("Dietary", 1, 9),
    ("Medication and insulin", 10, 16),
    ("Lifestyle and exercise", 17, 21),
    ("Complications and safety", 22, 26),
    ("Symptoms and early detection", 27, 32),
    ("Special situations", 33, 40),
]


def domain_of(q: int) -> str:
    for name, lo, hi in DOMAINS:
        if lo <= q <= hi:
            return name
    raise ValueError(q)


# ----------------------------------------------------------------------------
# 1. Load the scoring sheet
# ----------------------------------------------------------------------------
def load_scores(xlsx: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (responses, reliability) in long format.

    Sheet layout: rows 3-82 hold Question 1..40 x Response 1..2. Each configuration
    occupies 7 columns: medical accuracy (2 merged), completeness (2 merged),
    safety, readability, reliability (reliability only on the Response-1 row).
    """
    ws = openpyxl.load_workbook(xlsx, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(min_row=3, max_row=82, values_only=True))
    resp, rel = [], []
    q = None
    for r in rows:
        if r[0] is not None:
            q = int(str(r[0]).split()[-1])
        it = int(str(r[1]).split()[-1])
        for cfg, s in SHEET_BLOCK_START.items():
            resp.append(dict(question=q, iteration=it, config=cfg,
                             MA=float(r[s]), COMP=float(r[s + 2]), SAFE=float(r[s + 4]), READ=float(r[s + 5])))
            if r[s + 6] is not None:
                rel.append(dict(question=q, config=cfg, REL=float(r[s + 6])))
    responses = pd.DataFrame(resp)
    reliability = pd.DataFrame(rel)
    assert responses.shape == (400, 7), responses.shape
    assert reliability.shape == (200, 3), reliability.shape
    assert responses.isna().sum().sum() == 0
    responses["domain"] = responses["question"].map(domain_of)
    reliability["domain"] = reliability["question"].map(domain_of)
    return responses, reliability


def question_level(responses: pd.DataFrame, reliability: pd.DataFrame) -> pd.DataFrame:
    """One row per question x configuration: mean of the two generations + reliability.

    Means are rounded to 6 decimals so that equal scores stay exactly equal
    (e.g. (0.80 + 0.90) / 2 and (0.85 + 0.85) / 2 are both 0.85); otherwise
    floating-point noise would break ties in the rank-based tests."""
    qa = responses.groupby(["question", "config"], as_index=False)[RESPONSE_CRITERIA].mean()
    qa[RESPONSE_CRITERIA] = qa[RESPONSE_CRITERIA].round(DECIMALS)
    qa = qa.merge(reliability[["question", "config", "REL"]], on=["question", "config"])
    qa["domain"] = qa["question"].map(domain_of)
    return qa


def wide(qa: pd.DataFrame, crit: str) -> pd.DataFrame:
    return qa.pivot(index="question", columns="config", values=crit)[CONFIGS]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014); zero differences dropped."""
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return float((w_pos - w_neg) / (w_pos + w_neg))


def boot_mean_ci(d: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def signflip_pvalue(d: np.ndarray, rng: np.random.Generator, n_perm: int = N_PERM) -> float:
    """Two-sided paired permutation (sign-flip) test on the mean difference.
    Exact enumeration when <= 20 non-zero differences, otherwise Monte Carlo."""
    d = d[d != 0].astype(float)
    m = d.size
    if m == 0:
        return 1.0
    obs = abs(d.mean())
    if m <= 20:
        signs = np.array(list(itertools.product([-1, 1], repeat=m)))
        perm = np.abs((signs * d).mean(axis=1))
        return float(np.mean(perm >= obs - 1e-12))
    signs = rng.choice([-1, 1], size=(n_perm, m))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= obs - 1e-12) + 1) / (n_perm + 1))


def pct1(x: float) -> str:
    """Percentage with one decimal, rounding half up (9/80 -> 11.3)."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(round(x * 100, 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------
# 2. Analyses
# ----------------------------------------------------------------------------
def descriptives(responses, reliability):
    out = []
    for c in CONFIGS:
        for crit in CRITERIA:
            s = (reliability if crit == "REL" else responses).query("config == @c")[crit]
            q1, med, q3 = np.percentile(s, [25, 50, 75])
            out.append(dict(config=c, criterion=crit, n=s.size, mean=s.mean(), sd=s.std(ddof=1),
                            median=med, q1=q1, q3=q3))
    return pd.DataFrame(out)


def friedman_tests(qa):
    out = []
    n, k = qa["question"].nunique(), len(CONFIGS)
    for crit in CRITERIA:
        w = wide(qa, crit)
        chi2, p = stats.friedmanchisquare(*[w[c] for c in CONFIGS])
        out.append(dict(criterion=crit, chi2=chi2, df=k - 1, p=p, kendalls_w=chi2 / (n * (k - 1)), n=n))
    return pd.DataFrame(out)


def pairwise_vs_rag(qa, rng):
    out = []
    for crit in CRITERIA:
        w = wide(qa, crit)
        for b in BASELINES:
            d = (w[RAG] - w[b]).round(DECIMALS).to_numpy()  # exact ties and zeros
            stat, p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
            lo, hi = boot_mean_ci(d, rng)
            out.append(dict(criterion=crit, baseline=b, mean_rag=w[RAG].mean(), mean_base=w[b].mean(),
                            diff=d.mean(), ci_lo=lo, ci_hi=hi, median_diff=np.median(d),
                            wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()),
                            W=stat, p=p, r_rb=rank_biserial(d)))
    df = pd.DataFrame(out)
    df["p_holm"] = multipletests(df["p"], method="holm")[1]
    return df


def pairwise_all(qa):
    out = []
    for crit in CRITERIA:
        w = wide(qa, crit)
        for a, b in itertools.combinations(CONFIGS, 2):
            d = (w[a] - w[b]).round(DECIMALS).to_numpy()
            stat, p = stats.wilcoxon(d, zero_method="wilcox")
            out.append(dict(criterion=crit, config_a=a, config_b=b, diff_a_minus_b=d.mean(), W=stat, p=p,
                            r_rb=rank_biserial(d)))
    df = pd.DataFrame(out)
    df["p_holm"] = multipletests(df["p"], method="holm")[1]
    return df


def hallucination(responses, rng, threshold=THRESHOLD):
    r = responses.copy()
    r["hall"] = (r["MA"] < threshold).astype(int)
    r["unsafe"] = (r["SAFE"] < threshold).astype(int)
    r["any_compromised"] = (r[RESPONSE_CRITERIA].min(axis=1) < threshold).astype(int)
    per_q = r.groupby(["question", "config"])[["hall", "unsafe"]].sum().unstack("config")
    rows = []
    for c in CONFIGS:
        s = r.query("config == @c")
        k, n = int(s["hall"].sum()), len(s)
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        counts = per_q["hall"][c].to_numpy()
        boots = rng.integers(0, counts.size, size=(N_BOOT, counts.size))
        bs = counts[boots].sum(axis=1) / n
        ku = int(s["unsafe"].sum())
        ulo, uhi = proportion_confint(ku, n, alpha=0.05, method="wilson")
        rows.append(dict(config=c, k=k, n=n, hr=k / n, wilson_lo=lo, wilson_hi=hi,
                         cluster_lo=float(np.percentile(bs, 2.5)), cluster_hi=float(np.percentile(bs, 97.5)),
                         questions_affected=int((counts > 0).sum()),
                         unsafe_k=ku, unsafe_rate=ku / n, unsafe_lo=ulo, unsafe_hi=uhi,
                         any_compromised_k=int(s["any_compromised"].sum())))
    hr = pd.DataFrame(rows)
    comps = []
    for b in BASELINES:
        d = (per_q["hall"][RAG] - per_q["hall"][b]).to_numpy()
        boots = rng.integers(0, d.size, size=(N_BOOT, d.size))
        bs = d[boots].sum(axis=1) / 80.0
        comps.append(dict(baseline=b, diff_pp=d.sum() / 80.0, ci_lo=float(np.percentile(bs, 2.5)),
                          ci_hi=float(np.percentile(bs, 97.5)), nonzero_pairs=int((d != 0).sum()),
                          p=signflip_pvalue(d, rng)))
    comps = pd.DataFrame(comps)
    comps["p_holm"] = multipletests(comps["p"], method="holm")[1]
    return hr, comps, r


def threshold_sensitivity(responses, thresholds=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)):
    rows = []
    for t in thresholds:
        for c in CONFIGS:
            s = responses.query("config == @c")["MA"]
            rows.append(dict(threshold=t, config=c, k=int((s < t).sum()), n=s.size, hr=float((s < t).mean())))
    return pd.DataFrame(rows)


def mixed_model_sensitivity(responses, reliability):
    import statsmodels.formula.api as smf
    import warnings
    out = []
    for crit in CRITERIA:
        d = (reliability if crit == "REL" else responses).copy()
        d["config"] = pd.Categorical(d["config"], categories=[RAG] + BASELINES)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(f"{crit} ~ C(config)", d, groups=d["question"]).fit(reml=True)
        for b in BASELINES:
            name = f"C(config)[T.{b}]"
            out.append(dict(criterion=crit, baseline=b, diff_rag_minus_base=-fit.params[name],
                            se=fit.bse[name], p=fit.pvalues[name]))
    df = pd.DataFrame(out)
    df["p_holm"] = multipletests(df["p"], method="holm")[1]
    return df


def anova_tukey_sensitivity(responses, reliability):
    """One-way ANOVA + Tukey HSD treating every response as independent (for comparability)."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    out_a, out_t = [], []
    for crit in CRITERIA:
        d = reliability if crit == "REL" else responses
        groups = [d.query("config == @c")[crit] for c in CONFIGS]
        F, p = stats.f_oneway(*groups)
        out_a.append(dict(criterion=crit, F=F, df1=len(CONFIGS) - 1, df2=len(d) - len(CONFIGS), p=p,
                          levene_bf_p=stats.levene(*groups, center="median").pvalue))
        tk = pairwise_tukeyhsd(d[crit], d["config"])
        res = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
        for _, row in res.iterrows():
            if RAG in (row["group1"], row["group2"]):
                other = row["group2"] if row["group1"] == RAG else row["group1"]
                sign = 1 if row["group1"] == other else -1  # meandiff = group2 - group1
                out_t.append(dict(criterion=crit, baseline=other, diff_rag_minus_base=sign * row["meandiff"],
                                  p_adj=row["p-adj"]))
    return pd.DataFrame(out_a), pd.DataFrame(out_t)


def run_agreement(responses, threshold=THRESHOLD):
    rows = []
    for c in CONFIGS:
        s = responses.query("config == @c").pivot(index="question", columns="iteration", values="MA")
        h1, h2 = s[1] < threshold, s[2] < threshold
        rho = stats.spearmanr(s[1], s[2]).correlation
        rows.append(dict(config=c, agree_pct=float((h1 == h2).mean() * 100),
                         both_hall=int((h1 & h2).sum()), one_hall=int((h1 ^ h2).sum()), spearman_ma=rho))
    return pd.DataFrame(rows)


def domain_breakdown(r_flagged):
    g = r_flagged.groupby(["domain", "config"]).agg(k=("hall", "sum"), n=("hall", "size"), ma=("MA", "mean"))
    g = g.reset_index()
    g["hr"] = g["k"] / g["n"]
    return g


# ----------------------------------------------------------------------------
# Exploratory: words in scripts other than Bangla and Latin
# ----------------------------------------------------------------------------
IGNORED_SCRIPTS = {"LATIN", "BENGALI", "COMBINING", "MODIFIER", "GREEK"}  # Greek: scientific symbols


def foreign_scripts(text: str) -> dict[str, int]:
    """Count letters and combining marks that belong to scripts other than Bengali and Latin.

    The script is taken from the first word of the Unicode character name
    (e.g. DEVANAGARI, CYRILLIC, TAMIL, MALAYALAM, CJK, HANGUL, KATAKANA, ARABIC).
    Digits, punctuation, symbols, spaces and format characters are ignored.
    """
    import unicodedata
    found: dict[str, int] = {}
    for ch in text:
        if unicodedata.category(ch)[0] not in "LM" or ch.isascii() or 0x0980 <= ord(ch) <= 0x09FF:
            continue
        script = unicodedata.name(ch, "UNKNOWN").split()[0]
        if script not in IGNORED_SCRIPTS:
            found[script] = found.get(script, 0) + 1
    return found


def script_mixing(responses_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(responses_csv)
    assert len(raw) == 400, len(raw)
    detail = []
    for _, r in raw.iterrows():
        found = foreign_scripts(str(r["response"]))
        if found:
            detail.append(dict(config=r["config"], question=int(r["question"]), iteration=int(r["iteration"]),
                               scripts="; ".join(f"{k} ({v})" for k, v in sorted(found.items()))))
    detail = pd.DataFrame(detail, columns=["config", "question", "iteration", "scripts"])
    summary = []
    for c in CONFIGS:
        d = detail.query("config == @c")
        n = int((raw["config"] == c).sum())
        scripts = sorted({s.split(" (")[0] for x in d["scripts"] for s in x.split("; ")})
        summary.append(dict(config=c, n=n, k=len(d), pct=len(d) / n * 100, scripts=", ".join(scripts)))
    return pd.DataFrame(summary), detail


# ----------------------------------------------------------------------------
# 3. Figures
# ----------------------------------------------------------------------------
PALETTE = {  # validated categorical slots (dataviz reference palette), fixed per entity
    "BanglaDiaRAG": "#2a78d6", "Qwen3.5-9B": "#eb6834", "TigerLLM-9B": "#1baf7a",
    "Qwen3.6-35B-A3B": "#eda100", "Gemma-4-26B-A4B": "#e87ba4"}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": False, "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})


def _save(fig, figdir, name):
    """Save PDF (vector preview) and a PLOS-compliant TIFF (RGB, LZW, 600 dpi)."""
    from PIL import Image
    fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    png = figdir / f"{name}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.03)
    im = Image.open(png).convert("RGB")
    w_in, h_in = im.size[0] / 600, im.size[1] / 600
    # PLOS: width 2.63-7.5 in, height <= 8.75 in, 300-600 dpi, < 10 MB
    assert 2.63 <= w_in <= 7.5 and h_in <= 8.75, (name, w_in, h_in)
    im.save(figdir / f"{name}.tif", compression="tiff_lzw", dpi=(600, 600))
    os.remove(png)


def fig_distributions(responses, reliability, figdir):
    import matplotlib.pyplot as plt
    _style()
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 5, figsize=(7.5, 2.9), sharey=True)
    for ax, crit in zip(axes, CRITERIA):
        d = reliability if crit == "REL" else responses
        data = [d.query("config == @c")[crit].to_numpy() for c in CONFIGS]
        for i, (c, v) in enumerate(zip(CONFIGS, data)):
            x = i + rng.uniform(-0.18, 0.18, size=v.size)
            ax.scatter(x, v, s=5, color=PALETTE[c], alpha=0.45, linewidths=0, zorder=2)
        bp = ax.boxplot(data, positions=range(len(CONFIGS)), widths=0.55, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, linewidth=1.1),
                        whiskerprops=dict(color=INK2, linewidth=0.7), capprops=dict(color=INK2, linewidth=0.7),
                        boxprops=dict(linewidth=0.7, edgecolor=INK2))
        for patch in bp["boxes"]:
            patch.set_facecolor("none")
        means = [v.mean() for v in data]
        ax.scatter(range(len(CONFIGS)), means, marker="D", s=14, color="white", edgecolors=INK,
                   linewidths=0.7, zorder=4)
        ax.axhline(THRESHOLD, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
        ax.set_title(CRITERION_LABELS[crit], color=INK, pad=4)
        ax.set_xticks(range(len(CONFIGS)))
        ax.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS], rotation=55, ha="right")
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Expert score (0-1)")
    axes[-1].text(4.55, THRESHOLD, "0.70", va="center", ha="left", fontsize=6.5, color=MUTED, clip_on=False)
    fig.tight_layout(w_pad=0.6)
    _save(fig, figdir, "Fig3")
    plt.close(fig)


def fig_hallucination(hr, dom, figdir):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    # (A) hallucination rate with Wilson CI
    order = CONFIGS
    y = np.arange(len(order))[::-1]
    for yi, c in zip(y, order):
        row = hr.set_index("config").loc[c]
        ax1.plot([row.wilson_lo * 100, row.wilson_hi * 100], [yi, yi], color=PALETTE[c], linewidth=1.6,
                 solid_capstyle="round")
        ax1.scatter(row.hr * 100, yi, s=34, color=PALETTE[c], edgecolors="white", linewidths=1.2, zorder=3)
        ax1.text(row.wilson_hi * 100 + 2.5, yi, f"{pct1(row.hr)}% ({int(row.k)}/{int(row.n)})",
                 va="center", fontsize=7, color=INK2)
    ax1.set_yticks(y)
    ax1.set_yticklabels([CONFIG_LABELS[c] for c in order])
    ax1.set_xlim(0, 118)
    ax1.set_xticks([0, 20, 40, 60, 80, 100])
    ax1.set_xlabel("Hallucination rate, % of responses (95% CI)")
    ax1.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title("A  Overall", loc="left", color=INK, fontweight="bold")
    # (B) sub-domain heatmap: neutral single-hue (lightness) ramp, so colour stays reserved for
    # configuration identity in the other panels and figures
    ramp = ["#ffffff", "#f0efec", "#e1e0d9", "#c3c2b7", "#a3a29b", "#898781", "#6b6a66", "#52514e", "#2c2c2a"]
    cmap = LinearSegmentedColormap.from_list("neutral_seq", ramp)
    dnames = [d[0] for d in DOMAINS]
    mat = np.array([[dom.set_index(["domain", "config"]).loc[(dn, c), "hr"] for c in CONFIGS] for dn in dnames])
    ax2.imshow(mat * 100, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i, dn in enumerate(dnames):
        for j, c in enumerate(CONFIGS):
            rec = dom.set_index(["domain", "config"]).loc[(dn, c)]
            v = rec["hr"] * 100
            ax2.text(j, i, f"{int(rec['k'])}/{int(rec['n'])}", ha="center", va="center", fontsize=6.8,
                     color="white" if v >= 50 else INK)
    ax2.set_xticks(range(len(CONFIGS)))
    ax2.set_xticklabels([CONFIG_LABELS[c] for c in CONFIGS], rotation=35, ha="right")
    ax2.set_yticks(range(len(dnames)))
    short = {"Dietary": "Dietary", "Medication and insulin": "Medication/insulin",
             "Lifestyle and exercise": "Lifestyle/exercise", "Complications and safety": "Complications/safety",
             "Symptoms and early detection": "Symptoms/detection", "Special situations": "Special situations"}
    ax2.set_yticklabels([f"{short[dn]} ({hi - lo + 1} Q)" for dn, lo, hi in DOMAINS])
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(length=0)
    ax2.set_xticks(np.arange(-0.5, len(CONFIGS), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(dnames), 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=2)
    ax2.tick_params(which="minor", length=0)
    ax2.set_title("B  By clinical sub-domain", loc="left", color=INK, fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    _save(fig, figdir, "Fig4")
    plt.close(fig)


def fig_differences(pw, figdir):
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 5, figsize=(7.5, 2.5), sharey=True)
    y = np.arange(len(BASELINES))[::-1]
    for ax, crit in zip(axes, CRITERIA):
        sub = pw.query("criterion == @crit").set_index("baseline")
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
        for yi, b in zip(y, BASELINES):
            r = sub.loc[b]
            sig = r.p_holm < 0.05
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=PALETTE[b], linewidth=1.6, solid_capstyle="round", zorder=2)
            ax.scatter(r["diff"], yi, s=30, zorder=3, linewidths=1.1,
                       color=PALETTE[b] if sig else "white", edgecolors=PALETTE[b])
        ax.set_title(CRITERION_LABELS[crit], color=INK, pad=4)
        ax.set_xlim(-0.26, 0.48)
        ax.set_xticks([-0.2, 0, 0.2, 0.4])
        ax.set_xticklabels(["\u22120.2", "0", "0.2", "0.4"])
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"vs {CONFIG_LABELS[b]}" for b in BASELINES])
    fig.supxlabel("Mean paired difference, BanglaDiaRAG minus baseline (95% bootstrap CI)", fontsize=8,
                  color=INK2)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor=INK2, markeredgecolor=INK2,
                      label="Holm-adjusted p < 0.05"),
               Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor="white", markeredgecolor=INK2,
                      label="Not significant")]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.995, 1.06))
    fig.tight_layout(w_pad=0.5, rect=(0, 0, 1, 0.95))
    _save(fig, figdir, "Fig5")
    plt.close(fig)


def fig_threshold(ts, figdir):
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for c in CONFIGS:
        s = ts.query("config == @c")
        ax.plot(s["threshold"], s["hr"] * 100, color=PALETTE[c], linewidth=1.6, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=CONFIG_LABELS[c])
    ax.axvline(THRESHOLD, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.text(THRESHOLD + 0.004, 101, "rubric boundary (0.70)", fontsize=6.5, color=MUTED, va="top")
    ax.set_xlabel("Medical accuracy threshold for classifying a response as hallucinated")
    ax.set_ylabel("Hallucination rate (%)")
    ax.set_ylim(0, 102)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    _save(fig, figdir, "S1_Fig")
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", default="data/Scoring_Table_BanglaDiaHallu.xlsx")
    ap.add_argument("--responses", default="data/model_responses.csv",
                    help="all 400 responses (for the script-mixing analysis); skipped if missing")
    ap.add_argument("--out", default="results")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    out, figdir = Path(args.out), Path(args.figdir)
    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    responses, reliability = load_scores(args.xlsx)
    qa = question_level(responses, reliability)
    responses.to_csv(out / "tidy_responses.csv", index=False)
    reliability.to_csv(out / "tidy_reliability.csv", index=False)

    desc = descriptives(responses, reliability)
    fr = friedman_tests(qa)
    pw = pairwise_vs_rag(qa, rng)
    pw_all = pairwise_all(qa)
    hr, hr_comp, flagged = hallucination(responses, rng)
    ts = threshold_sensitivity(responses)
    lmm = mixed_model_sensitivity(responses, reliability)
    anova, tukey = anova_tukey_sensitivity(responses, reliability)
    agree = run_agreement(responses)
    dom = domain_breakdown(flagged)

    for name, df in [("descriptives", desc), ("friedman", fr), ("pairwise_rag_vs_baselines", pw),
                     ("pairwise_all_configurations", pw_all), ("hallucination_rates", hr),
                     ("hallucination_rag_vs_baselines", hr_comp), ("threshold_sensitivity", ts),
                     ("sensitivity_mixed_model", lmm), ("sensitivity_oneway_anova", anova),
                     ("sensitivity_tukey_hsd", tukey), ("run_to_run_agreement", agree),
                     ("subdomain_breakdown", dom)]:
        df.to_csv(out / f"{name}.csv", index=False)

    if Path(args.responses).exists():
        mix, mix_detail = script_mixing(args.responses)
        mix.to_csv(out / "script_mixing.csv", index=False)
        mix_detail.to_csv(out / "script_mixing_responses.csv", index=False)
        print("\n=== Responses with scripts other than Bangla and Latin ===\n", mix.round(1))

    summary = {
        "n_questions": int(qa["question"].nunique()), "n_responses": int(len(responses)),
        "seed": SEED, "n_boot": N_BOOT, "n_perm": N_PERM, "threshold": THRESHOLD,
    }
    (out / "run_info.json").write_text(json.dumps(summary, indent=2))

    if not args.no_figures:
        fig_distributions(responses, reliability, figdir)
        fig_hallucination(hr, dom, figdir)
        fig_differences(pw, figdir)
        fig_threshold(ts, figdir)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n=== Descriptives (response level) ===")
    print(desc.pivot(index="criterion", columns="config", values="mean")[CONFIGS].round(3))
    print("\n=== Friedman ===\n", fr.round(4))
    print("\n=== BanglaDiaRAG vs baselines (question level) ===\n",
          pw[["criterion", "baseline", "diff", "ci_lo", "ci_hi", "r_rb", "p", "p_holm"]].round(4))
    print("\n=== Hallucination ===\n", hr.round(4))
    print("\n", hr_comp.round(4))
    print("\n=== Mixed model ===\n", lmm.round(4))
    print("\n=== One-way ANOVA ===\n", anova.round(4))
    print("\n=== Run-to-run agreement ===\n", agree.round(3))


if __name__ == "__main__":
    main()
