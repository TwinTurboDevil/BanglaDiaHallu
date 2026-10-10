"""Generate the LaTeX rows of Tables 4-6 of the manuscript from the analysis outputs.

Means are recomputed exactly from the two-decimal scores and all values are rounded
half up, so the printed tables match the numbers quoted in the text.

Usage (from the repository root, after analysis/Statistical_Analysis.py):
    python analysis/make_tables.py > results/latex_table_rows.tex
"""
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
CONFIGS = ["Qwen3.5-9B","TigerLLM-9B","Qwen3.6-35B-A3B","Gemma-4-26B-A4B","BanglaDiaRAG"]
LAB = {"MA":"Medical accuracy","COMP":"Completeness","SAFE":"Safety","READ":"Readability","REL":"Reliability"}
def f(x, nd=3):
    q = Decimal(1).scaleb(-nd)
    # remove floating-point noise first (0.04949999999999998 -> 0.0495), then round half up
    d = Decimal(repr(round(float(x), 9))).quantize(q, rounding=ROUND_HALF_UP)
    s = f"{d:.{nd}f}"
    return s.replace('-', '$-$') if s.startswith('-') else s
def exact_mean(vals):
    vals=[Decimal(str(round(v,2))) for v in vals]; return sum(vals)/Decimal(len(vals))
r = pd.read_csv('results/tidy_responses.csv'); rel = pd.read_csv('results/tidy_reliability.csv')
desc = pd.read_csv('results/descriptives.csv'); fr = pd.read_csv('results/friedman.csv').set_index('criterion')
print('% Table 4 rows (scores)')
for c in CONFIGS:
    cells=[]
    for crit in ['MA','COMP','SAFE','READ','REL']:
        vals=(rel if crit=='REL' else r).query('config==@c')[crit].tolist()
        m=exact_mean(vals); sd=desc.query('config==@c and criterion==@crit')['sd'].iloc[0]
        cells.append(f"{f(m)} ({f(sd)})")
    print(f"{c} & " + " & ".join(cells) + " \\\\ \\hline")
print("\\thickhline")
print("Friedman $\\chi^{2}(4)$ & " + " & ".join(f(fr.loc[c,'chi2'],1) for c in ['MA','COMP','SAFE','READ','REL']) + " \\\\ \\hline")
print("Kendall's $W$ & " + " & ".join(f(fr.loc[c,'kendalls_w'],2) for c in ['MA','COMP','SAFE','READ','REL']) + " \\\\ \\hline")
pw = pd.read_csv('results/pairwise_rag_vs_baselines.csv')
print('% Table 6 rows (paired comparisons)')
for crit in ['MA','COMP','SAFE','READ','REL']:
    sub = pw[pw.criterion==crit]
    first=True
    for _,x in sub.iterrows():
        # exact mean diff
        if crit=='REL':
            a=rel.query('config=="BanglaDiaRAG"')[crit].tolist(); b=rel.query('config==@x.baseline')[crit].tolist()
        else:
            a=r.query('config=="BanglaDiaRAG"')[crit].tolist(); b=r.query('config==@x.baseline')[crit].tolist()
        d=exact_mean(a)-exact_mean(b)
        p = '$<0.001$' if x.p_holm < 0.001 else f(x.p_holm)
        print(f"{LAB[crit] if first else ''} & {x.baseline} & {f(d)} ({f(x.ci_lo)} to {f(x.ci_hi)}) & {x.wins} / {x.ties} / {x.losses} & {f(x.r_rb,2)} & {p} \\\\" + (" \\hline" if x.baseline=='Gemma-4-26B-A4B' else ""))
        first=False
hr = pd.read_csv('results/hallucination_rates.csv').set_index('config')
hc = pd.read_csv('results/hallucination_rag_vs_baselines.csv').set_index('baseline')
print('% Table 5 rows (hallucination)')
for c in CONFIGS:
    x=hr.loc[c]
    if c!='BanglaDiaRAG':
        y=hc.loc[c]; diff=f"{f(-y.diff_pp*100,1)} ({f(-y.ci_hi*100,1)} to {f(-y.ci_lo*100,1)})"
    else: diff='--'
    print(f"{c} & {int(x.k)}/{int(x.n)} ({f(x.hr*100,1)}) & {f(x.wilson_lo*100,1)}--{f(x.wilson_hi*100,1)} & {f(x.cluster_lo*100,1)}--{f(x.cluster_hi*100,1)} & {int(x.questions_affected)}/40 & {diff} & {int(x.unsafe_k)}/{int(x.n)} ({f(x.unsafe_rate*100,1)}) \\\\ \\hline")
print('% HR Holm p:', hc.p_holm.round(4).to_dict(), 'raw', hc.p.round(4).to_dict())
