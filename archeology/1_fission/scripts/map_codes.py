"""Heuristic mapping of Ngogo male 2-letter codes (network files) to full names (GPS file).
Candidate pairs: name starts with code[0] and contains code[1] later (or initials of two-word names).
Score: Jaccard of active years 2011-2022 (network degree>0 vs GPS presence). Hungarian assignment.
Usage: python -I map_codes.py <ngogo_data_dir> <out_csv>"""
import sys, numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
D, out = sys.argv[1], sys.argv[2]
names = np.load(f'{D}/learned/L/data/chimps_names.npy')
males = list(names[:97])
years = range(2011, 2023)
act = {c: set() for c in males}
for y in years:
    if y == 2020: pass
    A = sum(np.nan_to_num(np.load(f'{D}/learned/L/data/learned_graphs/full_array_{y}_{m}.npy')) for m in range(1, 13))
    d = (A + A.T > 0).sum(1)
    for i, c in enumerate(males):
        if d[i] > 0: act[c].add(y)
g = pd.read_csv(f'{D}/space/space-use-archive/data/gps_qualified.csv')
gm = g[g.Sex == 'M']
gy = gm.groupby('Individual').Year.apply(lambda s: set(s) & set(years)).to_dict()
gnames = list(gy)
def cand(code, name):
    n = name.lower().replace(' ', '')
    parts = name.lower().split()
    if len(parts) == 2 and code == parts[0][0] + parts[1][0]: return True
    return n[0] == code[0] and code[1] in n[1:]
def strong(code, name):
    n = name.lower().replace(' ', '')
    return n[:2] == code or (n[0] == code[0] and n[1:].find(code[1]) >= 0 and n[1:].find(code[1]) <= 3)
S = np.full((len(males), len(gnames)), -10.0)
for i, c in enumerate(males):
    for j, nm in enumerate(gnames):
        if cand(c, nm):
            a, b = act[c], gy[nm]
            jac = len(a & b) / max(1, len(a | b))
            S[i, j] = jac + (0.3 if strong(c, nm) else 0) + (0.2 if nm.lower()[:2] == c else 0)
r, cidx = linear_sum_assignment(-S)
rows = []
for i, j in zip(r, cidx):
    if S[i, j] > 0.3:
        a, b = act[males[i]], gy[gnames[j]]
        rows.append(dict(code=males[i], name=gnames[j], score=round(S[i, j], 2),
                         jaccard=round(len(a & b) / max(1, len(a | b)), 2),
                         net_years=f'{min(a) if a else "-"}-{max(a) if a else "-"}',
                         gps_years=f'{min(b) if b else "-"}-{max(b) if b else "-"}'))
df = pd.DataFrame(rows).sort_values('code')
df.to_csv(out, index=False)
print(df.to_string())
print('mapped', len(df), 'of', len(males), 'codes;', len(gnames), 'GPS males')
