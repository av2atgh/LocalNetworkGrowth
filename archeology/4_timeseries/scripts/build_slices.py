"""Build per-slice similarity matrices + coords + attributes for Chaco (Mills et al. 2018 apportioned
wares, 50-yr slices AD 800-1300) and SWSN (Mills et al. 2013 BR matrices AD 1200-1450).
Output: results/slices.pkl  {dataset: {period: dict(sites, S, xy, attr)}}"""
import sys, pickle, numpy as np, pandas as pd, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, RES, br_similarity
from swsn_load import load_all, PERIODS

out = {}
# ---- Chaco ----
c = pd.read_csv(os.path.join(DATA, 'chaco_timeavg', 'AllCeramics.csv'), index_col=0)
at = pd.read_csv(os.path.join(DATA, 'Archaeological-Network-Uncertainty', 'AD1050attr.csv'),
                 encoding='latin-1').drop_duplicates('SWSN_ID').set_index('SWSN_ID')
P = [x for x in c.columns if x.startswith('P')]
ch = {}
for k in range(0, len(P), 2):
    per = int(P[k][1:])
    tab = c.assign(v=c[P[k]] + c[P[k + 1]]).pivot_table(index='SWSN_ID', columns='SWSN_Ware', values='v', aggfunc='sum', fill_value=0)
    tab = tab[tab.sum(1) >= 10]
    tab = tab[tab.index.isin(at.index)]
    tab = tab.loc[:, tab.sum(0) > 0]
    X = tab.values.astype(float)
    ch[per] = dict(sites=np.array(tab.index), S=br_similarity(X),
                   xy=at.loc[tab.index, ['EASTING', 'NORTHING']].values / 1000.0,
                   attr=at.loc[tab.index, 'CSN_macro_group'].fillna('NA').values,
                   gh=(at.loc[tab.index, 'GH_Count'].fillna(0).values > 0).astype(int), counts=X)
    print('Chaco', per, len(tab), 'wares', X.shape[1])
out['chaco'] = ch
# ---- SWSN ----
a, S = load_all()
sw = {}
for p in PERIODS:
    s = S[p]
    sw[p] = dict(sites=np.array(s.index), S=s.values.astype(float),
                 xy=a.loc[s.index, ['EASTING', 'NORTHING']].values / 1000.0,
                 attr=a.loc[s.index, 'Macro'].values, micro=a.loc[s.index, 'Micro'].values)
    print('SWSN', p, len(s))
out['swsn'] = sw
pickle.dump(out, open(os.path.join(RES, 'slices.pkl'), 'wb'))
