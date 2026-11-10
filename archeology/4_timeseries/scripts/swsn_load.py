"""Loader for SWSN_2013 similarity matrices (Mills et al. 2013 PNAS)."""
import numpy as np, pandas as pd, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'SWSN_2013')
PERIODS = [1200, 1250, 1300, 1350, 1400]

def load_attr():
    a = pd.read_csv(os.path.join(D, 'attributes.csv'))
    a = a.drop_duplicates('SWSN_Site').set_index('SWSN_Site')
    return a

def load_sim(p):
    s = pd.read_csv(os.path.join(D, f'AD{p}sim.csv'), index_col=0)
    s.columns = s.index
    return s

def load_all():
    a = load_attr()
    out = {}
    for p in PERIODS:
        s = load_sim(p)
        keep = [x for x in s.index if x in a.index]
        s = s.loc[keep, keep]
        out[p] = s
    return a, out
