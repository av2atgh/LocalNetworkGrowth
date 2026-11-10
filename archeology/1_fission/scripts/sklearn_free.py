import numpy as np
from math import comb
def ari(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    C = np.zeros((len(ua), len(ub)), int)
    np.add.at(C, (ia, ib), 1)
    s = sum(comb(int(x), 2) for x in C.ravel())
    sa = sum(comb(int(x), 2) for x in C.sum(1)); sb = sum(comb(int(x), 2) for x in C.sum(0))
    n2 = comb(len(a), 2); e = sa * sb / n2; mx = (sa + sb) / 2
    return (s - e) / (mx - e) if mx != e else 0.0
