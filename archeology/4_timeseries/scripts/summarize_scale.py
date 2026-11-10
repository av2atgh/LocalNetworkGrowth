import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES
thr = sys.argv[1] if len(sys.argv) > 1 else '0.75'
out = pickle.load(open(os.path.join(RES, f'scale_thr{thr}.pkl'), 'rb'))
L = [f'# threshold {thr}', '## A. slices as observed (n = sites in slice; nLCC; B; Sigma=(DL_B1-DL)/E). Nulls: P(B>=2), mean Sigma over 20 samples']
L.append(f"{'ds':6s} {'per':>5s} {'n':>4s} {'dens':>5s} {'nLCC':>4s} {'B':>3s} {'Sig':>6s} | " + ' | '.join(f'{k:>16s}' for k in ['dist', 'attr', 'dist_attr', 'config']))
for r in sorted(out, key=lambda r: (r['ds'], r['per'])):
    o = r['obs']
    cells = []
    for k in ['dist', 'attr', 'dist_attr', 'config']:
        B = np.array([x['B'] for x in r[k]]); S = np.array([x['sig'] for x in r[k]])
        cells.append(f'{np.mean(B>=2):4.2f} B{np.median(B):3.0f} S{S.mean():5.3f}')
    L.append(f"{r['ds']:6s} {r['per']:5d} {r['n']:4d} {r['density']:5.3f} {o['n']:4d} {o['B']:3d} {o['sig']:6.3f} | " + ' | '.join(cells))
L.append('\n## B. nested subsets within large slices: P(B>=2) [mean Sigma] vs subset size')
for r in sorted(out, key=lambda r: (r['ds'], r['per'])):
    if not r['sub']: continue
    for mode in ['knn', 'rand']:
        rows = [x for x in r['sub'] if x['mode'] == mode]
        sizes = sorted(set(x['size'] for x in rows))
        L.append(f"{r['ds']} {r['per']} n={r['n']} subsets={mode}")
        for k in ['obs', 'dist', 'dist_attr', 'config']:
            s = []
            for z in sizes:
                rr = [x[k] for x in rows if x['size'] == z]
                s.append(f"{np.mean([q['B']>=2 for q in rr]):.2f}[{np.mean([q['sig'] for q in rr]):.2f}]")
            L.append(f"   {k:9s} " + ' '.join(f'{z:>4d}:{c}' for z, c in zip(sizes, s)))
txt = '\n'.join(L); print(txt)
open(os.path.join(RES, f'scale_thr{thr}.txt'), 'w').write(txt + '\n')
