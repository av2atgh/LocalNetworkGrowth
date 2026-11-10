"""Main figures for manuscript.tex: two panels each, essentials only. Reads saved results only."""
import os, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
A = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(A, 'figs_ms')
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'sans-serif', 'pdf.fonttype': 42, 'axes.linewidth': .6,
                     'xtick.major.width': .6, 'ytick.major.width': .6, 'legend.frameon': False})
BLUE, RED, GREY, DARK = '#1f5fa0', '#c8403a', '#9a9a9a', '#333333'
W, H = 7.2, 2.7

def label(ax, s):
    ax.text(-0.17, 1.08, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')

def endlabel(ax, x, y, s, c, dy=0):
    ax.text(x, y + dy, ' ' + s, color=c, va='center', fontsize=7.5)

# ---------------- Fig 1: what can be learned from small societies
pw = pd.read_csv(os.path.join(A, '0_power/data/power.csv'))
pw = pw[(pw.dense == 0) & (pw.h == 0.8) & (pw.K == 2) & (pw.contrast == 'LSc0_vs_LSc1')]
rs = pd.read_csv(os.path.join(A, '0_power/data/results.csv'))
rs = rs[(rs.dense == 0) & (rs.h.isin([0.8])) & (rs.K == 2)]
fig, ax = plt.subplots(1, 2, figsize=(W, H))
TK = [20, 50, 100, 200, 500]
a = ax[0]
for st, c in [('conv', BLUE), ('align_ari', GREY), ('b_same', RED)]:
    t = pw[pw.stat == st].sort_values('n')
    a.plot(t.n, t.auc, 'o-', c=c, ms=3.5, lw=1.4)
a.text(100, .99, 'temporal order', color=BLUE, fontsize=7.5, ha='right', va='bottom')
a.text(110, .36, 'community–trait\noverlap', color=GREY, fontsize=7.5, ha='left', va='center')
a.text(24, .17, 'trait similarity\nof ties', color=RED, fontsize=7.5, ha='left', va='top')
a.axhline(.5, c=DARK, lw=.5, ls=':'); a.text(560, .52, 'chance', fontsize=7, color=DARK, ha='right', va='bottom')
a.set_title(r'$\bf{a}$   traits spread along ties', fontsize=8, loc='left')
a.set_xscale('log'); a.set_xticks(TK); a.set_xticklabels(TK); a.minorticks_off()
a.set_xlim(16, 620); a.set_ylim(-.03, 1.08)
a.set_xlabel('group size'); a.set_ylabel('power to identify mechanism (AUC)')
b = ax[1]
for lam, c in [(0.0, BLUE), (1.0, RED)]:
    t = rs[(rs.family == 'LS') & (rs.lam == lam)].groupby('n').detect.mean()
    b.plot(t.index, t.values, 'o-', c=c, ms=3.5, lw=1.4)
b.text(250, .92, 'connectivity', color=BLUE, fontsize=7.5, ha='center', va='top')
b.text(200, .33, 'homophily', color=RED, fontsize=7.5, ha='center', va='bottom')
b.axvspan(80, 100, color='#e8e8e8', zorder=0); b.text(90, 1.04, r'$r_\kappa$', ha='center', va='bottom', fontsize=8)
b.set_xscale('log'); b.set_xticks(TK); b.set_xticklabels(TK); b.minorticks_off()
b.set_xlim(16, 620); b.set_ylim(-.03, 1.13)
b.set_xlabel('group size'); b.set_ylabel('probability of detecting communities')
b.set_title(r'$\bf{b}$   detectable communities', fontsize=8, loc='left')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig1.pdf')); plt.close(fig)

# ---------------- Fig 2: not sex or age
t = pd.read_csv(os.path.join(A, '1_fission/results/ngogo_temporal.csv'))
t = t[(t.n_lab1 > 0) & (t.n_eval >= 20)]
fig, ax = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={'width_ratios': [1.35, 1]})
a = ax[0]
a.axvspan(2015, 2017.4, color='#ececec', zorder=0); a.text(2016.2, .14, 'split', ha='center', fontsize=7)
a.plot(t.year, t.neighvote_balacc, 'o-', c=BLUE, ms=3, lw=1.3)
s = t.dropna(subset=['space_balacc'])
a.plot(s.year, s.space_balacc, 's-', c=DARK, ms=3, lw=1.3)
a.plot(s.year, s.attr_balacc, 'v-', c=RED, ms=3, lw=1.3)
a.text(2003.2, .87, 'social network', color=BLUE, fontsize=7.5)
a.text(2010.6, 1.035, 'ranging area', color=DARK, fontsize=7.5)
a.text(2010.6, .36, 'sex and age', color=RED, fontsize=7.5)
a.axhline(.5, c=DARK, lw=.5, ls=':')
a.set_ylim(.1, 1.08); a.set_xlim(2002.5, 2017.5); a.set_xticks(range(2003, 2018, 2))
a.set_xlabel('year'); a.set_ylabel('accuracy predicting later group')
a.set_title(r'$\bf{a}$   Ngogo chimpanzees: who went where', fontsize=8, loc='left')
b = ax[1]
vals = {'Arang Dak': [.647, .707, .829], 'Agta': [.56, .57, .78]}
cats = [('sex and age', RED), ('kinship', '#d9a441'), ('network structure', BLUE)]
x0 = np.arange(len(vals)); wd = .26
for k, (nm, c) in enumerate(cats):
    v = [vals[g][k] for g in vals]
    b.bar(x0 + (k - 1) * wd, np.array(v) - .5, bottom=.5, width=wd * .92, color=c, label=nm)
b.axhline(.5, c=DARK, lw=.5, ls=':')
b.set_xticks(x0); b.set_xticklabels(list(vals)); b.set_ylim(.45, .9)
b.set_ylabel('accuracy predicting ties (AUC)'); b.legend(fontsize=7, loc='upper right')
b.set_title(r'$\bf{b}$   human support networks', fontsize=8, loc='left')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2.pdf')); plt.close(fig)

# ---------------- Fig 3: arrow of time in Chaco
d = pickle.load(open(os.path.join(A, '4_timeseries/results/dynamics_thr0.75_null200.pkl'), 'rb'))
d = [r for r in d if r['ds'] == 'chaco' and r['obs']['n'] >= 30]
d.sort(key=lambda r: (r['p1'] - r['p0'], r['p0']))
fig, ax = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={'width_ratios': [1.35, 1]})
a = ax[0]; x = np.arange(len(d))
for key, c, off, mk, nm in [('beta_fwd', BLUE, -.14, 'o', 'forward in time'), ('beta_bwd', RED, .14, 'o', 'time reversed')]:
    v = np.array([r['obs'][key] for r in d]); ci = np.array([r['obs']['boot'][key] for r in d])
    a.errorbar(x + off, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt=mk, c=c, ms=4, lw=.9, capsize=0, label=nm)
a.axhline(0, c=DARK, lw=.5)
a.axvline(3.5, c=GREY, lw=.5, ls='--')
a.text(1.5, .33, '50 years apart', ha='center', fontsize=7, color=DARK)
a.text(5.5, .33, '100 years apart', ha='center', fontsize=7, color=DARK)
a.set_xticks(x); a.set_xticklabels([f"{r['p0']}" for r in d], fontsize=7)
a.set_xlabel('starting date (AD)'); a.set_ylabel('convergence within communities')
a.set_ylim(-.62, .4); a.legend(fontsize=7, loc='lower left')
a.set_title(r'$\bf{a}$   Chaco: similarity gained after sharing a community', fontsize=8, loc='left')
b = ax[1]
nd = np.load(os.path.join(A, '4_timeseries/results/dynamics_joint_null500_draws.npy'))[:, 0]
obs = float(np.mean([r['obs']['beta_fwd'] - r['obs']['beta_bwd'] for r in d]))
import json
obs = json.load(open(os.path.join(A, '4_timeseries/results/dynamics_joint_null500.json')))['asym_all']['obs']
b.hist(nd, bins=30, color=GREY, edgecolor='white', lw=.3)
b.axvline(obs, c=BLUE, lw=2)
b.set_xlim(nd.min() - .005, obs + .03)
b.text(obs - .006, b.get_ylim()[1] * .92, 'observed\n$P<0.002$', ha='right', va='top', color=BLUE, fontsize=7.5)
b.set_title(r'$\bf{b}$   all intervals vs stationary null', fontsize=8, loc='left')
b.text(np.quantile(nd, .01), b.get_ylim()[1] * .95, 'no change\nover time', ha='right', va='top', color=DARK, fontsize=7)
b.set_xlabel('forward minus time-reversed'); b.set_ylabel('null realisations'); b.set_yticks([])
b.spines['left'].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig3.pdf')); plt.close(fig)
print('ok')
