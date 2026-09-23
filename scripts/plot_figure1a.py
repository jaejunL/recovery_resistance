"""Figure 1(a) of the paper: the 2x2 factorial vs lambda (tabular Q-learning) with DQN on S_down.

    python scripts/plot_figure1a.py [--tabular factorial] [--dqn dqn] [--out results/figure1a.png]
Uses reference values for any series that has not been run yet (--use-reference).
"""
import os, json, argparse, collections
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from _common import ROOT, RESULTS, REFERENCE, LAMS_SWEEP, load_results
from gridworld.stats import seed_ci
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 160, 'savefig.bbox': 'tight'})


def cells_from(name, prefix, use_reference, ref_file):
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    rows = load_results(name)
    if rows:
        for r in rows:
            cells[r['cfg']['arm']][float(r['cfg']['lam'])].append(r['rho_int'])
    elif use_reference:
        ref = json.load(open(os.path.join(REFERENCE, ref_file)))
        for key, cell in ref.items():
            if key.startswith('_'):
                continue
            arm, lam = key.rsplit('_l', 1)
            cells[arm][float(lam)] = [v['rho_int'] for v in cell['seeds'].values()]
    return cells


def errs(vs):
    e = [seed_ci(v) for v in vs]
    return np.array([[m - lo for m, lo, _ in e], [hi - m for m, _, hi in e]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tabular', default='factorial'); ap.add_argument('--dqn', default='dqn')
    ap.add_argument('--out', default=os.path.join(RESULTS, 'figure1a.png'))
    ap.add_argument('--use-reference', action='store_true', help='fall back to reference values for missing series')
    a = ap.parse_args()
    tab = cells_from(a.tabular, 'q', a.use_reference, 'tabular_env1.json')
    dqn = cells_from(a.dqn, 'dqn', a.use_reference, 'dqn_env1.json')
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    style = [('S_down', 'S$_{\\rm down}$', '#b03a2e', 'o', '-'), ('H_down', 'H$_{\\rm down}$', '#e08a7a', 's', '-'),
             ('S_maint', 'S$_{\\rm maint}$', '#1f5fa8', '^', 'none'), ('H_maint', 'H$_{\\rm maint}$', '#8fb4e3', 'D', 'none')]
    for arm, lab, col, mk, ls in style:
        lams = [l for l in LAMS_SWEEP if tab[arm].get(float(l))]
        if not lams: continue
        m = [np.mean(tab[arm][float(l)]) for l in lams]
        ax.errorbar(lams, m, yerr=errs([tab[arm][float(l)] for l in lams]), color=col, marker=mk, ls=ls, ms=4.5, lw=1.4, elinewidth=0.7, capsize=2, label=lab, mfc=('none' if arm == 'S_maint' else col))
    if dqn['S_down']:
        dl = sorted(dqn['S_down'])
        ax.errorbar(dl, [np.mean(dqn['S_down'][l]) for l in dl], yerr=errs([dqn['S_down'][l] for l in dl]), color='#8c8c8c', marker='o', ms=4.5, ls='--', elinewidth=0.7, capsize=2, label='DQN, S$_{\\rm down}$')
    ax.set_xlabel(r'recovery-objective strength $\lambda$'); ax.set_ylabel(r'resistance $\rho_{\rm int}$')
    ax.set_ylim(-0.03, 1.06); ax.set_xticks([0, 1, 2, 3, 4, 6, 8]); ax.legend(frameon=False, fontsize=7.5, loc='center right')
    ax.set_title(r'2$\times$2 factorial vs. $\lambda$ (tabular Q; grey: DQN on S$_{\rm down}$)', fontsize=9)
    os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out); print('saved', a.out)


if __name__ == '__main__':
    main()
