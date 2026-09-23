"""Appendix F figure: DQN counterpart of Figure 1(a) (four arms vs lambda) and S_down tabular vs DQN.

    python scripts/plot_dqn_arms.py [--dqn dqn] [--tabular factorial] [--use-reference]
"""
import os, argparse
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from _common import RESULTS, LAMS_SWEEP
from plot_figure1a import cells_from, errs
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 160, 'savefig.bbox': 'tight'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dqn', default='dqn'); ap.add_argument('--tabular', default='factorial')
    ap.add_argument('--out', default=os.path.join(RESULTS, 'dqn_arms.png')); ap.add_argument('--use-reference', action='store_true')
    a = ap.parse_args()
    dqn = cells_from(a.dqn, 'dqn', a.use_reference, 'dqn_env1.json'); tab = cells_from(a.tabular, 'q', a.use_reference, 'tabular_env1.json')
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 3.1)); fig.subplots_adjust(wspace=0.3)
    style = [('S_down', 'S$_{\\rm down}$', '#b03a2e', 'o', '-'), ('H_down', 'H$_{\\rm down}$', '#e08a7a', 's', '-'),
             ('S_maint', 'S$_{\\rm maint}$', '#1f5fa8', '^', 'none'), ('H_maint', 'H$_{\\rm maint}$', '#8fb4e3', 'D', 'none')]
    for arm, lab, col, mk, ls in style:
        lams = [l for l in LAMS_SWEEP if dqn[arm].get(float(l))]
        if not lams: continue
        ax[0].errorbar(lams, [np.mean(dqn[arm][float(l)]) for l in lams], yerr=errs([dqn[arm][float(l)] for l in lams]), color=col, marker=mk, ls=ls, ms=4.5, lw=1.4, elinewidth=0.7, capsize=2, label=lab, mfc=('none' if arm == 'S_maint' else col))
    ax[0].set_title('(a) DQN: 2$\\times$2 factorial vs. $\\lambda$', fontsize=9); ax[0].legend(frameon=False, fontsize=7.5, loc='center right')
    for cells, lab, col, ls in ((tab, 'tabular Q', '#b03a2e', '-'), (dqn, 'DQN', '#8c8c8c', '--')):
        lams = [l for l in LAMS_SWEEP if cells['S_down'].get(float(l))]
        if lams:
            ax[1].errorbar(lams, [np.mean(cells['S_down'][float(l)]) for l in lams], yerr=errs([cells['S_down'][float(l)] for l in lams]), color=col, marker='o', ls=ls, ms=4.5, lw=1.4, elinewidth=0.7, capsize=2, label=lab)
    ax[1].set_title('(b) S$_{\\rm down}$: tabular Q vs DQN', fontsize=9); ax[1].legend(frameon=False, fontsize=7.5, loc='lower right')
    for x in ax:
        x.set_xlabel(r'recovery-objective strength $\lambda$'); x.set_ylabel(r'resistance $\rho_{\rm int}$'); x.set_ylim(-0.03, 1.06); x.set_xticks([0, 1, 2, 3, 4, 6, 8])
    os.makedirs(os.path.dirname(a.out), exist_ok=True); fig.savefig(a.out); print('saved', a.out)


if __name__ == '__main__':
    main()
