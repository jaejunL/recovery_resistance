"""Summarize a results directory: per-cell seed means and 95% seed-bootstrap intervals, the
dose-response Spearman statistic for S_down, and a summary.json.

    python scripts/summarize.py factorial
    python scripts/summarize.py dqn
"""
import os, sys, json, collections
import numpy as np
from _common import RESULTS, load_results
from gridworld.stats import seed_ci, hier_bootstrap_spearman


def main(name):
    rows = load_results(name)
    if not rows:
        sys.exit(f'no results in results/{name}')
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r['cfg']['arm'], float(r['cfg']['lam']))].append(r)
    summ = {}
    print(f"{'arm':8s} {'lam':>4s} {'n':>3s} {'rho_int':>8s} {'95% CI':>16s} {'rho^goal':>9s} {'ret_E2':>7s} {'recov':>6s} {'press_E3':>8s}")
    for (arm, lam), rs in sorted(cells.items()):
        rs = sorted(rs, key=lambda r: r['cfg']['seed'])
        rho = [r['rho_int'] for r in rs]
        m, lo, hi = seed_ci(rho)
        recov = [r['repaired'] if arm.startswith('S') else r['calls'] for r in rs]
        c = dict(n=len(rs), rho_int=round(m, 3), rho_ci=[round(lo, 3), round(hi, 3)],
                 rho_seeds=[round(x, 3) for x in rho],
                 rho_goal=round(float(np.mean([r['rho_goal_E1'] for r in rs])), 3),
                 ret_dmg=round(float(np.mean([r['ret_dmg'] for r in rs])), 2),
                 recovery=round(float(np.mean(recov)), 3),
                 press_clean=round(float(np.mean([r['press_clean'] for r in rs])), 3))
        summ[f'{arm}_l{lam}'] = c
        print(f"{arm:8s} {lam:4g} {c['n']:3d} {c['rho_int']:8.3f} [{lo:6.3f}, {hi:6.3f}] {c['rho_goal']:9.3f} {c['ret_dmg']:7.2f} {c['recovery']:6.3f} {c['press_clean']:8.3f}")
    S = {lam: [r['rho_int'] for r in rs] for (arm, lam), rs in cells.items() if arm == 'S_down'}
    if len(S) >= 3:
        pt, (lo, hi) = hier_bootstrap_spearman(S)
        summ['spearman_S_down'] = dict(point=round(pt, 3), ci=[round(lo, 3), round(hi, 3)], n_lam=len(S))
        print(f"\nS_down dose-response: Spearman(lambda, seed-mean rho) = {pt:.3f}, hierarchical seed-bootstrap 95% interval [{lo:.3f}, {hi:.3f}]")
    json.dump(summ, open(os.path.join(RESULTS, name, 'summary.json'), 'w'), indent=1)
    print(f'wrote results/{name}/summary.json')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'factorial')
