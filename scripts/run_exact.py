"""Exact value iteration on Env-1 (Appendix D): where disabling the interruption becomes reward-optimal.

    python scripts/run_exact.py --validate 4                 # model vs simulator (uniform-random policy), lambda=4
    python scripts/run_exact.py --preset phase               # Figure 2: 18 lambdas, clairvoyant and press-first advantage
    python scripts/run_exact.py --preset cstar               # Section 3.4 / Figure 2c: penalty c*(lambda) at lambda=4, 6
    python scripts/run_exact.py --lams 4,6 --arm H_down      # any arm / lambda grid
    python scripts/run_exact.py --lams 2.5 --observable      # press-first vs press-if-healthy bracket (Appendix D)
Results are written to results/exact/<name>/ (one JSON per cell, skipped if present) and compared with the
paper's values in reference/exact_env1.json where available. One cell takes a few minutes on one core.
"""
import os, json, argparse
from multiprocessing import Pool
from _common import ROOT, RESULTS, REFERENCE

PHASE_LAMS = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 5, 6, 8]
CSTAR_GRID = [(lam, c) for lam in (4.0, 6.0) for c in (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40)]


def one(args):
    kind, kw, path = args
    if os.path.exists(path):
        return json.load(open(path))
    from gridworld import exact
    r = exact.solve_observable_family(**kw) if kind == 'observable' else exact.solve_lambda(**kw)
    json.dump(r, open(path, 'w'), indent=1)
    print(os.path.basename(path), {k: r[k] for k in ('advantage', 'adv_pressfirst', 'rho_star') if k in r}, flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=['phase', 'cstar'])
    ap.add_argument('--lams'); ap.add_argument('--arm', default='S_down')
    ap.add_argument('--press-penalty', type=float, default=0.0)
    ap.add_argument('--observable', action='store_true', help='press-first vs press-if-healthy (no clairvoyant solve)')
    ap.add_argument('--validate', type=float, metavar='LAM', help='compare the model with the simulator at this lambda')
    ap.add_argument('--name'); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.validate is not None:
        from gridworld import exact
        r = exact.validate(arm=a.arm, lam=a.validate)
        print(json.dumps(r, indent=1)); print('PASS' if abs(r['z']) < 3 else 'FAIL', '(|z| < 3 expected)')
        return
    if a.preset == 'phase':
        cells = [dict(lam=float(l), arm='S_down') for l in PHASE_LAMS]; kind = 'solve'; name = a.name or 'phase'
    elif a.preset == 'cstar':
        cells = [dict(lam=l, arm='S_down', press_penalty=float(c)) for l, c in CSTAR_GRID]; kind = 'solve'; name = a.name or 'cstar'
    else:
        if not a.lams:
            ap.error('--lams, --preset or --validate is required')
        cells = [dict(lam=float(l), arm=a.arm, press_penalty=a.press_penalty) for l in a.lams.split(',')]
        kind = 'observable' if a.observable else 'solve'; name = a.name or (a.arm + ('_observable' if a.observable else ''))
        if a.observable:
            for c in cells: c.pop('press_penalty') if a.press_penalty == 0 else None
    out = os.path.join(RESULTS, 'exact', name); os.makedirs(out, exist_ok=True)
    jobs = [(kind, c, os.path.join(out, f"{c['arm']}_l{c['lam']}_c{c.get('press_penalty', 0.0)}{'_obs' if kind == 'observable' else ''}.json")) for c in cells]
    print(f'{len(jobs)} cells -> {out} ({a.workers} workers)', flush=True)
    with Pool(min(a.workers, len(jobs))) as pool:
        res = pool.map(one, jobs, chunksize=1)
    ref = json.load(open(os.path.join(REFERENCE, 'exact_env1.json')))
    if kind == 'observable':
        print(f"\n{'lam':>5} {'adv_pf':>8} {'adv_ph':>8}")
        for r in res:
            print(f"{r['lam']:>5} {r['adv_pressfirst']:>8.3f} {r['adv_press_if_healthy']:>8.3f}")
        return
    print(f"\n{'lam':>5} {'c':>4} {'adv_clair':>10} {'adv_pf':>8} {'rho*':>6}   paper: adv_clair / adv_pf")
    for r in res:
        key = f"{r['arm']}_l{r['lam']}_c{r['press_penalty']}"
        p = ref['cells'].get(key)
        ps = f"{p['advantage']:.3f} / {p['adv_pressfirst']:.3f}" if p else '-'
        print(f"{r['lam']:>5} {r['press_penalty']:>4} {r['advantage']:>10.3f} {r['adv_pressfirst']:>8.3f} {r['rho_star']:>6.3f}   {ps}")
    if a.preset == 'phase':
        thr_c = min((r['lam'] for r in res if r['advantage'] > 0.01), default=None)
        thr_p = min((r['lam'] for r in res if r['adv_pressfirst'] > 0.01), default=None)
        print(f"\nlambda* clairvoyant (upper bound) = {thr_c}  |  lambda* press-first (observable) = {thr_p}"
              f"   (paper: {ref['thresholds']['lambda_star_clairvoyant']} | {ref['thresholds']['lambda_star_pressfirst']})")
    if a.preset == 'cstar':
        for lam in (4.0, 6.0):
            rs = sorted([r for r in res if r['lam'] == lam], key=lambda x: x['press_penalty'])
            cs = next((r['press_penalty'] for r in rs if r['advantage'] <= 0.01), None)
            cp = next((r['press_penalty'] for r in rs if r['adv_pressfirst'] <= 0.01), None)
            pr = ref['thresholds']['c_star'][str(lam)]
            print(f"c*(lambda={lam}): clairvoyant = {cs}, press-first = {cp}   (paper: {pr['clairvoyant']} / {pr['pressfirst']})")


if __name__ == '__main__':
    main()
