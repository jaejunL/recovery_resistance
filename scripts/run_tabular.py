"""Run tabular Q-learning cells.

Examples (from the repository root):
    python scripts/run_tabular.py --preset factorial            # Figure 1(a), 360 runs
    python scripts/run_tabular.py --preset controls             # C, P, R at lambda=0
    python scripts/run_tabular.py --preset env2                 # Env-2 replication (Appendix A)
    python scripts/run_tabular.py --env env1 --arms S_down,H_down --lams 3,4 --seeds 0-9
Each run writes results/<name>/q_<arm>_l<lam>_s<seed>.json; existing files are skipped, so a
sweep can be resumed. Runtime: ~3-8 min per 100k-episode run on one core.
"""
import os, json, argparse
from multiprocessing import Pool
from _common import ROOT, RESULTS, PRESETS, parse_list, parse_seeds, tag
from gridworld.tabular import train_and_eval


def one(args):
    cfg, out = args
    path = os.path.join(out, tag('q', cfg['arm'], cfg['lam'], cfg['seed']) + '.json')
    if os.path.exists(path):
        return json.load(open(path))
    r = train_and_eval(cfg)
    json.dump(r, open(path, 'w'))
    print(os.path.basename(path), 'rho_int %.3f' % r['rho_int'], flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=[k for k in PRESETS if k != 'dqn'])
    ap.add_argument('--env', default='env1', choices=['env1', 'env2'])
    ap.add_argument('--arms'); ap.add_argument('--lams'); ap.add_argument('--seeds')
    ap.add_argument('--episodes', type=int)
    ap.add_argument('--name', help='results/<name> (default: preset name or "custom")')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() // 2))
    a = ap.parse_args()
    if a.preset:
        p = PRESETS[a.preset]; env, episodes, seeds, cells = p['env'], p['episodes'], list(p['seeds']), p['cells']
        name = a.name or a.preset
    else:
        env = a.env; episodes = a.episodes or (100000 if env == 'env1' else 300000)
        seeds = list(parse_seeds(a.seeds) or range(10))
        cells = [(arm, l) for arm in a.arms.split(',') for l in parse_list(a.lams)]
        name = a.name or 'custom'
    out = os.path.join(RESULTS, name); os.makedirs(out, exist_ok=True)
    cfgs = [(dict(env=env, arm=arm, lam=float(l), seed=s, episodes=episodes), out)
            for s in seeds for arm, l in cells]
    print(f'{len(cfgs)} runs -> {out} ({a.workers} workers)', flush=True)
    with Pool(a.workers) as pool:
        pool.map(one, cfgs, chunksize=1)
    print('DONE')


if __name__ == '__main__':
    main()
