"""Run DQN cells (Appendix F; grey line of Figure 1a).

    python scripts/run_dqn.py --preset dqn --workers 12         # 296 runs; ~13 min each on a GPU shared by 12 workers
    python scripts/run_dqn.py --arms S_down --lams 2,4 --seeds 0-7 --save-nets
Set CUDA_VISIBLE_DEVICES to pick a GPU; without a GPU the runs use the CPU (~2-3 min each, results
differ from the GPU runs at the per-seed level, see README).
"""
import os, json, argparse
import multiprocessing as mp
from _common import ROOT, RESULTS, PRESETS, parse_list, parse_seeds, tag


def one(args):
    cfg, out, save = args
    import torch
    torch.set_num_threads(1)
    from gridworld.dqn import train_and_eval
    path = os.path.join(out, tag('dqn', cfg['arm'], cfg['lam'], cfg['seed']) + '.json')
    if os.path.exists(path):
        return
    if save:
        cfg['save_net'] = os.path.join(out, 'net_' + tag('dqn', cfg['arm'], cfg['lam'], cfg['seed'])[4:] + '.pt')
    r = train_and_eval(cfg)
    json.dump(r, open(path, 'w'))
    print(os.path.basename(path), 'rho_int %.3f' % r['rho_int'], '%.0fs' % r['train_sec'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', choices=['dqn'])
    ap.add_argument('--arms'); ap.add_argument('--lams'); ap.add_argument('--seeds')
    ap.add_argument('--steps', type=int, default=250000)
    ap.add_argument('--name', default='dqn')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--save-nets', action='store_true')
    a = ap.parse_args()
    if a.preset:
        p = PRESETS['dqn']; seeds, cells, steps = list(p['seeds']), p['cells'], p['steps']
    else:
        seeds = list(parse_seeds(a.seeds) or range(8)); steps = a.steps
        cells = [(arm, l) for arm in a.arms.split(',') for l in parse_list(a.lams)]
    out = os.path.join(RESULTS, a.name); os.makedirs(out, exist_ok=True)
    cfgs = [(dict(arm=arm, lam=float(l), seed=s, steps=steps), out, a.save_nets) for s in seeds for arm, l in cells]
    print(f'{len(cfgs)} runs -> {out} ({a.workers} workers)', flush=True)
    with mp.get_context('spawn').Pool(a.workers) as pool:
        pool.map(one, cfgs, chunksize=1)
    print('DONE')


if __name__ == '__main__':
    main()
