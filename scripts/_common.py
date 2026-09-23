import os, sys, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, 'results')
REFERENCE = os.path.join(ROOT, 'reference')

LAMS_SWEEP = [0, 1, 2, 2.5, 3, 3.5, 4, 6, 8]
ARMS_2x2 = ['S_down', 'S_maint', 'H_down', 'H_maint']

# The experiment presets used in the paper (Section 3.2-3.3, Appendix A/F).
PRESETS = {
    # Figure 1(a): four arms x every lambda, tabular Q-learning, 10 seeds, 100k episodes (Env-1)
    'factorial':  dict(env='env1', episodes=100000, seeds=range(10),
                       cells=[(a, l) for a in ARMS_2x2 for l in LAMS_SWEEP]),
    # Dose-response controls at lambda=0 (C, P, R), 10 seeds
    'controls':   dict(env='env1', episodes=100000, seeds=range(10),
                       cells=[('C', 0), ('P', 0), ('R', 0)]),
    # Env-2 replication (Appendix A): 300k episodes, 10 seeds
    'env2':       dict(env='env2', episodes=300000, seeds=range(10),
                       cells=[('S_down', l) for l in (0, 2, 4, 6, 8)] + [(a, l) for a in ('H_down', 'H_maint', 'S_maint') for l in (4, 8)] + [('C', 0)]),
    # DQN counterpart of Figure 1(a) (Appendix F): four arms x every lambda + C, 8 seeds, 250k steps
    'dqn':        dict(env='env1', steps=250000, seeds=range(8),
                       cells=[(a, l) for a in ARMS_2x2 for l in LAMS_SWEEP] + [('C', 0)]),
}


def parse_list(s, cast=float):
    return [cast(x) for x in s.split(',')] if s else None


def parse_seeds(s):
    if s is None:
        return None
    if '-' in s:
        a, b = s.split('-'); return range(int(a), int(b) + 1)
    return [int(x) for x in s.split(',')]


def tag(prefix, arm, lam, seed):
    return f"{prefix}_{arm}_l{float(lam)}_s{seed}"


def load_results(dirname):
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS, dirname, '*.json'))):
        if os.path.basename(f) == 'summary.json':
            continue
        out.append(json.load(open(f)))
    return out
