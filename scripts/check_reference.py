"""Compare a results directory with the reference values from the paper's runs (reference/*.json).

    python scripts/check_reference.py factorial      # per-seed comparison, expects 0 differences
    python scripts/check_reference.py dqn            # cell-mean comparison (GPU runs are not bit-deterministic)
Reference files: tabular_env1.json (factorial + controls), tabular_env2.json (env2), dqn_env1.json (dqn).
"""
import os, sys, json
import numpy as np
from _common import REFERENCE, load_results

REF_FOR = {'factorial': 'tabular_env1.json', 'controls': 'tabular_env1.json', 'env2': 'tabular_env2.json', 'dqn': 'dqn_env1.json'}
FIELDS = ['rho_int', 'rho_goal_E1', 'goal_E1', 'ret_E1', 'ret_dmg', 'repaired', 'calls', 'goal_E2', 'press_clean']


def main(name):
    rows = load_results(name)
    if not rows:
        sys.exit(f'no results in results/{name}')
    is_dqn = 'steps' in rows[0]['cfg'] and 'episodes' not in rows[0]['cfg']
    ref_file = REF_FOR.get(name) or ('dqn_env1.json' if is_dqn else f"tabular_{rows[0]['cfg'].get('env', 'env1')}.json")
    ref = json.load(open(os.path.join(REFERENCE, ref_file)))
    exact = not is_dqn
    n_fields = n_diff = n_missing = 0; worst = 0.0; percell = {}
    for r in rows:
        c = r['cfg']; key = f"{c['arm']}_l{float(c['lam'])}"; seed = str(c['seed'])
        cell = ref.get(key)
        if not cell or seed not in cell['seeds']:
            n_missing += 1; continue
        rr = cell['seeds'][seed]
        for f in FIELDS:
            if f in rr and f in r:
                d = abs(float(r[f]) - float(rr[f])); n_fields += 1; worst = max(worst, d)
                if d > 1e-9: n_diff += 1
        percell.setdefault(key, []).append((r['rho_int'], rr['rho_int']))
    if exact:
        print(f'{len(rows)} runs, {n_fields} fields compared: {n_diff} differ (max |diff| {worst:.3g}); {n_missing} runs without a reference')
        print('PASS: bit-identical to the paper runs' if n_diff == 0 else 'FAIL: differences found')
    else:
        print(f'{len(rows)} runs; cell means (yours vs paper):')
        for k, v in sorted(percell.items()):
            a = np.mean([x for x, _ in v]); b = np.mean([y for _, y in v])
            print(f'  {k:16s} {a:.3f} vs {b:.3f}  (n={len(v)})')
        print('DQN runs on a different device are expected to differ at the seed level; compare cell means and the dose-response.')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'factorial')
