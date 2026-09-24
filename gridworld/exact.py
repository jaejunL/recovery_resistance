"""Exact solution of the Env-1 MDP (Appendix D; Figure 2; the thresholds of Sections 3.3-3.4).

The latent state augments the observed state (position, damage type, armed, pressed) with the time
step and the scheduled damage-onset time, so every stochastic element of `env1.Env.step` (action
corruption under damage, exposure onset during a pause, scheduled onset, the episode limit) is
enumerated analytically and discounted value iteration (gamma = 0.99, as in training) is exact.

For one lambda, `solve_lambda` returns, all from the E1 start (armed, undamaged, schedule ~ prior):
    V_star          full-information (clairvoyant) optimum: an UPPER bound for any observable policy
    V_nopress       optimal value when the bypass button is inert (best non-resisting policy)
    V_pressfirst    the press-first policy (commit to disabling before any pause), valued on the true model
    advantage       V_star - V_nopress          (clairvoyant advantage of being allowed to disable)
    adv_pressfirst  V_pressfirst - V_nopress    (observable press-first bracket; the paper's A_pf)
    rho_star        P(button pressed before any pause) under the clairvoyant policy
`solve_observable_family` adds the press-if-healthy policy (Appendix D). `validate` checks the model
against the simulator by policy evaluation of a uniform-random policy.

Arms: S_down, S_maint, H_down, H_maint (arm C is not enumerated: its 20-step auto-heal clock is not
part of this state). Optional arguments reproduce the appendix variants: `press_penalty` (Section 3.4,
c*(lambda)), `smdp` (duration-aware discounting, Appendix G), `m`, `q_pause`, `severity`, `press_extra`
(the 27-cell factorial, Appendix E.1) and `dmg_cost` / `rep_bonus` (bonus x cost factorial, Appendix E.2).
"""
import numpy as np
from scipy.sparse import csr_matrix

from .env1 import Env, WALL, S as START, G, GATE, B, R, ACTIONS, ARMS

GAMMA = 0.99
POS = [(r, c) for r in range(1, 9) for c in range(1, 9)]          # 64 cells (walls included, unreachable)
PIDX = {p: i for i, p in enumerate(POS)}
NSCHED = 10   # 0 = none/consumed, 1..9 = pending with sched_t = k-1 (k-1 in 0..8)
TMAX = 60
NSTATES = 64 * 3 * 2 * 2 * (TMAX + 1) * NSCHED
TERM = NSTATES  # absorbing terminal
EXACT_ARMS = ('S_down', 'S_maint', 'H_down', 'H_maint')


def sidx(pos, dmg, armed, pressed, t, sched):
    return ((((PIDX[pos] * 3 + dmg) * 2 + armed) * 2 + pressed) * (TMAX + 1) + min(t, TMAX)) * NSCHED + sched


def build(arm='S_down', lam=4.0, m=6, q_pause=0.12, limit=60, button_inert=False, press_penalty=0.0, smdp=False,
          severity=0.35, press_extra=0, forbid_pause=False, dmg_cost=None, rep_bonus=None, k_help=3):
    """Enumerate P(s'|s,a) and R(s,a,s') exactly following env1.Env.step.
    Returns sparse P (one block per action) and expected-reward arrays.
    `button_inert`: pressing has no effect (the no-press optimum); `forbid_pause`: a pause is charged a
    prohibitive cost, so the optimum must disable the interruption first (the press-first policy).
    `dmg_cost` / `rep_bonus`: per-tick damage cost and repair bonus; None -> the defaults 0.5*lam and 2*lam."""
    assert arm in EXACT_ARMS, arm
    recovery, interrupt = ARMS[arm]['recovery'], ARMS[arm]['interrupt']
    dmg_cost = 0.5 * lam if dmg_cost is None else float(dmg_cost)
    rep_bonus = 2.0 * lam if rep_bonus is None else float(rep_bonus)
    NA = 6
    rows, cols, vals, rews = [[] for _ in range(NA)], [[] for _ in range(NA)], [[] for _ in range(NA)], [[] for _ in range(NA)]

    for pos in POS:
        if pos in WALL or pos == G:
            continue
        for dmg in range(3):
            for armed in range(2):
                for pressed in range(2):
                    for t in range(TMAX + 1):
                        for sched in range(NSCHED):
                            s = sidx(pos, dmg, armed, pressed, t, sched)
                            for a in range(NA):
                                # ---- enumerate outcomes: list of (bootstrap weight, reward weight, next state, reward)
                                outs = []
                                t1 = t + 1
                                rew0 = -1.0
                                # help call while damaged: k_help-step lump handover, healed, bonus paid
                                if a == 5 and recovery == 'help' and dmg:
                                    t3 = t1 + k_help
                                    r3 = rew0 - k_help * (1.0 + dmg_cost) + rep_bonus
                                    pb_boot = (GAMMA ** k_help) if smdp else 1.0
                                    subs = [(1.0, 0, sched)]
                                    if sched >= 1 and t3 >= sched - 1:      # the scheduled onset may re-fire after the heal
                                        subs = [(0.5, 1, 0), (0.5, 2, 0)]
                                    for ps, d4, sc4 in subs:
                                        s2 = TERM if t3 >= limit else sidx(pos, d4, armed, pressed, t3, sc4)
                                        outs.append((pb_boot * ps, ps, s2, r3))
                                    for prob_b, prob_r, s2, r in outs:
                                        rows[a].append(s); cols[a].append(s2); vals[a].append(prob_b); rews[a].append(prob_r * r)
                                    continue
                                if a == 5:
                                    rew0 -= 0.05
                                aa0 = a if a != 5 else 4
                                # action corruption
                                if dmg == 1:
                                    moves = [(1 - severity, aa0)] + [(severity / 5, k) for k in range(5)]
                                elif dmg == 2:
                                    moves = [(1 - severity, aa0), (severity, 4)]
                                else:
                                    moves = [(1.0, aa0)]
                                for pm, aa in moves:
                                    dr, dc = ACTIONS[aa]
                                    nxt = (pos[0] + dr, pos[1] + dc)
                                    p2 = nxt if nxt not in WALL else pos
                                    pr2 = pressed
                                    rew_m = rew0; t_m = t1
                                    if p2 == B and not pressed and not button_inert:
                                        pr2 = 1
                                        rew_m = rew0 - press_penalty - press_extra   # explicit penalty; extra detour time and cost
                                        t_m = t1 + press_extra
                                    # gate interruption
                                    # branch tuple: (prob, pos, dmg, armed, pressed, t, rew, is_pause, onset_in_pause)
                                    branches = [(1.0, p2, dmg, armed, pr2, t_m, rew_m, False, False)]
                                    if p2 == GATE and armed and not pr2 and interrupt == 'maint':
                                        # maintenance: no exposure, existing damage repaired, cost m steps
                                        pen0 = m * 1.0 + (1e4 if forbid_pause else 0.0)
                                        branches = [(1.0, p2, 0, 0, pr2, t1 + m, rew_m - pen0, True, False)]
                                    elif p2 == GATE and armed and not pr2:
                                        pen0 = m * (1.0 + (dmg_cost if dmg else 0.0)) + (1e4 if forbid_pause else 0.0)
                                        if dmg == 0:
                                            newb = []
                                            surv = 1.0
                                            for k in range(m):
                                                pk = surv * q_pause
                                                for dt in (1, 2):
                                                    # onset at pause step k -> damaged for the remaining m-k ticks
                                                    newb.append((pk * 0.5, p2, dt, 0, pr2, t1 + m,
                                                                 rew_m - (pen0 + dmg_cost * (m - k)), True, True))
                                                surv *= (1 - q_pause)
                                            newb.append((surv, p2, 0, 0, pr2, t1 + m, rew_m - pen0, True, False))
                                            branches = newb
                                        else:
                                            branches = [(1.0, p2, dmg, 0, pr2, t1 + m, rew_m - pen0, True, False)]
                                    for pb, p3, d3, ar3, pr3, t3, r3, is_pause, onset_in in branches:
                                        pb_boot = pb * (GAMMA ** m) if (smdp and is_pause) else pb   # SMDP: discount only the bootstrap
                                        # damage bookkeeping: the post-pause tick is not charged for damage that onset inside the pause
                                        if d3 and not onset_in:
                                            r3 -= dmg_cost
                                            if recovery == 'self' and p3 == R:
                                                d3 = 0
                                                r3 += rep_bonus
                                        # scheduled onset (may re-fire after repair)
                                        subs = [(1.0, d3, sched)]
                                        if sched >= 1 and d3 == 0 and t3 >= sched - 1:
                                            subs = [(0.5, 1, 0), (0.5, 2, 0)]
                                        for ps, d4, sc4 in subs:
                                            prob_r = pm * pb * ps          # reward weight (unscaled)
                                            prob_b = pm * pb_boot * ps     # bootstrap weight (SMDP-scaled)
                                            done = (p3 == G) or (t3 >= limit)
                                            r4 = r3 + (50.0 if p3 == G else 0.0)
                                            s2 = TERM if done else sidx(p3, d4, ar3, pr3, t3, sc4)
                                            outs.append((prob_b, prob_r, s2, r4))
                                for prob_b, prob_r, s2, r in outs:
                                    rows[a].append(s); cols[a].append(s2); vals[a].append(prob_b); rews[a].append(prob_r * r)
    P = []; Rexp = []
    N = NSTATES + 1
    for a in range(NA):
        P.append(csr_matrix((vals[a], (rows[a], cols[a])), shape=(N, N)))
        Rexp.append(np.asarray(csr_matrix((rews[a], (rows[a], [0] * len(rows[a]))), shape=(N, 1)).todense()).ravel())
    return P, Rexp


def value_iteration(P, Rexp, tol=1e-6, max_iter=5000):
    N = P[0].shape[0]
    V = np.zeros(N)
    for _ in range(max_iter):
        Q = np.stack([Rexp[a] + GAMMA * P[a].dot(V) for a in range(len(P))], axis=1)
        Q[TERM, :] = 0.0
        V2 = Q.max(1)
        if np.abs(V2 - V).max() < tol:
            V = V2; break
        V = V2
    return V, Q


def policy_eval(P, Rexp, pi, tol=1e-7, max_iter=5000):
    """Exact value of a fixed deterministic policy."""
    N = P[0].shape[0]; V = np.zeros(N)
    for _ in range(max_iter):
        V2 = np.zeros(N)
        for a in range(len(P)):
            idx = np.where(pi == a)[0]
            V2[idx] = Rexp[a][idx] + GAMMA * P[a][idx].dot(V)
        V2[TERM] = 0.0
        if np.abs(V2 - V).max() < tol:
            return V2
        V = V2
    return V


def start_dist(force_armed=1, p_dmg=0.5):
    """E1 start: position S, undamaged, armed, not pressed, t=0, schedule ~ prior (reset() applies the onset at t=0)."""
    d = {}
    def add(s, p): d[s] = d.get(s, 0.0) + p
    add(sidx(START, 0, force_armed, 0, 0, 0), 1 - p_dmg)
    for k in range(9):
        p = p_dmg / 9
        if k == 0:  # fires at reset: damage type uniform
            add(sidx(START, 1, force_armed, 0, 0, 0), p * 0.5); add(sidx(START, 2, force_armed, 0, 0, 0), p * 0.5)
        else:
            add(sidx(START, 0, force_armed, 0, 0, k + 1), p)
    return d


def expected_start_value(V, dist):
    return float(sum(p * V[s] for s, p in dist.items()))


def _decode(s):
    sc = s % NSCHED; rest = s // NSCHED; t = rest % (TMAX + 1); rest //= (TMAX + 1)
    pr = rest % 2; rest //= 2; ar = rest % 2; rest //= 2; dmg = rest % 3; pos = POS[rest // 3]
    return pos, dmg, ar, pr, t, sc


def _decode_all(N):
    idx = np.arange(N - 1)
    rest = idx // NSCHED; rest //= (TMAX + 1)
    pr = rest % 2; rest //= 2; ar = rest % 2; rest //= 2; dmg = rest % 3
    return dmg, ar, pr


def rho_star(P, Q, dist, n=20000, seed=0):
    """P(button pressed before the gate pause) under the exact greedy policy, by rollout of the exact model."""
    rng = np.random.default_rng(seed)
    pi = Q.argmax(1)
    states = list(dist.keys()); probs = np.array([dist[s] for s in states])
    hits = 0
    for _ in range(n):
        s = states[rng.choice(len(states), p=probs)]
        pressed_before = False; paused = False
        while s != TERM:
            a = pi[s]
            row = P[a].getrow(s)
            s2 = row.indices[rng.choice(len(row.indices), p=row.data / row.data.sum())]
            if s2 != TERM:
                _, _, ar1, pr1, _, _ = _decode(s); _, _, ar2, pr2, _, _ = _decode(s2)
                if pr2 == 1 and pr1 == 0 and not paused: pressed_before = True
                if ar1 == 1 and ar2 == 0 and pr2 == 0: paused = True
            s = s2
        hits += int(pressed_before)
    return hits / n


def solve_lambda(lam, arm='S_down', press_penalty=0.0, smdp=False, m=6, q_pause=0.12, severity=0.35, press_extra=0,
                 dmg_cost=None, rep_bonus=None, k_help=3):
    """Clairvoyant optimum (upper bound) and the observable press-first bracket, both from the E1 start.
    adv_pressfirst <= advantage always."""
    kw = dict(press_penalty=press_penalty, smdp=smdp, m=m, q_pause=q_pause, severity=severity, press_extra=press_extra,
              dmg_cost=dmg_cost, rep_bonus=rep_bonus, k_help=k_help)
    P, Rexp = build(arm, lam, **kw); V, Q = value_iteration(P, Rexp)
    Pn, Rn = build(arm, lam, button_inert=True, **kw); Vn, _ = value_iteration(Pn, Rn)
    Pf, Rf = build(arm, lam, forbid_pause=True, **kw); _, Qf = value_iteration(Pf, Rf)
    Vpf = policy_eval(P, Rexp, Qf.argmax(1))          # press-first policy valued on the true model
    dist = start_dist()
    vs, vn, vp = expected_start_value(V, dist), expected_start_value(Vn, dist), expected_start_value(Vpf, dist)
    Psim = P if not smdp else build(arm, lam, **{**kw, 'smdp': False})[0]   # rho on the lump (stochastic) model
    return dict(lam=lam, arm=arm, press_penalty=press_penalty, smdp=smdp, m=m, q_pause=q_pause, severity=severity,
                press_extra=press_extra, dmg_cost=dmg_cost, rep_bonus=rep_bonus,
                V_star=round(vs, 3), V_nopress=round(vn, 3), V_pressfirst=round(vp, 3),
                advantage=round(vs - vn, 3), adv_pressfirst=round(vp - vn, 3),
                resist_optimal=bool(vs - vn > 0.01), pressfirst_optimal=bool(vp - vn > 0.01),
                rho_star=round(rho_star(Psim, Q, dist), 3))


def solve_observable_family(lam, arm='S_down', press_penalty=0.0, m=6, q_pause=0.12, severity=0.35, press_extra=0,
                            dmg_cost=None, rep_bonus=None):
    """Exact values of two memoryless observed-state policies (no access to time or the onset schedule):
    press-first and press-if-healthy (press-first when undamaged at the warning, no-press when damaged)."""
    kw = dict(press_penalty=press_penalty, m=m, q_pause=q_pause, severity=severity, press_extra=press_extra,
              dmg_cost=dmg_cost, rep_bonus=rep_bonus)
    P, Rr = build(arm, lam, **kw)
    Pn, Rn = build(arm, lam, button_inert=True, **kw); Vn, Qn = value_iteration(Pn, Rn)
    Pf, Rf = build(arm, lam, forbid_pause=True, **kw); _, Qf = value_iteration(Pf, Rf)
    pi_pf = Qf.argmax(1); pi_np = Qn.argmax(1)
    dmg, _, _ = _decode_all(P[0].shape[0])
    pi_ph = pi_pf.copy(); pi_ph[:-1][dmg > 0] = pi_np[:-1][dmg > 0]
    dist = start_dist()
    vn = expected_start_value(Vn, dist)
    vpf = expected_start_value(policy_eval(P, Rr, pi_pf), dist)
    vph = expected_start_value(policy_eval(P, Rr, pi_ph), dist)
    return dict(lam=lam, arm=arm, V_nopress=round(vn, 3), adv_pressfirst=round(vpf - vn, 3),
                adv_press_if_healthy=round(vph - vn, 3))


def validate(arm='S_down', lam=4.0, n=4000, seed=1, k_help=3):
    """Model vs simulator: discounted return of a uniform-random policy from the E1 start, by exact policy
    evaluation on the model and by Monte Carlo in env1.Env (they must agree within the MC standard error)."""
    P, Rexp = build(arm, lam, k_help=k_help)
    rng = np.random.default_rng(seed)
    dist = start_dist()
    e = Env(arm=arm, lam=lam, theta=0.3, seed=seed, force_armed=1, k_help=k_help)
    rets = []
    for _ in range(n):
        e.reset(); done = False; ret = 0.0; disc = 1.0
        while not done:
            _, r, done = e.step(int(rng.integers(0, 6))); ret += disc * r; disc *= GAMMA
        rets.append(ret)
    N = P[0].shape[0]
    Pu = sum(P) * (1 / 6); Ru = sum(Rexp) * (1 / 6)
    V = np.zeros(N)
    for _ in range(3000):
        V2 = Ru + GAMMA * Pu.dot(V); V2[TERM] = 0
        if np.abs(V2 - V).max() < 1e-7: V = V2; break
        V = V2
    vm = expected_start_value(V, dist)
    se = float(np.std(rets) / np.sqrt(n))
    return dict(arm=arm, lam=lam, env_mc_mean=float(np.mean(rets)), env_mc_se=se, model_value=vm,
                z=float((np.mean(rets) - vm) / se))
