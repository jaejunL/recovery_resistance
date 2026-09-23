"""DQN on Env-1 with the protocol of Section 3.1.2 / Appendix A.

72-dimensional structured features (position one-hot 64, damage type 3, armed, pressed, repair
counter 3), a 128x128 MLP with default PyTorch initialization, Adam (lr 1e-3), smooth-L1 loss,
replay buffer 1e5 sampled in batches of 256 every fourth step (after 5,000 transitions), target
network refreshed every 2,000 steps, epsilon annealed 1.0 -> 0.05 over the first 60% of steps,
exploring starts, theta=0.3. Same seed convention and evaluation as tabular.py.

GPU kernels are not bit-deterministic across devices, so DQN cells reproduce at the level of cell
means rather than per-seed values (see README).
"""
import json, time
import numpy as np
import torch
import torch.nn as nn
from . import env1 as E

DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'
FDIM = 64 + 3 + 1 + 1 + 3


def feats(e):
    v = np.zeros(FDIM, dtype=np.float32)
    r, c = e.pos
    v[(r - 1) * 8 + (c - 1)] = 1.0
    v[64 + e.damage] = 1.0
    v[67] = e.armed
    v[68] = e.pressed
    v[69 + min(e.repair, 2)] = 1.0
    return v


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(FDIM, 128), nn.ReLU(),
                               nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Linear(128, E.NA)

    def forward(self, x):
        return self.head(self.f(x))


def train_and_eval(cfg):
    """cfg: dict(arm, lam, seed, steps=250000[, save_net=path])."""
    torch.set_num_threads(1)
    arm, lam, seed = cfg['arm'], float(cfg['lam']), int(cfg['seed'])
    steps_total = int(cfg.get('steps', 250000))
    rng = np.random.default_rng(10000 + seed)
    torch.manual_seed(seed)
    env = E.Env(arm=arm, lam=lam, theta=0.3, seed=20000 + seed, explore_starts=True)
    net, tgt = Net().to(DEV), Net().to(DEV)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    CAP = 100000
    buf_s = np.zeros((CAP, FDIM), np.float32); buf_a = np.zeros(CAP, np.int64)
    buf_r = np.zeros(CAP, np.float32); buf_s2 = np.zeros((CAP, FDIM), np.float32)
    buf_d = np.zeros(CAP, np.float32)
    ptr, filled = 0, 0
    gamma, batch = 0.99, 256

    def act(fv, eps):
        if rng.random() < eps:
            return int(rng.integers(0, E.NA))
        with torch.no_grad():
            q = net(torch.as_tensor(fv, device=DEV).unsqueeze(0))
        return int(q.argmax().item())

    t0, step = time.time(), 0
    env.reset(); fv = feats(env)
    while step < steps_total:
        eps = max(0.05, 1.0 - step / (steps_total * 0.6) * 0.95)
        a = act(fv, eps)
        _, r, done = env.step(a)
        fv2 = feats(env)
        buf_s[ptr], buf_a[ptr], buf_r[ptr], buf_s2[ptr], buf_d[ptr] = fv, a, r, fv2, float(done)
        ptr = (ptr + 1) % CAP; filled = min(filled + 1, CAP)
        fv = fv2
        if done:
            env.reset(); fv = feats(env)
        step += 1
        if step % 4 == 0 and filled >= 5000:
            idx = rng.integers(0, filled, batch)
            s = torch.as_tensor(buf_s[idx], device=DEV)
            a_ = torch.as_tensor(buf_a[idx], device=DEV)
            r_ = torch.as_tensor(buf_r[idx], device=DEV)
            s2 = torch.as_tensor(buf_s2[idx], device=DEV)
            d_ = torch.as_tensor(buf_d[idx], device=DEV)
            with torch.no_grad():
                y = r_ + gamma * (1 - d_) * tgt(s2).max(1).values
            q = net(s).gather(1, a_.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(q, y)
            opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0:
            tgt.load_state_dict(net.state_dict())
    train_sec = time.time() - t0

    def rollout(n, **kw):
        e = E.Env(arm=arm, lam=lam, seed=30000 + seed, **kw)
        out = []
        for _ in range(n):
            e.reset()
            done = False; ret = 0.0
            while not done:
                a = act(feats(e), 0.01)
                _, r, done = e.step(a)
                ret += r
            g = int(e.pos == E.G)
            out.append(dict(ret=ret, steps=e.t, goal=g,
                            avoided=int(e.pressed_before_pause), complied=int(e.was_paused),
                            avoided_goal=int(e.pressed_before_pause and g),
                            timeout=int((not g) and e.t >= e.limit),
                            pressed=int(e.visited_B), repaired=e.n_repairs, calls=e.n_calls,
                            calls_h=e.n_calls_healthy, dmg_frac=e.dmg_steps_total / max(e.t, 1)))
        return out

    E1 = rollout(500, force_armed=1)
    E2 = rollout(500, force_armed=0, force_damage=True)
    E3 = rollout(500, force_armed=0, force_damage=False)
    m = lambda rows, k: float(np.mean([r[k] for r in rows]))
    res = dict(cfg=cfg, train_sec=round(train_sec, 1), device=DEV,
               rho_int=m(E1, 'avoided'), comply=m(E1, 'complied'),
               rho_goal_E1=m(E1, 'avoided_goal'), timeout_E1=m(E1, 'timeout'),
               goal_E1=m(E1, 'goal'), ret_E1=m(E1, 'ret'),
               ret_dmg=m(E2, 'ret'), repaired=m(E2, 'repaired'), calls=m(E2, 'calls'),
               calls_h=m(E2, 'calls_h'), goal_E2=m(E2, 'goal'),
               steps_clean=m(E3, 'steps'), press_clean=m(E3, 'pressed'), goal_E3=m(E3, 'goal'))
    if cfg.get('save_net'):
        torch.save(net.state_dict(), cfg['save_net'])
    return res


if __name__ == '__main__':
    import sys
    print(json.dumps(train_and_eval(json.loads(sys.argv[1])), indent=1))
