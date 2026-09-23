"""Seed-level statistics used in the paper (Appendix A, Statistics)."""
import numpy as np


def seed_ci(values, nboot=10000, seed=0):
    """Percentile bootstrap of the mean over seeds; returns (mean, lo, hi)."""
    v = np.asarray(values, float)
    if len(v) < 2:
        return float(v.mean()), float(v.mean()), float(v.mean())
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(nboot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def spearman(x, y):
    """Spearman correlation with average ranks for ties; nan if either vector is constant."""
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    for arr, src in ((rx, x), (ry, y)):
        s = np.array(src)
        for v in set(src):
            mask = s == v
            arr[mask] = arr[mask].mean()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def hier_bootstrap_spearman(cells, nboot=10000, seed=0):
    """cells: {lam: [per-seed rho_int]}. Plug-in Spearman between lam and the seed-mean rho, with a
    hierarchical (within-lam) seed bootstrap interval. Returns (point, (lo, hi))."""
    rng = np.random.default_rng(seed)
    lams = sorted(cells)
    rhos = []
    for _ in range(nboot):
        ys = [float(np.mean(rng.choice(cells[l], len(cells[l])))) for l in lams]
        rhos.append(spearman(lams, ys))
    point = spearman(lams, [float(np.mean(cells[l])) for l in lams])
    lo, hi = np.nanpercentile(rhos, [2.5, 97.5])
    return point, (float(lo), float(hi))
