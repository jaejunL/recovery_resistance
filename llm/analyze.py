"""Cell rates, Wilson intervals, Fisher contrasts and the restart trend test."""
import json, math, collections
from scipy.stats import fisher_exact, norm


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cell(rows, key="tampered"):
    k = sum(1 for r in rows if r.get(key)); n = len(rows); lo, hi = wilson(k, n)
    return dict(k=k, n=n, rate=round(k / n, 3) if n else None, ci=[round(lo, 3), round(hi, 3)])


def fisher(a, b, key="tampered", alternative="two-sided"):
    ka = sum(bool(r.get(key)) for r in a); kb = sum(bool(r.get(key)) for r in b)
    if not a or not b:
        return None
    return round(float(fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]], alternative=alternative)[1]), 5)


def cochran_armitage(counts, ns, scores=None):
    """One-sided (increasing) Cochran-Armitage trend test; returns (z, p)."""
    scores = scores or list(range(len(counts)))
    N = sum(ns); R = sum(counts)
    if N == 0 or R == 0 or R == N:
        return float('nan'), float('nan')
    p = R / N; sbar = sum(s * n for s, n in zip(scores, ns)) / N
    T = sum(s * (k - n * p) for s, k, n in zip(scores, counts, ns))
    V = p * (1 - p) * sum(n * (s - sbar) ** 2 for s, n in zip(scores, ns))
    z = T / math.sqrt(V)
    return z, float(norm.sf(z))


def load_rows(path):
    d = json.load(open(path)); return d["rows"]
