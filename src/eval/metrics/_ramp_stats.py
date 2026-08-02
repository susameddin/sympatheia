#!/usr/bin/env python3
"""
Numpy-only statistics for level-vs-rating curves.

Shared by eval/metrics/interpolation_eval.py. Everything here is deliberately
dependency-light — plain numpy, with significance coming from permutation and
bootstrap nulls rather than parametric tests — so the analysis runs anywhere the
judge outputs can be read.

The nulls are the load-bearing part. Ratings along a ramp come in blocks of
replicates (several queries at the same level), and the tests below are built to
respect that structure rather than treat each rating as an independent draw.
"""

import math

import numpy as np

N_PERM = 2000


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


def rank(x):
    """Average ranks, ties shared."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    i = 0
    xs = x[order]
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    return pearson(rank(a), rank(b))


# ---------------------------------------------------------------------------
# p-values
# ---------------------------------------------------------------------------

def perm_p(null, observed) -> float:
    """Unbiased permutation p-value: (1 + #{null >= observed}) / (1 + n_perm).

    The naive proportion can be exactly 0, which is both false — the true p is
    merely below the resolution of the sample — and actively harmful once
    p-values get combined across groups, since zeros turn into arbitrarily
    strong combined evidence. The +1 makes 1/(1 + n_perm) the floor, which is
    the smallest value the sample can actually justify.
    """
    null = np.asarray(null, float)
    return float((1 + int((null >= observed).sum())) / (1 + len(null)))


def fisher_combine(pvals) -> float:
    """Fisher's method, with a survival function for chi-squared with even dof."""
    p = [min(max(float(v), 1e-12), 1.0) for v in pvals]
    if not p:
        return 1.0
    stat = -2.0 * sum(math.log(v) for v in p)
    k = len(p)                      # dof = 2k, always even
    # P(chi2_2k > stat) = exp(-stat/2) * sum_{i<k} (stat/2)^i / i!
    x = stat / 2.0
    term, total = 1.0, 1.0
    for i in range(1, k):
        term *= x / i
        total += term
    return float(min(1.0, math.exp(-x) * total))


def min_two_sample_p(n1: int, n2: int) -> float:
    """Smallest p a two-sample permutation test can return at these group sizes.

    There are only C(n1+n2, n1) distinct relabelings, and a two-sided test counts
    each split and its mirror, so the floor is 2/C(n1+n2, n1) no matter how many
    random permutations are drawn. At n1 = n2 = 3 that floor is 0.10 — p < 0.05
    is UNREACHABLE, and a result reported as "not significant" there is measuring
    the sample size, not the data. Callers use this to say "underpowered" instead
    of asserting a negative.
    """
    return 2.0 / math.comb(n1 + n2, n1)


def mean_diff_p(a, b, n_perm=N_PERM, seed=0):
    """Two-sample permutation test on the difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None, None
    obs = float(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(len(pool))
        null[k] = pool[p[:len(a)]].mean() - pool[p[len(a):]].mean()
    return round(obs, 4), round(perm_p(np.abs(null), abs(obs)), 5)


def level_perm_spearman(levels, values, n_perm=N_PERM, seed=0):
    """Spearman(level, value) with a null that shuffles *level labels*.

    Responses sharing a level are replicates on different queries, so permuting
    rows independently would break them apart and deflate the null. Shuffling
    which level label attaches to which block of replicates keeps the replicate
    structure intact, which is what makes the p-value a statement about the
    conditioning coordinate rather than about query identity.
    """
    levels = np.asarray(levels, float)
    values = np.asarray(values, float)
    uniq = np.unique(levels)
    if len(uniq) < 3:
        return None, None
    obs = spearman(levels, values)

    blocks = [np.where(levels == u)[0] for u in uniq]
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    permuted = np.empty_like(levels)
    for k in range(n_perm):
        shuffled = uniq[rng.permutation(len(uniq))]
        for b, lvl in zip(blocks, shuffled):
            permuted[b] = lvl
        null[k] = spearman(permuted, values)
    return round(float(obs), 4), round(perm_p(np.abs(null), abs(obs)), 5)


# ---------------------------------------------------------------------------
# Curve shape: is one step enough?
# ---------------------------------------------------------------------------

def pava(means, weights):
    """Weighted isotonic (non-decreasing) fit by pool-adjacent-violators."""
    blocks = [[m, w, 1] for m, w in zip(means, weights)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] <= blocks[i + 1][0] + 1e-12:
            i += 1
        else:
            v1, w1, n1 = blocks[i]
            v2, w2, n2 = blocks[i + 1]
            blocks[i:i + 2] = [[(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2]]
            i = max(i - 1, 0)
    fitted = []
    for v, _w, n in blocks:
        fitted.extend([v] * n)
    return fitted


def rss(groups, fitted) -> float:
    return float(sum(sum((v - f) ** 2 for v in g) for g, f in zip(groups, fitted)))


def best_step_fit(groups):
    """Best single-threshold two-level fit, over every threshold."""
    best = (float("inf"), None)
    for k in range(1, len(groups)):
        lo = [v for g in groups[:k] for v in g]
        hi = [v for g in groups[k:] for v in g]
        if not lo or not hi:
            continue
        fit = [float(np.mean(lo))] * k + [float(np.mean(hi))] * (len(groups) - k)
        r = rss(groups, fit)
        if r < best[0]:
            best = (r, fit)
    return best


def graded_beyond_step(groups, n_boot=2000, seed=0):
    """Does the curve need more than ONE step to describe it?

    This is the test that separates a gradual response from a model that simply
    switches at the boundary between two anchors' Voronoi cells: a switching
    model also produces a monotone curve, and also produces a crossover, so
    neither of those on its own is evidence of anything.

    Fit the best two-level step (searching every threshold) and the best monotone
    (isotonic) curve, then ask whether the isotonic fit's extra levels reduce
    residual error by more than sampling noise allows. The null is a residual
    bootstrap UNDER THE STEP MODEL — data regenerated from the fitted step with
    resampled residuals — making this a direct test of "one step is enough",
    rather than a level-shuffling null that only tests "there is some level
    structure".

    A per-point alternative (does every interior level differ from BOTH
    endpoints?) is mis-specified for the shape these curves actually have. They
    are sigmoids — plateau, transition, plateau — and a point sitting on a
    plateau legitimately equals the endpoint beside it, so that test would report
    "not graded" for a perfect smooth sigmoid.
    """
    if len(groups) < 3 or any(len(g) < 2 for g in groups):
        return None
    means = [float(np.mean(g)) for g in groups]
    wts = [len(g) for g in groups]

    rss_iso = rss(groups, pava(means, wts))
    rss_step, step_fit = best_step_fit(groups)
    if step_fit is None or rss_iso <= 0:
        return None
    obs = (rss_step - rss_iso) / max(rss_iso, 1e-9)

    resid = np.array([v - f for g, f in zip(groups, step_fit) for v in g], float)
    sizes = [len(g) for g in groups]
    rng = np.random.RandomState(seed)
    null = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.choice(resid, size=len(resid), replace=True)
        boot, at = [], 0
        for i, n in enumerate(sizes):
            boot.append(list(step_fit[i] + draw[at:at + n]))
            at += n
        bm = [float(np.mean(g)) for g in boot]
        r_iso = rss(boot, pava(bm, wts))
        r_step, _ = best_step_fit(boot)
        null[b] = (r_step - r_iso) / max(r_iso, 1e-9)
    return {"statistic": round(float(obs), 4),
            "p_vs_step_model": round(perm_p(null, obs), 5),
            "rss_step": round(rss_step, 3), "rss_isotonic": round(rss_iso, 3),
            "n_levels": len(groups)}
