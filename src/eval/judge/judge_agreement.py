#!/usr/bin/env python3
"""
Inter-judge agreement between the Qwen3-Omni judge and the Gemini judge.

If two judges from different model families rank the systems the same way and
agree at the utterance level, the empathy result is not an artifact of
Qwen-family judge bias. This script quantifies both.

For each Gemini output file (judgments_gemini*.jsonl) in the eval dir, it finds
the matching Qwen file (judgments*.jsonl, same suffix), joins ratings on
(id, condition), and reports per-condition sample-level agreement plus each
judge's per-system mean. Writes a new summary file; overwrites nothing else.

The Spearman rank correlation over the *system ranking* is only meaningful once
every system has been judged by both judges; with a partial run the
utterance-level metrics (Pearson + quadratic-weighted kappa) and the per-system
means still apply.

Usage (read-only except for the agreement summary it writes). Run from src/:
    python -m eval.judge.judge_agreement \\
        --eval-dir /path/to/eval_emotional_<experiment>_ckpt<step>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_judgments(path: Path) -> dict:
    """Return {(id, condition): rating} for non-null ratings."""
    out = {}
    with open(path) as f:
        for line in f:
            j = json.loads(line)
            if j.get("rating") is None:
                continue
            out[(j["id"], j["condition"])] = j["rating"]
    return out


def quadratic_weighted_kappa(a, b, k_min=1, k_max=5) -> float:
    """Cohen's quadratic-weighted kappa for integer ratings in [k_min, k_max]."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n_cat = k_max - k_min + 1
    O = np.zeros((n_cat, n_cat))
    for x, y in zip(a, b):
        O[x - k_min, y - k_min] += 1
    W = np.zeros((n_cat, n_cat))
    for i in range(n_cat):
        for j in range(n_cat):
            W[i, j] = ((i - j) ** 2) / ((n_cat - 1) ** 2)
    ha = O.sum(axis=1)
    hb = O.sum(axis=0)
    E = np.outer(ha, hb) / O.sum()
    denom = (W * E).sum()
    if denom == 0:
        return float("nan")
    return float(1 - (W * O).sum() / denom)


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rankdata(x):
    """Average ranks (ties averaged), like scipy.stats.rankdata."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # average tied ranks
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(a, b) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    if len(a) < 2:
        return float("nan")
    return pearson(_rankdata(a), _rankdata(b))


def find_pairs(eval_dir: Path):
    """Yield (condition_suffix, qwen_path, gemini_path) for matching files."""
    for gpath in sorted(eval_dir.glob("judgments_gemini*.jsonl")):
        suffix = gpath.stem[len("judgments_gemini"):]  # "" or "_qwen3omni"
        qpath = eval_dir / f"judgments{suffix}.jsonl"
        if qpath.exists():
            yield suffix, qpath, gpath
        else:
            print(f"  (no matching Qwen file for {gpath.name}; expected {qpath.name})")


def main():
    p = argparse.ArgumentParser(description="Qwen vs Gemini inter-judge agreement")
    p.add_argument("--eval-dir", required=True, nargs="+",
                   help="One or more eval folders containing judgments*.jsonl and judgments_gemini*.jsonl")
    p.add_argument("--out", default=None,
                   help="Agreement summary JSON path. Default: <first eval-dir>/agreement_qwen_vs_gemini.json")
    args = p.parse_args()

    eval_dirs = [Path(d) for d in args.eval_dir]
    report = {"per_condition": {}, "pooled": {}}
    pooled_q, pooled_g = [], []
    per_system = {}   # condition -> {"qwen_mean":, "gemini_mean":, "n":}

    for eval_dir in eval_dirs:
        if not eval_dir.is_dir():
            print(f"ERROR: not a directory: {eval_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"\n=== {eval_dir} ===")
        for suffix, qpath, gpath in find_pairs(eval_dir):
            qmap = load_judgments(qpath)
            gmap = load_judgments(gpath)
            # Join on shared (id, condition) keys.
            shared = sorted(set(qmap) & set(gmap))
            if not shared:
                print(f"  {gpath.name}: no overlapping (id, condition) with {qpath.name}")
                continue
            # Group by condition so per-system means are meaningful.
            by_cond = {}
            for key in shared:
                by_cond.setdefault(key[1], []).append(key)
            for cond, keys in sorted(by_cond.items()):
                qv = [qmap[k] for k in keys]
                gv = [gmap[k] for k in keys]
                pooled_q.extend(qv)
                pooled_g.extend(gv)
                diffs = np.abs(np.array(qv) - np.array(gv))
                stats = {
                    "n": len(keys),
                    "qwen_mean": round(float(np.mean(qv)), 3),
                    "gemini_mean": round(float(np.mean(gv)), 3),
                    "mean_abs_diff": round(float(diffs.mean()), 3),
                    "exact_agree_pct": round(float((diffs == 0).mean() * 100), 1),
                    "within1_pct": round(float((diffs <= 1).mean() * 100), 1),
                    "pearson": round(pearson(qv, gv), 3),
                    "qwqk": round(quadratic_weighted_kappa(qv, gv), 3),
                }
                # Key on (eval dir, condition) in BOTH maps: several --eval-dir
                # arguments can carry the same condition name, and keying the
                # per-system map on the bare condition would silently drop all
                # but the last one from the ranking and the Spearman.
                key = f"{eval_dir.name}:{cond}"
                report["per_condition"][key] = stats
                per_system[key] = {"qwen_mean": stats["qwen_mean"],
                                   "gemini_mean": stats["gemini_mean"], "n": stats["n"]}
                print(f"  [{cond:22s}] n={stats['n']:4d}  "
                      f"Qwen={stats['qwen_mean']:.3f}  Gemini={stats['gemini_mean']:.3f}  "
                      f"r={stats['pearson']:.3f}  qwk={stats['qwqk']:.3f}  "
                      f"exact={stats['exact_agree_pct']:.0f}%  ±1={stats['within1_pct']:.0f}%")

    if pooled_q:
        diffs = np.abs(np.array(pooled_q) - np.array(pooled_g))
        report["pooled"] = {
            "n": len(pooled_q),
            "qwen_mean": round(float(np.mean(pooled_q)), 3),
            "gemini_mean": round(float(np.mean(pooled_g)), 3),
            "mean_abs_diff": round(float(diffs.mean()), 3),
            "exact_agree_pct": round(float((diffs == 0).mean() * 100), 1),
            "within1_pct": round(float((diffs <= 1).mean() * 100), 1),
            "pearson": round(pearson(pooled_q, pooled_g), 3),
            "qwqk": round(quadratic_weighted_kappa(pooled_q, pooled_g), 3),
        }
        print(f"\nPOOLED  n={report['pooled']['n']}  "
              f"Qwen={report['pooled']['qwen_mean']:.3f}  Gemini={report['pooled']['gemini_mean']:.3f}  "
              f"r={report['pooled']['pearson']:.3f}  qwk={report['pooled']['qwqk']:.3f}")

    # Per-system ranking + Spearman rank correlation over the system means.
    if len(per_system) >= 2:
        systems = list(per_system)
        qm = [per_system[c]["qwen_mean"] for c in systems]
        gm = [per_system[c]["gemini_mean"] for c in systems]
        rho = spearman(qm, gm)
        q_rank = sorted(per_system, key=lambda c: per_system[c]["qwen_mean"], reverse=True)
        g_rank = sorted(per_system, key=lambda c: per_system[c]["gemini_mean"], reverse=True)
        report["system_means"] = per_system
        report["system_ranking"] = {"by_qwen": q_rank, "by_gemini": g_rank,
                                     "spearman_rho": round(rho, 3), "n_systems": len(systems)}
        print(f"\nPer-system means:")
        for c in q_rank:
            s = per_system[c]
            print(f"    {c:24s} Qwen={s['qwen_mean']:.3f}  Gemini={s['gemini_mean']:.3f}  n={s['n']}")
        print(f"\nSystem ranking by Qwen  : {q_rank}")
        print(f"System ranking by Gemini: {g_rank}")
        print(f"Spearman rank correlation (Qwen vs Gemini, {len(systems)} systems): rho={rho:.3f}")

    out_path = Path(args.out) if args.out else eval_dirs[0] / "agreement_qwen_vs_gemini.json"
    if out_path.exists():
        print(f"\nNOTE: {out_path} exists — writing to {out_path.with_suffix('.new.json')} instead.")
        out_path = out_path.with_suffix(".new.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved agreement summary: {out_path}")


if __name__ == "__main__":
    main()
