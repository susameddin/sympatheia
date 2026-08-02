#!/usr/bin/env python3
"""
Curve statistics for the two VA interpolation analyses.

Does the response change *gradually* along a path through valence-arousal space,
or does it snap to the nearest of the 12 emotion anchors?

INTERPOLATION (anchor A -> anchor B, parameterised by t).
    Each response is rated twice by judge_interpolation.py's `dualfit` mode: how
    well it suits A, and how well it suits B. Plotting both against t should show
    one curve falling and the other rising, crossing near the middle — one
    emotion disappearing as the other appears.

    The load-bearing statistic is NOT the crossover. A model that switched
    abruptly at the Voronoi boundary would also produce two monotone curves that
    cross. What separates gradual from switching is `graded_beyond_step`: fit the
    best single two-level step and the best monotone curve, and ask whether the
    extra levels beat a residual bootstrap drawn from the step model itself.

INTENSITY (origin -> anchor, parameterised by radius r).
    The PRIMARY measure is analyse_intensity_axis, which uses the `axis` mode's
    SAM-style valence/arousal ratings: scaling an anchor toward the origin
    shrinks both |V| and |A|, so the JUDGED coordinate should contract toward
    neutral as r falls. Neutral is not assumed — it is the mean judged position
    of the r=0 responses, which are literally generated at the origin. Each axis
    is standardised before the radius is taken, so the two rating scales cannot
    contribute unequally, and only the rank correlation with r is reported, so no
    mapping between judged units and anchor units is ever needed.

    The `intensity` mode's dedicated degree rubric is reported as a SECONDARY
    measure, alongside an opposite-emotion control: a ray on which both the
    target and the control rise is measuring expressiveness, not the emotion.

All statistics are numpy-only with permutation and bootstrap nulls; see
_ramp_stats.py.

Usage (run from src/):
    python -m eval.metrics.interpolation_eval \\
        --work-dir /path/to/interp/intensity --design intensity --judge qwen

    python -m eval.metrics.interpolation_eval \\
        --work-dir /path/to/interp/interpolation --design interpolation \\
        --levels 0,0.25,0.5,0.75,1.0

Outputs, under <work-dir>/results/:
    ramps_{judge}[_{rubric}].json          per-path / per-ray statistics
    ramps_pooled_{judge}[_{rubric}][_sep{X}].json   the averaged curve
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# PROJECT_ROOT is the src/ dir; its parent is the repo root, which is what
# `from src.constants ...` resolves against. Both go on sys.path so the module
# imports regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from src.constants import EMOTION_VA_MAPPING
from eval.metrics._ramp_stats import (fisher_combine, graded_beyond_step,
                                      level_perm_spearman, mean_diff_p,
                                      min_two_sample_p)

DEFAULT_RUBRIC = "custom"

# The judge mode each design reads its ratings from.
DESIGN_MODES = {
    "intensity": "intensity",
    "interpolation": "dualfit",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ratings(work_dir: Path, mode: str, judge_tag: str = "qwen",
                 rubric: str = DEFAULT_RUBRIC) -> list:
    """Every judgment for one (mode, judge, rubric), merged across shards.

    A staged run writes group-suffixed files, so the glob has to be permissive
    enough to pick all of them up; deduplicating on (id, probe) is what makes
    that safe.

    But it must NOT be permissive across RUBRICS. ``single_intensity_qwen*`` also
    matches ``single_intensity_qwen_rubric-published.*``, and the two carry the
    same (id, probe) keys on different scales (1-7 custom vs 1-5 published).
    Merging them yields a curve that is 1-7 ratings for some queries and 1-5 for
    others — a mixture of two scales reported as one number. Filtering on the
    filename marker is what keeps them apart.
    """
    res = work_dir / "results"
    shards = sorted(res.glob(f"single_{mode}_{judge_tag}*.jsonl"))
    if rubric == "published":
        shards = [s for s in shards if "_rubric-published" in s.name]
    else:
        shards = [s for s in shards if "_rubric-" not in s.name]
    if not shards:
        raise SystemExit(
            f"no {mode}/{judge_tag}/{rubric} judgments under {res}\n"
            f"  run: python -m eval.judge.judge_interpolation "
            f"--work-dir {work_dir} --mode {mode} --backend {judge_tag}")

    rows, seen = [], set()
    for s in shards:
        for line in open(s):
            r = json.loads(line)
            k = (r["id"], r["probe"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
    return [r for r in rows if r.get("rating") is not None]


def _separation(group: str) -> float:
    """VA distance between the two anchors named in an 'A-B' group label."""
    a, _, b = group.partition("-")
    if a in EMOTION_VA_MAPPING and b in EMOTION_VA_MAPPING:
        return math.dist(EMOTION_VA_MAPPING[a], EMOTION_VA_MAPPING[b])
    return float("inf")


# ---------------------------------------------------------------------------
# Inter-emotion interpolation
# ---------------------------------------------------------------------------

def analyse_interpolation(work_dir: Path, judge_tag: str = "qwen",
                          rubric: str = DEFAULT_RUBRIC) -> dict:
    rows = load_ratings(work_dir, "dualfit", judge_tag, rubric)

    per_resp = defaultdict(dict)
    info = {}
    for r in rows:
        m = r.get("meta", {})
        if not m.get("endpoint_a"):
            continue
        path = f"{m['endpoint_a']}-{m['endpoint_b']}"
        per_resp[(path, r["id"])][r["probe"]] = r["rating"]
        info[(path, r["id"])] = (path, float(m["t"]),
                                 m["endpoint_a"], m["endpoint_b"])

    # One variant per (endpoint_a label, endpoint_b label) actually judged. A
    # --relabel run adds e.g. "endpoint_a:Afraid" alongside "endpoint_a:Anxious"
    # on the identical audio; keying on the label keeps them as separate paths
    # instead of letting whichever loaded last silently overwrite the other.
    by_path = defaultdict(list)
    for key, d in per_resp.items():
        labs_a = [k for k in d if k.startswith("endpoint_a:")]
        labs_b = [k for k in d if k.startswith("endpoint_b:")]
        path, t, a, b = info[key]
        for ka in labs_a:
            for kb in labs_b:
                la, lb = ka.split(":", 1)[1], kb.split(":", 1)[1]
                by_path[f"{la}-{lb}"].append({
                    "t": t, "fit_a": d[ka], "fit_b": d[kb],
                    "diff": d[kb] - d[ka],
                    "endpoint_a": la, "endpoint_b": lb})

    out = {}
    for path, rows_p in sorted(by_path.items()):
        ts = [r["t"] for r in rows_p]
        res = {
            "n_responses": len(rows_p),
            "endpoint_a": rows_p[0]["endpoint_a"],
            "endpoint_b": rows_p[0]["endpoint_b"],
        }
        for field in ("fit_a", "fit_b", "diff"):
            rho, p = level_perm_spearman(ts, [r[field] for r in rows_p])
            res[f"spearman_t_{field}"] = rho
            res[f"p_{field}"] = p

        by_t = defaultdict(list)
        for r in rows_p:
            by_t[round(r["t"], 3)].append(r)
        curve = {}
        for t in sorted(by_t):
            g = by_t[t]
            curve[f"{t:.2f}"] = {
                "n": len(g),
                "fit_a": round(float(np.mean([x["fit_a"] for x in g])), 3),
                "fit_b": round(float(np.mean([x["fit_b"] for x in g])), 3),
                "diff": round(float(np.mean([x["diff"] for x in g])), 3),
            }
        res["curve"] = curve

        # Crossing point: first sign change of the mean difference.
        tvals = sorted(by_t)
        dvals = [curve[f"{t:.2f}"]["diff"] for t in tvals]
        res["crossing_t"] = None
        for i in range(len(tvals) - 1):
            if dvals[i] <= 0 <= dvals[i + 1] and dvals[i + 1] != dvals[i]:
                frac = -dvals[i] / (dvals[i + 1] - dvals[i])
                res["crossing_t"] = round(
                    tvals[i] + frac * (tvals[i + 1] - tvals[i]), 3)
                break
        res["monotone_diff"] = all(dvals[i] <= dvals[i + 1] + 1e-9
                                   for i in range(len(dvals) - 1))

        # Secondary, per-point view: an interior t counts only if it differs
        # from BOTH ends. See _ramp_stats.graded_beyond_step for why this is not
        # the primary verdict for a sigmoid.
        lo_t, hi_t = tvals[0], tvals[-1]
        d_lo = [r["diff"] for r in by_t[lo_t]]
        d_hi = [r["diff"] for r in by_t[hi_t]]
        graded = {}
        for t in tvals[1:-1]:
            d_t = [r["diff"] for r in by_t[t]]
            _, p_lo = mean_diff_p(d_t, d_lo)
            _, p_hi = mean_diff_p(d_t, d_hi)
            floor = max(min_two_sample_p(len(d_t), len(d_lo)),
                        min_two_sample_p(len(d_t), len(d_hi)))
            graded[f"{t:.2f}"] = {
                "p_vs_start": p_lo, "p_vs_end": p_hi,
                "min_achievable_p": round(floor, 4),
                "underpowered": bool(floor > 0.05),
                "intermediate": bool(p_lo is not None and p_hi is not None
                                     and floor <= 0.05
                                     and p_lo < 0.05 and p_hi < 0.05),
            }
        res["gradedness"] = graded
        res["shape"] = graded_beyond_step([[r["diff"] for r in by_t[t]]
                                           for t in tvals])
        res["n_graded_interior"] = sum(1 for v in graded.values() if v["intermediate"])
        res["n_interior"] = len(graded)
        # If p<0.05 is unreachable at these group sizes the verdict is "cannot
        # tell", not "not graded". Reporting 0/3 without this would be reporting
        # the sample size as if it were a result.
        res["gradedness_underpowered"] = bool(
            graded and all(v["underpowered"] for v in graded.values()))
        out[path] = res
    return out


# ---------------------------------------------------------------------------
# Intensity ramps
# ---------------------------------------------------------------------------

def analyse_intensity_axis(work_dir: Path, judge_tag: str = "qwen") -> dict:
    """PRIMARY intensity measure: does the judged coordinate contract to neutral?"""
    rows = load_ratings(work_dir, "axis", judge_tag)

    per, meta = defaultdict(dict), {}
    for r in rows:
        per[r["id"]][r["probe"]] = r["rating"]
        meta[r["id"]] = r
    recs = []
    for rid, d in per.items():
        if "valence" in d and "arousal" in d:
            m = meta[rid]
            recs.append({"id": rid, "jv": d["valence"], "ja": d["arousal"],
                         "radius": m.get("meta", {}).get("radius"),
                         "ray": m.get("ray_group")
                                or m.get("meta", {}).get("ray_anchor")})
    recs = [r for r in recs if r["radius"] is not None]
    if not recs:
        return {"error": "no axis ratings with a radius found"}

    jv = np.array([r["jv"] for r in recs], float)
    ja = np.array([r["ja"] for r in recs], float)
    sv, sa = jv.std() or 1.0, ja.std() or 1.0

    origin = [r for r in recs if r["radius"] == 0.0]
    if origin:
        cv = float(np.mean([r["jv"] for r in origin]))
        ca = float(np.mean([r["ja"] for r in origin]))
        centre_src = f"empirical, from {len(origin)} r=0 responses"
    else:
        cv, ca = float(jv.mean()), float(ja.mean())
        centre_src = "grand mean (no r=0 responses present)"

    for r in recs:
        r["jr"] = math.hypot((r["jv"] - cv) / sv, (r["ja"] - ca) / sa)

    out = {"n": len(recs), "neutral_centre": [round(cv, 3), round(ca, 3)],
           "centre_source": centre_src}
    rho, p = level_perm_spearman([r["radius"] for r in recs],
                                 [r["jr"] for r in recs])
    out["pooled"] = {"spearman_r_vs_judged_radius": rho, "p_perm": p}

    by_ray = defaultdict(list)
    for r in recs:
        if r["ray"]:
            by_ray[r["ray"]].append(r)
    # r=0 is shared by every ray, so it joins each one as that ray's floor.
    for ray, rr in list(by_ray.items()):
        for o in origin:
            if o not in rr:
                rr.append(o)

    per_ray = {}
    for ray, rr in sorted(by_ray.items()):
        rho_r, p_r = level_perm_spearman([x["radius"] for x in rr],
                                         [x["jr"] for x in rr])
        by_r = defaultdict(list)
        for x in rr:
            by_r[round(x["radius"], 3)].append(x["jr"])
        per_ray[ray] = {
            "n": len(rr), "spearman": rho_r, "p_perm": p_r,
            "curve": {f"{k:.2f}": round(float(np.mean(v)), 3)
                      for k, v in sorted(by_r.items())},
        }
    out["per_ray"] = per_ray
    if per_ray:
        vals = [v["spearman"] for v in per_ray.values() if v["spearman"] is not None]
        out["mean_spearman"] = round(float(np.mean(vals)), 4) if vals else None
        ps = [v["p_perm"] for v in per_ray.values() if v["p_perm"] is not None]
        out["fisher_combined_p"] = round(fisher_combine(ps), 6) if ps else None
    return out


def analyse_intensity(work_dir: Path, judge_tag: str = "qwen",
                      rubric: str = DEFAULT_RUBRIC) -> dict:
    """SECONDARY intensity measure: the degree rubric plus its control."""
    rows = load_ratings(work_dir, "intensity", judge_tag, rubric)

    by_ray = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ray = r.get("ray_group")
        radius = r.get("meta", {}).get("radius")
        if ray is None or radius is None:
            continue
        by_ray[ray][r["probe"].split(":")[0]].append((float(radius), r["rating"]))

    out = {}
    for ray, probes in sorted(by_ray.items()):
        res = {}
        for role in ("target", "control"):
            pts = probes.get(role, [])
            if not pts:
                continue
            rho, p = level_perm_spearman([x for x, _ in pts], [y for _, y in pts])
            by_r = defaultdict(list)
            for x, y in pts:
                by_r[round(x, 3)].append(y)
            res[role] = {
                "n": len(pts),
                "spearman_r": rho,
                "p_perm": p,
                "curve": {f"{x:.2f}": round(float(np.mean(v)), 3)
                          for x, v in sorted(by_r.items())},
            }
        # The dissociation: the target should track the radius and the control
        # should not. A ray where both rise is measuring expressiveness, not the
        # emotion.
        if "target" in res and "control" in res:
            t_rho, c_rho = res["target"]["spearman_r"], res["control"]["spearman_r"]
            if t_rho is not None and c_rho is not None:
                res["dissociation"] = round(t_rho - c_rho, 4)
        out[ray] = res
    return out


# ---------------------------------------------------------------------------
# Pooled curve
# ---------------------------------------------------------------------------

def analyse_pooled(work_dir: Path, design: str, judge_tag: str = "qwen",
                   min_separation: float = 0.0, levels: str = None,
                   rubric: str = DEFAULT_RUBRIC) -> dict:
    """The average curve over all emotions — does a general pattern hold?

    Per-emotion curves are noisy at a handful of replicates per level, and
    individual rays or paths disagree. Pooling asks the prior question:
    averaged over emotions, does the response change monotonically with the
    conditioning parameter at all?

    Both a raw and a CENTRED pooling are reported. Emotions differ in baseline,
    so a raw average lets a high-baseline emotion dominate the level of the
    curve. The centred version subtracts each emotion's own mean, leaving only
    shape, and is the one to trust for the monotonicity claim.
    """
    out = {"design": design, "min_separation": min_separation,
           "judge": judge_tag, "levels": levels, "rubric": rubric}

    # The interpolation point set holds a superset of two t-grids. Without
    # selecting one, pooling mixes unevenly-populated levels from two designs.
    want_levels = ({round(float(x), 3) for x in levels.split(",") if x.strip()}
                   if levels else None)

    if design == "interpolation":
        rows = load_ratings(work_dir, "dualfit", judge_tag, rubric)
        per, meta = defaultdict(dict), {}
        for r in rows:
            m = r.get("meta", {})
            if m.get("endpoint_a"):
                per[r["id"]][r["probe"]] = r["rating"]
                meta[r["id"]] = m
        recs = []
        for rid, d in per.items():
            # Canonical endpoint label only: a --relabel run adds a second
            # endpoint_a probe on the same audio, and counting both would enter
            # that path twice into the average.
            ka = sorted(k for k in d if k.startswith("endpoint_a:"))
            kb = sorted(k for k in d if k.startswith("endpoint_b:"))
            if not ka or not kb:
                continue
            m = meta[rid]
            canonical = f"endpoint_a:{m['endpoint_a']}"
            key_a = canonical if canonical in d else ka[0]
            if want_levels is not None and round(float(m["t"]), 3) not in want_levels:
                continue
            recs.append({"level": round(float(m["t"]), 3),
                         "group": f"{m['endpoint_a']}-{m['endpoint_b']}",
                         "fit_start": d[key_a], "fit_end": d[kb[0]],
                         "value": d[kb[0]] - d[key_a]})
        if min_separation > 0:
            # Generation used the most inclusive threshold, so any higher one is
            # a strict subset already on disk: raising it needs no rerun.
            before = len({r["group"] for r in recs})
            recs = [r for r in recs if _separation(r["group"]) >= min_separation]
            out["pairs_kept"] = len({r["group"] for r in recs})
            out["pairs_dropped"] = before - out["pairs_kept"]
        series = [("fit_start", "fit to the start anchor"),
                  ("fit_end", "fit to the end anchor"),
                  ("value", "difference: end - start")]
        out["level_name"] = "t"
    else:
        rows = load_ratings(work_dir, "intensity", judge_tag, rubric)
        recs = []
        for r in rows:
            rad = r.get("meta", {}).get("radius")
            if rad is None:
                continue
            if want_levels is not None and round(float(rad), 3) not in want_levels:
                continue
            recs.append({"level": round(float(rad), 3),
                         "group": r.get("ray_group") or r["probe"].split(":")[-1],
                         "role": r["probe"].split(":")[0],
                         "value": r["rating"]})
        series = [("target", "target emotion"), ("control", "opposite emotion")]
        out["level_name"] = "radius"

    def _one(sel, key):
        if not sel:
            return None
        by = defaultdict(list)
        for r in sel:
            by[r["level"]].append(r[key])
        lvls = sorted(by)
        rho, p = level_perm_spearman([r["level"] for r in sel],
                                     [r[key] for r in sel])
        means = [float(np.mean(by[l])) for l in lvls]
        return {
            "n": len(sel),
            "curve": {f"{l:.2f}": round(float(np.mean(by[l])), 3) for l in lvls},
            "n_per_level": {f"{l:.2f}": len(by[l]) for l in lvls},
            "spearman": rho, "p_perm": p,
            "monotone_increasing": all(a <= b + 1e-9 for a, b in zip(means, means[1:])),
            "monotone_decreasing": all(a >= b - 1e-9 for a, b in zip(means, means[1:])),
            "shape": graded_beyond_step([by[l] for l in lvls]),
        }

    for name, label in series:
        if design == "interpolation":
            sel, key = recs, name
        else:
            sel, key = [r for r in recs if r["role"] == name], "value"
        res = _one(sel, key)
        if res:
            res["label"] = label
            out[name] = res

    # Centred: remove each emotion's own mean, keeping only the shape.
    if design == "interpolation":
        sel, key = recs, "value"
    else:
        sel, key = [r for r in recs if r["role"] == "target"], "value"
    if sel:
        mu = defaultdict(list)
        for r in sel:
            mu[r["group"]].append(r[key])
        means = {g: float(np.mean(v)) for g, v in mu.items()}
        cent = [dict(r, centred=r[key] - means[r["group"]]) for r in sel]
        res = _one(cent, "centred")
        res["label"] = "centred over emotions (offset removed)"
        res["n_groups"] = len(means)
        out["centred"] = res
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_interpolation(res: dict):
    print(f"\n{'='*78}")
    print("Inter-emotion interpolation — crossover (dual independent fit ratings)")
    print(f"{'='*78}")
    for path, v in res.items():
        a, b = v["endpoint_a"], v["endpoint_b"]
        print(f"\n  {path}   (n={v['n_responses']} responses)")
        print(f"    {'t':>6}" + "".join(f"{t:>8}" for t in v["curve"]))
        for field, label in (("fit_a", f"fit[{a}]"), ("fit_b", f"fit[{b}]"),
                             ("diff", "B - A")):
            print(f"    {label:>6}" + "".join(
                f"{v['curve'][t][field]:>8.2f}" for t in v["curve"]))
        print(f"    Spearman(t, fit[{a}]) = {v['spearman_t_fit_a']:+.3f} "
              f"(p={v['p_fit_a']})   expected negative")
        print(f"    Spearman(t, fit[{b}]) = {v['spearman_t_fit_b']:+.3f} "
              f"(p={v['p_fit_b']})   expected positive")
        print(f"    Spearman(t, B-A)     = {v['spearman_t_diff']:+.3f} "
              f"(p={v['p_diff']})")
        print(f"    crossing at t = {v['crossing_t']}   monotone: {v['monotone_diff']}")
        sh = v.get("shape")
        if sh:
            verdict = ("GRADED — a single step does not describe it"
                       if sh["p_vs_step_model"] < 0.05 else
                       "cannot rule out a single step")
            print(f"    SHAPE: {verdict}  "
                  f"(stat={sh['statistic']:.3f}, p={sh['p_vs_step_model']} "
                  f"vs step model, {sh['n_levels']} levels)")
        if v.get("gradedness_underpowered"):
            floor = max(g["min_achievable_p"] for g in v["gradedness"].values())
            print(f"    GRADEDNESS: UNDERPOWERED — smallest reachable p is "
                  f"{floor:.2f} at this replicate count, so p<0.05 cannot occur.")
            print(f"                Needs more queries per t; not evidence either way.")
        else:
            print(f"    GRADED interior points: {v['n_graded_interior']}/{v['n_interior']}"
                  f"   (differ from both endpoints at p<0.05)")


def print_intensity(res: dict):
    print(f"\n{'='*78}")
    print("Intensity ramp — degree rubric with an opposite-emotion control")
    print(f"{'='*78}")
    print(f"\n  {'ray':<12}{'target rho':>12}{'p':>8}{'control rho':>13}{'p':>8}"
          f"{'dissoc':>9}")
    for ray, v in res.items():
        t, c = v.get("target", {}), v.get("control", {})
        tr = f"{t.get('spearman_r'):+.3f}" if t.get("spearman_r") is not None else "—"
        cr = f"{c.get('spearman_r'):+.3f}" if c.get("spearman_r") is not None else "—"
        ds = f"{v.get('dissociation'):+.3f}" if v.get("dissociation") is not None else "—"
        print(f"  {ray:<12}{tr:>12}{str(t.get('p_perm')):>8}"
              f"{cr:>13}{str(c.get('p_perm')):>8}{ds:>9}")
    for ray, v in res.items():
        if "target" in v:
            print(f"\n  {ray} target curve:  " +
                  "  ".join(f"r={k}:{x:.2f}" for k, x in v["target"]["curve"].items()))
            if "control" in v:
                print(f"  {' ' * len(ray)} control curve: " +
                      "  ".join(f"r={k}:{x:.2f}" for k, x in v["control"]["curve"].items()))


def print_axis(res: dict):
    print(f"\n{'='*78}")
    print("Intensity ramp — judged VA coordinate vs. radius (PRIMARY)")
    print(f"{'='*78}")
    if res.get("error"):
        print(f"  {res['error']}")
        return
    print(f"  n = {res['n']}   neutral centre = {res['neutral_centre']} "
          f"({res['centre_source']})")
    pooled = res.get("pooled", {})
    print(f"  pooled Spearman(r, judged radius) = "
          f"{pooled.get('spearman_r_vs_judged_radius')}  "
          f"(p={pooled.get('p_perm')})")
    print(f"  mean per-ray Spearman = {res.get('mean_spearman')}   "
          f"Fisher-combined p = {res.get('fisher_combined_p')}")
    print(f"\n  {'ray':<12}{'rho':>10}{'p':>10}{'n':>6}")
    for ray, v in res.get("per_ray", {}).items():
        rho = f"{v['spearman']:+.3f}" if v["spearman"] is not None else "—"
        print(f"  {ray:<12}{rho:>10}{str(v['p_perm']):>10}{v['n']:>6}")


def print_pooled(res: dict):
    print(f"\n{'='*78}")
    print(f"Pooled over all groups (level = {res['level_name']})")
    print(f"{'='*78}")
    if res.get("pairs_dropped"):
        print(f"  min-separation {res['min_separation']}: kept "
              f"{res['pairs_kept']} pairs, dropped {res['pairs_dropped']}")
    for key in ("fit_start", "fit_end", "value", "target", "control", "centred"):
        v = res.get(key)
        if not v:
            continue
        print(f"\n  {v['label']}  (n={v['n']})")
        print("    " + "  ".join(f"{k}:{x:+.3f}" for k, x in v["curve"].items()))
        print(f"    Spearman = {v['spearman']}  (p={v['p_perm']})   "
              f"monotone up={v['monotone_increasing']} down={v['monotone_decreasing']}")
        sh = v.get("shape")
        if sh:
            verdict = ("GRADED beyond a single step"
                       if sh["p_vs_step_model"] < 0.05 else
                       "cannot rule out a single step")
            print(f"    SHAPE: {verdict} (p={sh['p_vs_step_model']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Curve statistics for the VA interpolation analyses")
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="Directory holding results/ from judge_interpolation.py")
    ap.add_argument("--design", required=True, choices=sorted(DESIGN_MODES))
    ap.add_argument("--judge", default="qwen", choices=["qwen", "gemini"])
    ap.add_argument("--rubric", default=DEFAULT_RUBRIC,
                    choices=[DEFAULT_RUBRIC, "published"],
                    help="Which rubric's judgments to read. They share "
                         "(id, probe) keys on different scales and must never "
                         "be merged.")
    ap.add_argument("--levels", default=None,
                    help="Restrict pooling to these t / radius values, e.g. "
                         "'0,0.25,0.5,0.75,1.0'")
    ap.add_argument("--min-separation", type=float, default=0.0,
                    help="interpolation only: report on anchor pairs at least "
                         "this far apart in VA space. Generation used the most "
                         "inclusive threshold, so raising this is free.")
    return ap.parse_args()


def main():
    args = parse_args()
    res_dir = args.work_dir / "results"
    rub = "" if args.rubric == DEFAULT_RUBRIC else f"_{args.rubric}"

    if args.design == "interpolation":
        res = analyse_interpolation(args.work_dir, args.judge, args.rubric)
        print_interpolation(res)
    else:
        res = {"degree_rubric": analyse_intensity(args.work_dir, args.judge,
                                                  args.rubric)}
        print_intensity(res["degree_rubric"])
        try:
            res["axis"] = analyse_intensity_axis(args.work_dir, args.judge)
            print_axis(res["axis"])
        except SystemExit as e:
            # The axis mode is a separate judging pass; report its absence
            # rather than losing the degree-rubric results that did load.
            print(f"\n  (axis ratings unavailable: {e})")

    out_path = res_dir / f"ramps_{args.judge}{rub}.json"
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nSaved {out_path}")

    pooled = analyse_pooled(args.work_dir, args.design, args.judge,
                            args.min_separation, args.levels, args.rubric)
    print_pooled(pooled)
    sep = f"_sep{args.min_separation}" if args.min_separation > 0 else ""
    pooled_path = res_dir / f"ramps_pooled_{args.judge}{rub}{sep}.json"
    pooled_path.write_text(json.dumps(pooled, indent=2))
    print(f"\nSaved {pooled_path}")


if __name__ == "__main__":
    main()
