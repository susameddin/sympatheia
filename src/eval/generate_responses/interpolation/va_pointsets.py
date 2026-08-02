#!/usr/bin/env python3
"""
Build the VA conditioning point sets for the two interpolation analyses.

Both analyses ask whether the model's response varies *gradually* along a path
through valence-arousal space, rather than snapping to the nearest of the 12
emotion anchors. They differ in which path is walked:

  intensity      Each anchor's coordinate scaled by a radius r in [0, 1], i.e. a
                 straight line from the VA origin out to the anchor. Tests
                 whether the *strength* of the expressed emotion tracks r.

  interpolation  A straight line from anchor A to anchor B, parameterised by
                 t in [0, 1]. Tests whether one emotion fades out as the other
                 fades in, and whether the transition is graded rather than a
                 step at the Voronoi boundary between the two cells.

A *point* is one conditioning coordinate plus the anchor whose Voronoi cell it
falls in. Point schema:

    {
      "point_id":        str,     # unique within the point set
      "valence":         float,   # the conditioning coordinate itself
      "arousal":         float,
      "anchor":          str,     # nearest anchor by Euclidean distance
      "anchor_valence":  float,
      "anchor_arousal":  float,
      "dist_to_anchor":  float,
      "group":           str,     # "ray:Happy", "interp:Happy-Sad", ...
      "emotion":         str,     # label used by the judge and by aggregation
      "meta":            dict,    # design-specific extras (radius / t / endpoints)
      "query_audio":     str|None # pin a specific query; None means "sample one"
    }

Usage (run from src/):
    python -m eval.generate_responses.interpolation.va_pointsets \\
        --design intensity --out /path/to/interp/intensity/points.json --verify

    python -m eval.generate_responses.interpolation.va_pointsets \\
        --design interpolation --out /path/to/interp/interpolation/points.json --verify

Point ids depend only on the anchor names and the level (r or t), so re-running
with a longer level list or a lower --min-separation is purely additive: every
previously generated response keeps its id and is reused rather than orphaned.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# parents[4] is the repo root, which is what `from src.constants ...` resolves
# against; parents[3] is src/.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3]))
sys.path.insert(0, str(_HERE.parents[4]))

from src.constants import EMOTION_VA_MAPPING

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anchor list as (name, v, a), order fixed for determinism.
ANCHORS = [(name, va[0], va[1]) for name, va in EMOTION_VA_MAPPING.items()]

# Every anchor except Neutral. Neutral's ray is degenerate: Neutral IS the
# origin, so scaling it by r leaves it at (0, 0) for every r.
DEFAULT_INTENSITY_ANCHORS = [n for n in EMOTION_VA_MAPPING if n != "Neutral"]

# r=0 is the origin, which is the Neutral coordinate itself — the natural floor
# of every ramp, and shared by all of them (build_intensity emits it once).
DEFAULT_RADII = [0.0, 0.25, 0.5, 0.75, 1.0]

# A union of two grids, not a replacement: the 0.2 grid (6 levels between the
# endpoints) and the 0.25 grid (5 levels). Ids are f"..._t{t:.2f}" so 0.20 and
# 0.25 never collide, and downstream stages select one grid with --levels. The
# two must never be pooled — they are different level spacings, not replicates.
DEFAULT_LEVELS = [0.0, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0]

# Minimum VA separation for an anchor pair to be worth interpolating between.
# A principled inclusion rule beats a curated list. Gradedness is expected to
# degrade as the endpoints get closer, which is what a continuous representation
# implies and a quantised one does not, so the threshold is a knob the analysis
# can sweep rather than a fixed choice. Below roughly 0.6 the nearest anchor
# pairs are close enough that "interpolating between them" is not a meaningful
# crossover at all.
DEFAULT_MIN_SEPARATION = 1.4


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def nearest_anchor(v: float, a: float):
    """Return (name, av, aa, distance) for the closest anchor."""
    best = min(ANCHORS, key=lambda t: (t[1] - v) ** 2 + (t[2] - a) ** 2)
    name, av, aa = best
    return name, av, aa, math.hypot(av - v, aa - a)


def in_bounds(v: float, a: float) -> bool:
    return -1.0 <= v <= 1.0 and -1.0 <= a <= 1.0


def make_point(point_id: str, v: float, a: float, group: str,
               emotion: str = None, query_audio: str = None, **meta) -> dict:
    """Wrap one conditioning coordinate with its Voronoi-cell metadata.

    Coordinates are stored at full precision, not rounded: the conditioning
    prompt renders them to 2 dp, and rounding here first can move a coordinate
    across a 2-dp boundary and change the prompt the model actually sees.
    """
    name, av, aa, dist = nearest_anchor(v, a)
    return {
        "point_id": point_id,
        "valence": float(v),
        "arousal": float(a),
        "anchor": name,
        "anchor_valence": av,
        "anchor_arousal": aa,
        "dist_to_anchor": round(dist, 4),
        "group": group,
        "emotion": emotion if emotion is not None else name,
        "query_audio": query_audio,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Intensity ramps: origin -> anchor
# ---------------------------------------------------------------------------

def build_intensity(anchors=None, radii=None) -> list:
    """Anchor coordinates scaled toward the origin.

    r=0 is the origin (0, 0) — literally the Neutral coordinate — and is the
    same point for every ray, so it is emitted ONCE and reused as the shared
    floor of all of them. Emitting it per ray would spend generation time on
    identical prompts and, worse, give each ray its own independent sample of
    decoding noise at the one radius they all have in common.
    """
    anchors = anchors if anchors is not None else DEFAULT_INTENSITY_ANCHORS
    radii = radii if radii is not None else DEFAULT_RADII

    pts, has_origin = [], False
    for anchor in anchors:
        av, aa = EMOTION_VA_MAPPING[anchor]
        for r in radii:
            if r == 0.0:
                has_origin = True
                continue
            pts.append(make_point(
                point_id=f"ray_{anchor.lower()}_r{r:.2f}",
                v=av * r, a=aa * r,
                group=f"ray:{anchor}",
                emotion=anchor,
                radius=r, ray_anchor=anchor,
            ))
    if has_origin:
        pts.append(make_point(
            point_id="ray_origin_r0.00", v=0.0, a=0.0,
            group="ray:origin", emotion="Neutral",
            radius=0.0, ray_anchor=None, shared_origin=True,
        ))
    return pts


def ray_records(records: list) -> dict:
    """{anchor: [record, ...]} with the shared origin spliced into every ray.

    The origin belongs to every ray at r=0 but is stored once, so any per-ray
    analysis has to add it back. It is spliced in PER QUERY so the pairing stays
    within-query — an origin clip generated from a different query is not a
    valid floor for this ray's ramp. Skipping this step silently drops the r=0
    level from every ramp.
    """
    origin = [r for r in records if r["point_id"].startswith("ray_origin")]
    by_query = defaultdict(list)
    for r in origin:
        by_query[r["query_audio"]].append(r)

    out = defaultdict(list)
    for r in records:
        anchor = r.get("meta", {}).get("ray_anchor")
        if anchor:
            out[anchor].append(r)
    for anchor in list(out):
        for r in list(out[anchor]):
            for o in by_query.get(r["query_audio"], []):
                if o not in out[anchor]:
                    out[anchor].append(o)
    return dict(out)


# ---------------------------------------------------------------------------
# Inter-emotion interpolation: anchor -> anchor
# ---------------------------------------------------------------------------

# Endpoint order for the pairs generated before the separation rule existed. A
# point_id is f"interp_{a}_{b}_t{t}", so emitting ("Sad", "Happy") where an
# earlier run wrote ("Happy", "Sad") mints new ids and orphans the responses and
# judgments already on disk. Pinning these four keeps every existing response in
# play and keeps the reported curves pointing the same way.
LEGACY_PAIR_ORIENTATION = {
    frozenset(("Happy", "Sad")): ("Happy", "Sad"),
    frozenset(("Anxious", "Relaxed")): ("Anxious", "Relaxed"),
    frozenset(("Angry", "Content")): ("Angry", "Content"),
    frozenset(("Excited", "Tired")): ("Excited", "Tired"),
}


def _orient(a: str, b: str) -> tuple:
    return LEGACY_PAIR_ORIENTATION.get(frozenset((a, b)), (a, b))


def pairs_by_separation(d_min: float, exclude=("Neutral",)) -> list:
    """Every anchor pair at least ``d_min`` apart in VA space.

    Ordered by DECREASING separation, so lowering the threshold only ever
    appends pairs. Combined with point ids that depend solely on the endpoint
    names and t, that makes a later, lower-threshold run purely additive over an
    earlier one — no response is orphaned or regenerated.
    """
    names = [n for n in EMOTION_VA_MAPPING if n not in exclude]
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = math.dist(EMOTION_VA_MAPPING[a], EMOTION_VA_MAPPING[b])
            if d >= d_min:
                out.append((_orient(a, b), d))
    out.sort(key=lambda t: -t[1])
    return [p for p, _ in out]


def build_interpolation(pairs=None, levels=None, min_separation=None) -> list:
    """Straight-line interpolation between each anchor pair.

    ``emotion`` flips at the midpoint so that each response carries the label of
    whichever endpoint it is nearer; the judge rates fit to BOTH endpoints
    independently regardless, so this label only affects aggregation.
    """
    if pairs is None:
        d_min = min_separation if min_separation is not None else DEFAULT_MIN_SEPARATION
        pairs = pairs_by_separation(d_min)
    levels = levels if levels is not None else DEFAULT_LEVELS

    pts = []
    for a1, a2 in pairs:
        v1, r1 = EMOTION_VA_MAPPING[a1]
        v2, r2 = EMOTION_VA_MAPPING[a2]
        for t in levels:
            pts.append(make_point(
                point_id=f"interp_{a1.lower()}_{a2.lower()}_t{t:.2f}",
                v=v1 + t * (v2 - v1),
                a=r1 + t * (r2 - r1),
                group=f"interp:{a1}-{a2}",
                emotion=a1 if t < 0.5 else a2,
                t=t, endpoint_a=a1, endpoint_b=a2,
            ))
    return pts


BUILDERS = {
    "intensity": build_intensity,
    "interpolation": build_interpolation,
}


# ---------------------------------------------------------------------------
# Verification — geometry invariants every downstream claim depends on
# ---------------------------------------------------------------------------

def verify(points: list, design: str) -> list:
    """Return a list of failure messages (empty means everything checks out)."""
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    for name, v, a in ANCHORS:
        check(nearest_anchor(v, a)[0] == name,
              f"anchor {name} is not its own nearest anchor")
        check(in_bounds(v, a), f"anchor {name} out of bounds")

    for p in points:
        check(in_bounds(p["valence"], p["arousal"]),
              f"{p['point_id']} out of [-1,1]^2")

    ids = [p["point_id"] for p in points]
    check(len(set(ids)) == len(ids), "duplicate point ids")

    if design == "intensity":
        # The origin must be shared across rays, not generated once per anchor.
        origins = [p for p in points if p["meta"].get("radius") == 0.0]
        check(len(origins) <= 1,
              f"{len(origins)} r=0 points; the origin must be shared across rays")
        for p in points:
            r = p["meta"].get("radius")
            anchor = p["meta"].get("ray_anchor")
            if anchor and r:
                av, aa = EMOTION_VA_MAPPING[anchor]
                check(abs(p["valence"] - av * r) < 1e-9
                      and abs(p["arousal"] - aa * r) < 1e-9,
                      f"{p['point_id']} is not its anchor scaled by r={r}")

    if design == "interpolation":
        for p in points:
            m = p["meta"]
            a1, a2, t = m.get("endpoint_a"), m.get("endpoint_b"), m.get("t")
            if a1 is None:
                continue
            v1, r1 = EMOTION_VA_MAPPING[a1]
            v2, r2 = EMOTION_VA_MAPPING[a2]
            check(abs(p["valence"] - (v1 + t * (v2 - v1))) < 1e-9
                  and abs(p["arousal"] - (r1 + t * (r2 - r1))) < 1e-9,
                  f"{p['point_id']} is not on the segment {a1}->{a2}")
        # Both endpoints must be present, or there is no curve to fit.
        by_group = defaultdict(set)
        for p in points:
            by_group[p["group"]].add(p["meta"].get("t"))
        for group, ts in by_group.items():
            check(0.0 in ts and 1.0 in ts,
                  f"{group} is missing an endpoint (t=0 and t=1 are both required)")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build VA point sets for the intensity and inter-emotion "
                    "interpolation analyses")
    p.add_argument("--design", required=True, choices=sorted(BUILDERS),
                   help="Which path to walk through VA space.")
    p.add_argument("--out", type=Path, required=True,
                   help="Where to write the point-set JSON.")
    p.add_argument("--anchors", nargs="+", default=None,
                   help="intensity only: anchors to build rays for. "
                        f"Default: every anchor except Neutral "
                        f"({len(DEFAULT_INTENSITY_ANCHORS)} of them).")
    p.add_argument("--radii", type=float, nargs="+", default=None,
                   help=f"intensity only: ray radii. Default: {DEFAULT_RADII}")
    p.add_argument("--levels", type=float, nargs="+", default=None,
                   help="interpolation only: t values along each segment. "
                        f"Default: {DEFAULT_LEVELS}")
    p.add_argument("--min-separation", type=float, default=DEFAULT_MIN_SEPARATION,
                   help="interpolation only: minimum VA distance between the two "
                        f"endpoints of a pair (default: {DEFAULT_MIN_SEPARATION}).")
    p.add_argument("--verify", action="store_true",
                   help="Run the geometry checks and exit non-zero on failure.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.design == "intensity":
        points = build_intensity(anchors=args.anchors, radii=args.radii)
    else:
        points = build_interpolation(levels=args.levels,
                                     min_separation=args.min_separation)

    groups = sorted({p["group"] for p in points})
    print(f"Design      : {args.design}")
    print(f"Points      : {len(points)}")
    print(f"Groups      : {len(groups)}")
    for g in groups:
        n = sum(1 for p in points if p["group"] == g)
        print(f"  {g:<28} {n} points")

    if args.verify:
        failures = verify(points, args.design)
        if failures:
            print(f"\nFAILED {len(failures)} geometry check(s):", file=sys.stderr)
            for m in failures:
                print(f"  - {m}", file=sys.stderr)
            sys.exit(1)
        print("\nGeometry checks passed.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(points, indent=2))
    print(f"\nWrote {args.out}")
    print(f"\nNext step:")
    print(f"  python -m eval.generate_responses.interpolation."
          f"generate_responses_interpolation \\")
    print(f"      --points {args.out} \\")
    print(f"      --checkpoint /path/to/checkpoint \\")
    print(f"      --query-dir /path/to/neutral/query/audio \\")
    print(f"      --work-dir {args.out.parent}")


if __name__ == "__main__":
    main()
