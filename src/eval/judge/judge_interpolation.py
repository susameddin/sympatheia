#!/usr/bin/env python3
"""
Single-clip audio rubrics for the two VA interpolation analyses.

One clip, one question, one number. Every quantity that needs a *comparison* is
obtained by asking two independent questions about the same clip rather than by
playing two clips in one prompt: audio judges are prone to position collapse
when given a pair, and asking for both numbers in a single response lets the
model trade them off against each other, which manufactures the very
anti-correlation these analyses set out to measure. The two probes of a mode are
therefore always separate model calls.

Three modes:

  axis        Two SAM-style questions per clip — how pleasant, and how
              activated, is the state the assistant is responding to. 1-7, each
              prompt instructing the judge to disregard the other dimension.
              This is the PRIMARY measure for the intensity analysis.

  intensity   How strongly the reply is pitched at the ray's emotion, plus the
              same question about that emotion's opposite as a control. 1-7. A
              ray where both rise is measuring expressiveness, not the emotion.

  dualfit     How well the reply suits each of the two endpoint emotions of an
              interpolation path. 1-5, asked once per endpoint.

Reads the manifests written by
eval/generate_responses/interpolation/generate_responses_interpolation.py and
writes, under --work-dir:

  results/single_{mode}_{judge}{suffix}.shard{ii}of{NN}.jsonl
  results/single_{mode}_{judge}{suffix}.shard{ii}of{NN}_summary.json

Usage (run from src/):
    python -m eval.judge.judge_interpolation --work-dir /path/to/interp/intensity \\
        --mode axis --backend qwen --model-path /path/to/Qwen3-Omni --skip-existing

    python -m eval.judge.judge_interpolation --work-dir /path/to/interp/interpolation \\
        --mode dualfit --backend gemini --levels 0,0.25,0.5,0.75,1.0 --skip-existing

Judging, not generation, dominates the schedule, so shard it the same way:
``--shard i/N`` with one shard per GPU (or per API worker).
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

# src/ on sys.path so `eval.judge...` resolves whether this is run as
# `python -m eval.judge.<name>` or as a plain script path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.generate_responses.interpolation.va_pointsets import (
    ANCHORS, nearest_anchor, ray_records)
from eval.judge import judge_backends

# ---------------------------------------------------------------------------
# axis — SAM-style valence and arousal, 1-7
#
# Each prompt tells the judge to disregard the other dimension. That cannot
# manufacture the result: an instruction to ignore arousal cannot create a
# correlation between judged valence and *target* valence, it can only strip out
# cross-talk. If the model did not separate the two axes in the first place, no
# instruction would let the judge read them separately.
# ---------------------------------------------------------------------------

AXIS_SYSTEM = """\
You are an expert evaluator of emotion-adaptive conversational AI.

You will hear ONE spoken reply from a voice assistant. The user's own message was \
emotionally NEUTRAL; the assistant was separately told how the user is feeling. \
Your task is to infer, from the reply alone, {what}.

Rate on a scale from 1 to 7:
  1 — {low}
  4 — {mid}
  7 — {high}
Use the intermediate numbers 2, 3, 5 and 6 for states in between.

Judge BOTH what is said and how it sounds. {ignore}

Respond with ONLY the number (1-7) followed by a vertical bar and a brief \
one-sentence justification.
Format exactly: <rating>|<justification>
Example: 5|The warm phrasing and easy pacing suggest a mildly positive state.\
"""

AXIS_PROBES = {
    "valence": dict(
        what="how PLEASANT or UNPLEASANT the user's feeling was",
        low="very unpleasant — miserable, distressed, negative",
        mid="neutral — neither pleasant nor unpleasant",
        high="very pleasant — happy, positive, content",
        ignore="Ignore how energetic or subdued the state is; that is a separate "
               "dimension and must not influence this rating.",
        user="Rate how pleasant or unpleasant the user's feeling was, from 1 "
             "(very unpleasant) to 7 (very pleasant)."),
    "arousal": dict(
        what="how ACTIVATED or CALM the user's feeling was",
        low="very calm — subdued, still, low energy, sleepy",
        mid="moderately activated",
        high="very activated — intense, agitated, high energy, excited",
        ignore="Ignore whether the state is pleasant or unpleasant; that is a "
               "separate dimension and must not influence this rating.",
        user="Rate how activated or calm the user's feeling was, from 1 (very "
             "calm) to 7 (very activated)."),
}

# ---------------------------------------------------------------------------
# intensity — degree, with an opposite-emotion control
#
# The scale is 1-7 rather than 1-5, and the rubric explicitly reserves the top
# of the range. A narrower scale saturates here: a warm, helpful reply aimed at
# a barely-sad user is still a good reply, so a judge asked "how well does this
# suit a sad user" rewards the assistant's habitual warmth instead of rating
# degree, and the ramp flattens against the ceiling. This is the SECONDARY
# measure for the intensity analysis; the primary one is the `axis` mode above.
# ---------------------------------------------------------------------------

INTENSITY_SYSTEM = """\
You are an expert evaluator of emotion-adaptive conversational AI.

You will hear ONE spoken reply from a voice assistant. The user's own message was \
emotionally NEUTRAL; the assistant was separately told how the user is feeling. \
Your task is to judge HOW STRONGLY the reply is pitched at a user who is feeling \
{emotion}.

Rate on a scale from 1 to 7:
  1 — not at all; nothing about this reply suggests a {emotion} user
  2 — a faint hint at most
  3 — mildly {emotion}
  4 — moderately {emotion}
  5 — clearly {emotion}
  6 — strongly {emotion}
  7 — extremely {emotion}; pitched at someone feeling this as intensely as possible

This is a question about DEGREE, not about quality. A warm, well-written, helpful \
reply aimed at a barely-{emotion} user should score LOW. Reserve 6 and 7 for \
replies that would be excessive for anyone only mildly {emotion}.

Judge BOTH what is said and how it sounds.

Respond with ONLY the number (1-7) followed by a vertical bar and a brief \
one-sentence justification.
Format exactly: <rating>|<justification>\
"""

INTENSITY_USER = ("How strongly is this reply pitched at a user who is feeling "
                  "{emotion}? Answer 1 (not at all) to 7 (extremely).")

# ---------------------------------------------------------------------------
# dualfit / named-emotion fit — 1-5
# ---------------------------------------------------------------------------

FIT_SYSTEM = """\
You are an expert evaluator of emotion-adaptive conversational AI. Your task is \
to rate how well a voice assistant's spoken response suits a user who is \
feeling a specific way.

IMPORTANT CONTEXT: The user's audio input is emotionally NEUTRAL. The assistant \
was separately informed of the user's emotional state. You are evaluating \
whether the response is well suited to that state, NOT whether the user sounds \
emotional.

Rate the response on a scale from 1 to 5 using this rubric:
  1 — Completely unsuited; ignores or contradicts the stated state
  2 — Minimal fit; largely generic regardless of the stated state
  3 — Some fit, but inconsistent, superficial, or aimed at a different feeling
  4 — Clear fit; tone and content are appropriate for someone feeling this way
  5 — Excellent fit: clearly tailored to this particular state, with \
appropriate tone, word choice, and support

Judge BOTH what is said and how it sounds.

Respond with ONLY the rating number (1-5) followed by a vertical bar and a \
brief one-sentence justification.
Format exactly: <rating>|<justification>
Example: 4|The gentle pacing and acknowledgement of self-blame suit someone feeling guilty.\
"""

FIT_USER = ("The user is feeling {state}. Listen to the voice assistant's "
            "response below and rate how well it suits someone feeling {state}.")

# The alternative rubric reuses the main evaluation's neutral-setting judge
# prompt verbatim, with one change to the user turn: the
# "(valence=..., arousal=...)" parenthetical is dropped, leaving only the
# emotion name. Two reasons:
#
#   1. Neither analysis has a single coordinate that belongs in that sentence.
#      The interpolation analysis conditions on a point BETWEEN two anchors and
#      scores against each in turn; the intensity analysis conditions on a point
#      part-way along a ray. Supplying the true coordinate would also leak the
#      level (t or r) across the two calls that are supposed to be independent.
#   2. Showing the judge the coordinate next to the emotion name is exactly the
#      confound that makes a robustness sweep unable to distinguish a continuous
#      model from one that snaps to the nearest anchor. Dropping it forces the
#      judge to score from the audio alone, which is stricter.
#
# The premise "the model was told the user is feeling X" is false at interior
# levels — but identically so for both probes, and both analyses' statistics are
# DIFFERENCES between the two probes, so a shared bias cancels. At the endpoints
# (t=0, t=1, r=1) it is true outright.
PUBLISHED_USER = ("The model was told the user is feeling {state}, but the "
                  "user's actual audio was emotionally neutral. Listen to the "
                  "voice assistant's response below and rate how well it adapts "
                  "to the stated emotion.")

ARM = "continuous"
DEFAULT_RUBRIC = "custom"


def fit_rubric(rubric: str):
    """(system_prompt, user_template) for the named-emotion probes."""
    if rubric == "published":
        # Imported lazily: the module pulls in the Qwen stack, which the Gemini
        # backend does not otherwise need.
        from eval.judge.judge_qwen3omni_neutral import JUDGE_SYSTEM_PROMPT
        return JUDGE_SYSTEM_PROMPT, PUBLISHED_USER
    return FIT_SYSTEM, FIT_USER


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------

def make_probes(mode: str, record: dict, rubric: str, relabel: dict) -> list:
    """[(probe_key, system_prompt, user_text)] for one response."""
    def label(emotion):
        return relabel.get(emotion, emotion)

    if mode == "axis":
        return [(name, AXIS_SYSTEM.format(**p), p["user"])
                for name, p in AXIS_PROBES.items()]

    if mode == "intensity":
        target = record["meta"].get("ray_anchor") or record.get("probe_emotion")
        if not target:
            return []
        tv, ta = dict((n, (v, a)) for n, v, a in ANCHORS)[target]
        # The control is the anchor opposite the target through the origin: a
        # ramp on which BOTH probes rise is measuring expressiveness, not the
        # target emotion.
        control = nearest_anchor(-tv, -ta)[0]
        roles = (("target", label(target)), ("control", label(control)))
        if rubric == "published":
            system, user_tmpl = fit_rubric(rubric)
            return [(f"{role}:{emo}", system, user_tmpl.format(state=emo))
                    for role, emo in roles]
        return [(f"{role}:{emo}",
                 INTENSITY_SYSTEM.format(emotion=emo),
                 INTENSITY_USER.format(emotion=emo))
                for role, emo in roles]

    if mode == "dualfit":
        system, user_tmpl = fit_rubric(rubric)
        pairs = (("endpoint_a", label(record["meta"]["endpoint_a"])),
                 ("endpoint_b", label(record["meta"]["endpoint_b"])))
        return [(f"{role}:{emo}", system, user_tmpl.format(state=emo))
                for role, emo in pairs]

    raise ValueError(f"unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Work list
# ---------------------------------------------------------------------------

def load_manifest(work_dir: Path) -> list:
    """Merge every shard manifest, keeping the first record per id."""
    man_dir = work_dir / "manifests"
    shards = sorted(man_dir.glob(f"{ARM}.shard*.jsonl"))
    if not shards:
        raise FileNotFoundError(
            f"no shard manifests in {man_dir}\n"
            f"  run: python -m eval.generate_responses.interpolation."
            f"generate_responses_interpolation --work-dir {work_dir} ...")
    seen, records = set(), []
    for s in shards:
        with open(s) as f:
            for line in f:
                r = json.loads(line)
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                records.append(r)
    return sorted(records, key=lambda r: r["id"])


def _query_index(record_id: str) -> int:
    m = re.search(r"__q(\d+)$", record_id)
    return int(m.group(1)) if m else 0


def build_work(work_dir: Path, mode: str, rubric: str, relabel: dict,
               only_group: str = None, levels: str = None,
               max_query: int = None) -> list:
    """[(record, probe_key, system, user)] — one entry per judgment."""
    records = [r for r in load_manifest(work_dir) if r.get("response_audio")]

    if only_group:
        prefixes = tuple(g.strip() for g in only_group.split(",") if g.strip())
        records = [r for r in records if r["group"].startswith(prefixes)]
        if not records:
            raise SystemExit(f"no responses with group prefix(es) {prefixes!r}")

    if max_query is not None:
        # Ids end in __q{i}; keeping the low indices thins replicates while
        # leaving every distinct coordinate represented, which is the right axis
        # to cut on a budget.
        records = [r for r in records if _query_index(r["id"]) < max_query]
        if not records:
            raise SystemExit(f"no responses with query index < {max_query}")

    if levels:
        # The interpolation point set carries a UNION of two t-grids (0.2 and
        # 0.25 steps). They are different level spacings, not replicates, so a
        # run selects one; judging the union also costs nearly twice as much on
        # a paid backend.
        want = {round(float(x), 3) for x in levels.split(",") if x.strip()}

        def _level(r):
            m = r.get("meta", {})
            v = m.get("t") if m.get("t") is not None else m.get("radius")
            return None if v is None else round(float(v), 3)

        records = [r for r in records if _level(r) in want]
        if not records:
            raise SystemExit(f"no responses at levels {sorted(want)}")

    if mode == "intensity":
        # The r=0 origin belongs to every ray, so it is probed once per ray's
        # emotion rather than once overall — otherwise every ramp loses its floor.
        expanded = []
        for anchor, rows in sorted(ray_records(records).items()):
            for r in rows:
                expanded.append(dict(r, probe_emotion=anchor, ray_group=anchor))
        records = expanded

    work = []
    for r in records:
        for key, system, user in make_probes(mode, r, rubric, relabel):
            work.append((r, key, system, user))
    return work


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(judgments: list, mode: str) -> dict:
    scored = [j for j in judgments if j.get("rating") is not None]
    out = {"n_judgments": len(judgments), "n_parsed": len(scored),
           "parse_rate": round(len(scored) / len(judgments), 4) if judgments else 0.0,
           "mode": mode}
    if not scored:
        return out

    by_probe = defaultdict(list)
    for j in scored:
        by_probe[j["probe"].split(":")[0]].append(j["rating"])
    out["per_probe"] = {
        k: {"n": len(v), "mean": round(statistics.fmean(v), 3),
            "sd": round(statistics.stdev(v) if len(v) > 1 else 0.0, 3),
            # A judge that answers the same number every time is not measuring
            # anything; this is the rating-scale analogue of position collapse.
            "distinct_values": len(set(v))}
        for k, v in sorted(by_probe.items())
    }
    return out


def print_summary(work_dir: Path, s: dict):
    print(f"\n{'='*72}")
    print(f"{work_dir.name} / {s['mode']} — {s['n_parsed']}/{s['n_judgments']} parsed")
    print(f"{'='*72}")
    for probe, v in s.get("per_probe", {}).items():
        flag = "  <-- CONSTANT, no signal" if v["distinct_values"] < 2 else ""
        print(f"  {probe:<14} mean={v['mean']:.3f}  sd={v['sd']:.3f}  "
              f"n={v['n']}  distinct={v['distinct_values']}{flag}")
    print(f"{'='*72}\n")


def reaggregate(work_dir: Path, mode: str, judge_tag: str):
    """Merge every shard of one (mode, judge) and rewrite the summary."""
    res_dir = work_dir / "results"
    merged = res_dir / f"single_{mode}_{judge_tag}.jsonl"
    if not merged.exists():
        shards = sorted(res_dir.glob(f"single_{mode}_{judge_tag}.shard*.jsonl"))
        if not shards:
            raise SystemExit(f"no judgments found under {res_dir}")
        rows, seen = [], set()
        for s in shards:
            for line in open(s):
                r = json.loads(line)
                k = (r["id"], r["probe"])
                if k not in seen:
                    seen.add(k)
                    rows.append(r)
        merged.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"merged {len(shards)} shard(s), {len(rows)} judgments -> {merged}")
    rows = [json.loads(l) for l in open(merged)]
    summ = aggregate(rows, mode)
    (res_dir / f"single_{mode}_{judge_tag}_summary.json").write_text(
        json.dumps(summ, indent=2))
    print_summary(work_dir, summ)
    return summ


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------

def output_suffix(only_group, relabel, rubric, max_query, levels) -> str:
    """Encode the run's variant flags into the output filename.

    Every non-default option gets its own files, so a variant run can never
    overwrite the working configuration, and the analysis stage can tell the
    variants apart by filename alone.
    """
    sfx = ""
    if only_group:
        # A group filter may be a comma-separated list of prefixes containing
        # ":" and ",", which make for hostile filenames and break downstream
        # globs. Keep it readable but filesystem-safe and bounded.
        clean = re.sub(r"[^A-Za-z0-9]+", "-", only_group).strip("-")
        sfx = "_" + (clean[:48] if len(clean) <= 48 else clean[:44] + "-etc")
    if relabel:
        sfx += "_relabel-" + "-".join(sorted(relabel.values()))
    if rubric != DEFAULT_RUBRIC:
        sfx += f"_rubric-{rubric}"
    if max_query is not None:
        sfx += f"_q{max_query}"
    if levels:
        sfx += "_lv-" + re.sub(r"[^0-9]+", "-", levels).strip("-")
    return sfx


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Single-clip audio rubrics for the VA interpolation analyses")
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="Directory holding manifests/ from the generation step; "
                         "results/ is written inside it")
    ap.add_argument("--mode", required=True,
                    choices=["axis", "intensity", "dualfit"])
    ap.add_argument("--backend", default="qwen", choices=["qwen", "gemini"])
    ap.add_argument("--model-path", default=None,
                    help="Judge model path or id; defaults to the backend's own")
    ap.add_argument("--rubric", default=DEFAULT_RUBRIC,
                    choices=[DEFAULT_RUBRIC, "published"],
                    help="dualfit/intensity only. 'published' reuses the main "
                         "evaluation's judge prompt with the VA parenthetical "
                         "dropped, and writes to its own files.")
    ap.add_argument("--relabel", default=None,
                    help="Judge-facing label swaps, e.g. 'Anxious=Afraid'. Costs "
                         "no regeneration: the model is conditioned on a "
                         "coordinate and never sees the emotion word.")
    ap.add_argument("--only-group", default=None,
                    help="Restrict to responses whose group starts with this "
                         "prefix (comma-separated list allowed)")
    ap.add_argument("--levels", default=None,
                    help="Comma-separated t (interpolation) or radius "
                         "(intensity) values to judge, e.g. "
                         "'0,0.25,0.5,0.75,1.0'. Selects one grid out of a "
                         "point set holding a superset of two.")
    ap.add_argument("--max-query", type=int, default=None,
                    help="Keep only responses whose __q index is below this, "
                         "thinning replicates while keeping every coordinate")
    ap.add_argument("--shard", default="0/1", help="i/N (default: 0/1)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Hard cap on judgments issued this run. A cost guard "
                         "for paid backends: applied AFTER the resume filter, "
                         "so it bounds new spend, not total work.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Resume: skip (id, probe) pairs already judged")
    ap.add_argument("--reaggregate", action="store_true",
                    help="Merge existing shards and rewrite the summary; no "
                         "model is loaded and no judgment is issued")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the first few prompts and exit")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    tag = "qwen" if args.backend == "qwen" else "gemini"

    if args.reaggregate:
        reaggregate(args.work_dir, args.mode, tag)
        return

    relabel = {}
    if args.relabel:
        for pair in args.relabel.split(","):
            old, _, new = pair.partition("=")
            relabel[old.strip()] = new.strip()
        print(f"judge-label swaps: {relabel}")

    work = build_work(args.work_dir, args.mode, args.rubric, relabel,
                      args.only_group, args.levels, args.max_query)
    shard, n_shards = (lambda s: (int(s.partition("/")[0]),
                                  int(s.partition("/")[2] or 1)))(args.shard)
    work = work[shard::n_shards]

    print(f"\n{'='*72}")
    print(f"{args.work_dir.name} / mode={args.mode} / backend={args.backend}")
    print(f"  rubric             : {args.rubric}")
    print(f"  judgments in shard : {len(work)}")
    print(f"{'='*72}")

    if args.dry_run:
        for r, key, system, user in work[:4]:
            print(f"  {r['id']:<34} probe={key}")
            print(f"      {user}")
        if len(work) > 4:
            print(f"  ... {len(work) - 4} more")
        return
    if not work:
        raise SystemExit("nothing to judge")

    # dualfit is 1-5 (the fit rubric); axis and intensity are 1-7. The published
    # rubric is 1-5 in every mode it applies to.
    scale = (1, 5) if (args.mode == "dualfit" or args.rubric == "published") else (1, 7)

    sfx = output_suffix(args.only_group, relabel, args.rubric,
                        args.max_query, args.levels)
    out_dir = args.work_dir / "results"
    path = out_dir / f"single_{args.mode}_{tag}{sfx}.shard{shard:02d}of{n_shards:02d}.jsonl"

    judge = judge_backends.make_judge(args.backend, args.model_path)
    # Match every shard of this same mode/judge/variant, whatever shard count
    # produced it, so resume survives a change in worker availability.
    writer = judge_backends.JudgmentWriter(
        path, ("id", "probe"), args.skip_existing,
        sibling_glob=f"single_{args.mode}_{tag}{sfx}.shard*.jsonl")

    todo = [w for w in work if not writer.done({"id": w[0]["id"], "probe": w[1]})]
    if args.limit:
        print(f"  limit: issuing {min(args.limit, len(todo))} of {len(todo)} outstanding")
        todo = todo[:args.limit]

    try:
        for i, (r, key, system, user) in enumerate(todo):
            t0 = time.time()
            try:
                raw, usage = judge.judge(system, user, r["response_audio"])
            except Exception as e:
                print(f"  ERROR {r['id']} {key}: {e}")
                raw, usage = "", {"error": str(e)}
            rating, just = judge_backends.parse_rating_scale(raw, *scale)
            writer.write({
                "id": r["id"], "probe": key, "mode": args.mode,
                "point_id": r["point_id"], "arm": ARM,
                "group": r["group"], "anchor": r["anchor"],
                "valence": r["valence"], "arousal": r["arousal"],
                "meta": r.get("meta", {}),
                "ray_group": r.get("ray_group"),
                "query_audio": r["query_audio"],
                "probe_label": key.split(":", 1)[-1] if ":" in key else key,
                "rating": rating, "justification": just,
                "raw_response": raw, "judge": tag, "usage": usage,
            })
            print(f"  [{i+1}/{len(todo)}] {r['id']:<32} {key:<20} "
                  f"rating={rating} ({time.time() - t0:.1f}s)")
    finally:
        writer.close()

    if args.backend == "gemini":
        from eval.judge.gemini_backend import get_pricing
        pr = get_pricing(args.model_path or judge_backends.DEFAULT_GEMINI_MODEL) or {}
        tin = tout = 0
        for rec in writer.records:
            u = rec.get("usage") or {}
            tin += u.get("prompt_token_count") or 0
            tout += u.get("candidates_token_count") or 0
        cost = (tin * pr.get("audio_in", 0) + tout * pr.get("text_out", 0)) / 1e6
        n = sum(1 for r in writer.records if (r.get("usage") or {}))
        print(f"\n  Indicative spend: {tin} input + {tout} output tokens over "
              f"{n} calls  ->  ${cost:.4f}")

    summ = aggregate(writer.records, args.mode)
    (out_dir / f"single_{args.mode}_{tag}{sfx}"
               f".shard{shard:02d}of{n_shards:02d}_summary.json"
     ).write_text(json.dumps(summ, indent=2))
    print_summary(args.work_dir, summ)
    print(f"Next step:")
    print(f"  python -m eval.metrics.interpolation_eval \\")
    print(f"      --work-dir {args.work_dir} \\")
    print(f"      --design {'intensity' if args.mode != 'dualfit' else 'interpolation'} "
          f"--judge {tag}")


if __name__ == "__main__":
    main()
