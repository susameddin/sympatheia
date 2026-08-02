#!/usr/bin/env python3
"""
Generate responses for a VA interpolation point set.

Takes a point set built by va_pointsets.py and, for every point, generates a
spoken response to each of a fixed sample of emotionally NEUTRAL query wavs. The
query audio is held constant across every point, so any difference between
responses must come from the VA conditioning coordinate rather than from the
query's own content or affect.

Outputs, under --work-dir:
  audio/continuous/{point_id}__q{i}.wav          — one response per (point, query)
  manifests/continuous.shard{ii}of{NN}.jsonl     — one record per response

Sharding is deterministic: every shard builds the identical work list and takes
work[shard::n_shards], so shards never collide and their union is exact. Run one
shard per GPU with CUDA_VISIBLE_DEVICES.

Usage (run from src/):
    python -m eval.generate_responses.interpolation.generate_responses_interpolation \\
        --points /path/to/interp/intensity/points.json \\
        --checkpoint /path/to/checkpoint \\
        --query-dir /path/to/Sympatheia_12Emo_Neutral_v2/audio/eval/query/neutral \\
        --work-dir /path/to/interp/intensity \\
        --num-queries 12 \\
        --skip-existing

    # Four GPUs, one shard each:
    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$i python -m eval.generate_responses.interpolation.\\
generate_responses_interpolation ... --shard $i/4 &
    done
"""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import soundfile as sf

# parents[3] is src/, parents[4] is the repo root. Both go on sys.path: the
# repo root is what `from src.vocoder_src ...` resolves against, and src/ is
# what sibling packages are found under, regardless of cwd.
_HERE = Path(__file__).resolve()
SRC_DIR = _HERE.parents[3]
REPO_ROOT = _HERE.parents[4]
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT))

from src.vocoder_src import GLM4CodecEncoder, GLM4CodecDecoder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "THUDM/glm-4-voice-9b"
DECODER_SAMPLE_RATE = 22050
ARM = "continuous"

# Matches the decoding settings used by the other generate_responses scripts.
GEN_KWARGS = dict(temperature=0.2, top_p=0.8, max_new_tokens=2000)

DEFAULT_SEED = 42
DEFAULT_NUM_QUERIES = 12

# The first --query-base queries are pinned across runs; see sample_queries.
DEFAULT_QUERY_BASE = 6


def va_system_prompt(valence: float, arousal: float) -> str:
    return (f"Please respond in English. "
            f"User emotion (valence={valence:.2f}, arousal={arousal:.2f})")


def build_prompt(user_tokens: str, system_prompt: str) -> str:
    return f"<|system|>\n{system_prompt}\n<|user|>\n{user_tokens}\n<|assistant|>\n"


# ---------------------------------------------------------------------------
# Work list
# ---------------------------------------------------------------------------

def sample_queries(query_dir: Path, n: int, seed: int = DEFAULT_SEED,
                   base_n: int = None) -> list:
    """Sample n neutral query wavs, deterministically.

    ``base_n`` makes the list *extensible*, which matters because sample ids
    carry the query's index as ``__q{i}``. ``rng.sample(wavs, 12)`` is not a
    superset of ``rng.sample(wavs, 6)``, so a plain redraw renumbers every id
    and orphans every response already on disk. With ``base_n=6`` the first six
    entries are exactly what a 6-query run produced, in the same order, and the
    extra queries are appended after them — so raising --num-queries only ever
    adds work.
    """
    rng = random.Random(seed)
    wavs = sorted(Path(query_dir).glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"no .wav files in {query_dir}")
    if base_n is None or n <= base_n:
        return sorted(rng.sample(wavs, min(n, len(wavs))))

    base = sorted(rng.sample(wavs, min(base_n, len(wavs))))
    rest = [w for w in wavs if w not in set(base)]
    # A separate stream, so drawing the extension cannot perturb the base draw.
    rng2 = random.Random(seed + 10_000)
    return base + sorted(rng2.sample(rest, min(n - len(base), len(rest))))


def build_work(points: list, queries: list, only_group: str = None) -> list:
    """[(sample_id, point, query_path)] — identical in every shard.

    ``only_group`` filters AFTER the work list is built, so staging a run in
    parts never changes the ids or the query assignment of the parts that were
    already generated.
    """
    if only_group:
        prefixes = tuple(g.strip() for g in only_group.split(",") if g.strip())
        points = [p for p in points if p["group"].startswith(prefixes)]
        if not points:
            raise SystemExit(f"no points with group prefix(es) {prefixes!r}")

    work = []
    for p in points:
        for qi, wav in enumerate(queries):
            work.append((f"{p['point_id']}__q{qi}", p, wav))
    return work


def parse_shard(s: str):
    shard, _, n = s.partition("/")
    if not n:
        return 0, 1
    return int(shard), int(n)


def manifest_path(work_dir: Path, shard: int, n_shards: int) -> Path:
    return (work_dir / "manifests"
            / f"{ARM}.shard{shard:02d}of{n_shards:02d}.jsonl")


def make_record(sid, point, wav, system_prompt, text, response_audio):
    return {
        "id": sid,
        "point_id": point["point_id"],
        "arm": ARM,
        "emotion": point["emotion"],
        "valence": point["valence"],
        "arousal": point["arousal"],
        "anchor": point["anchor"],
        "anchor_valence": point["anchor_valence"],
        "anchor_arousal": point["anchor_arousal"],
        "dist_to_anchor": point["dist_to_anchor"],
        "group": point["group"],
        "meta": point.get("meta", {}),
        "system_prompt": system_prompt,
        "query_audio": str(Path(wav).resolve()),
        "text": text,
        "response_audio": response_audio,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate responses for a VA interpolation point set")
    p.add_argument("--points", type=Path, required=True,
                   help="Point-set JSON written by va_pointsets.py")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="LoRA checkpoint directory to generate with")
    p.add_argument("--query-dir", type=Path, required=True,
                   help="Directory of emotionally neutral query .wav files")
    p.add_argument("--work-dir", type=Path, required=True,
                   help="Output root; audio/ and manifests/ are created inside")
    p.add_argument("--num-queries", type=int, default=DEFAULT_NUM_QUERIES,
                   help=f"Queries per point (default: {DEFAULT_NUM_QUERIES})")
    p.add_argument("--query-base", type=int, default=DEFAULT_QUERY_BASE,
                   help="Pin the first N queries so raising --num-queries stays "
                        f"additive (default: {DEFAULT_QUERY_BASE})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Query sampling seed (default: {DEFAULT_SEED})")
    p.add_argument("--shard", default="0/1", help="i/N (default: 0/1)")
    p.add_argument("--only-group", default=None,
                   help="Restrict to points whose group starts with this prefix "
                        "(comma-separated list allowed), e.g. 'ray:Happy'")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of generations (smoke test)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip points whose output wav already exists (resume)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the work list and exit (no model load)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_stack(checkpoint: Path):
    import torch
    from transformers import AutoTokenizer
    from peft import AutoPeftModelForCausalLM

    print("Loading tokenizer and speech codec components...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    audio_0_id = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    encoder = GLM4CodecEncoder()
    decoder = GLM4CodecDecoder(str(SRC_DIR / "glm-4-voice-decoder"))

    print(f"Loading model: {checkpoint}")
    t0 = time.time()
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(checkpoint), device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s (device={model.device})")
    return tokenizer, encoder, decoder, model, audio_0_id, torch


def generate_one(prompt, model, tokenizer, decoder, audio_0_id, torch):
    """Run generation. Returns (text_output: str, waveform: np.ndarray | None)."""
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**model_inputs, **GEN_KWARGS)
    generated = outputs[0][model_inputs["input_ids"].shape[1]:]

    audio_toks, text_toks = [], []
    for tok in generated:
        (audio_toks if tok.item() >= audio_0_id else text_toks).append(tok)

    text = tokenizer.decode(text_toks, skip_special_tokens=True)
    if not audio_toks:
        return text, None
    ids_shifted = torch.tensor([[t.item() - audio_0_id for t in audio_toks]],
                               dtype=torch.long)
    return text, decoder(ids_shifted).squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.points.exists():
        print(f"ERROR: point set not found: {args.points}\n"
              f"       build it with: python -m eval.generate_responses."
              f"interpolation.va_pointsets --design ... --out {args.points}",
              file=sys.stderr)
        sys.exit(1)
    if not args.checkpoint.exists() and not args.dry_run:
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    points = json.loads(args.points.read_text())
    queries = sample_queries(args.query_dir, args.num_queries,
                             seed=args.seed, base_n=args.query_base)
    shard, n_shards = parse_shard(args.shard)

    full_work = build_work(points, queries)[shard::n_shards]
    work = build_work(points, queries, args.only_group)[shard::n_shards]
    if args.limit:
        work = work[:args.limit]

    out_audio = args.work_dir / "audio" / ARM
    man_path = manifest_path(args.work_dir, shard, n_shards)

    if args.dry_run:
        print(f"{args.points.stem} shard {shard}/{n_shards}: {len(work)} items")
        for sid, p, wav in work[:10]:
            print(f"  {sid:<36} {p['group']:<22} {wav.name}")
        if len(work) > 10:
            print(f"  ... {len(work) - 10} more")
        return

    out_audio.mkdir(parents=True, exist_ok=True)
    man_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: keep records for anything already on disk.
    #
    # Read EVERY shard manifest for this arm, not just this shard's. Staging a
    # run in parts (--only-group) changes the work list, so an id generated by
    # shard 2 of a partial pass can land in shard 0 of the full pass. Reading
    # only our own file would lose that record's transcript: the wav exists, so
    # it is skipped, and the rewrite below would emit text=None over it.
    done = {}
    for prior in sorted(man_path.parent.glob(f"{ARM}.shard*.jsonl")):
        with open(prior) as f:
            for line in f:
                r = json.loads(line)
                # A record with a transcript always beats one without.
                if r.get("text") or r["id"] not in done:
                    done[r["id"]] = r

    todo = [w for w in work
            if not (args.skip_existing and (out_audio / f"{w[0]}.wav").exists())]

    print(f"\n{'='*64}")
    print(f"Point set     : {args.points}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Shard         : {shard + 1}/{n_shards}")
    print(f"Queries       : {len(queries)} (base {args.query_base}, seed {args.seed})")
    print(f"Work in shard : {len(work)}")
    print(f"To generate   : {len(todo)}")
    print(f"Audio dir     : {out_audio}")
    print(f"Manifest      : {man_path}")
    print(f"{'='*64}")

    if todo:
        tokenizer, encoder, decoder, model, audio_0_id, torch = load_stack(args.checkpoint)

        # Encode each distinct query once; every point reuses the same wavs.
        query_cache = {}

        for i, (sid, p, wav) in enumerate(todo):
            key = str(wav)
            if key not in query_cache:
                query_cache[key] = "".join(
                    f"<|audio_{x}|>" for x in encoder([key])[0])

            system_prompt = va_system_prompt(p["valence"], p["arousal"])
            prompt = build_prompt(query_cache[key], system_prompt)

            t0 = time.time()
            try:
                text, waveform = generate_one(
                    prompt, model, tokenizer, decoder, audio_0_id, torch)
            except Exception as e:
                print(f"  [{i+1}/{len(todo)}] {sid}  ERROR during generation: {e}")
                continue
            elapsed = time.time() - t0

            out_wav = out_audio / f"{sid}.wav"
            if waveform is not None:
                sf.write(str(out_wav), waveform, DECODER_SAMPLE_RATE)
            status = "OK" if waveform is not None else "NO_AUDIO"

            done[sid] = make_record(
                sid, p, wav, system_prompt, text,
                str(out_wav.resolve()) if waveform is not None else None)
            print(f"  [{i+1}/{len(todo)}] {sid:<34} {p['group']:<20} "
                  f"V={p['valence']:+.2f} A={p['arousal']:+.2f}  "
                  f"{status} ({elapsed:.1f}s)")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Rewrite the shard manifest over the UNFILTERED shard work list.
    #
    # Iterating the filtered `work` would be wrong whenever --only-group is in
    # play: the rewrite would emit only the filtered rows and silently drop
    # every other row the manifest previously held. Rewriting the full shard,
    # with `done` (which aggregates every shard file) supplying the rows this
    # pass did not touch, makes the staging orders interchangeable.
    records = []
    for sid, p, wav in full_work:
        rec = done.get(sid)
        if rec is None:
            out_wav = out_audio / f"{sid}.wav"
            rec = make_record(
                sid, p, wav, va_system_prompt(p["valence"], p["arousal"]), None,
                str(out_wav.resolve()) if out_wav.exists() else None)
        records.append(rec)

    with open(man_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_ok = sum(1 for r in records if r["response_audio"])
    print(f"\n{'='*64}")
    print(f"DONE")
    print(f"  With audio : {n_ok}/{len(records)}")
    print(f"  Manifest   : {man_path}")
    print(f"\nNext step:")
    print(f"  python -m eval.judge.judge_interpolation \\")
    print(f"      --work-dir {args.work_dir} \\")
    print(f"      --mode axis --backend qwen --skip-existing")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
