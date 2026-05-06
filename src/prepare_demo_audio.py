#!/usr/bin/env python3
"""
prepare_demo_audio.py
---------------------
Curates audio files and metadata for the Sympatheia GitHub Pages demo.
Run once (or re-run to refresh) to populate docs/audio/ and docs/img/.

Usage:
    python prepare_demo_audio.py
"""

import json
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Source paths
# ---------------------------------------------------------------------------
SYMPATHEIA_18K = Path("/engram/naplab/users/sd3705/Datasets/Sympatheia-18k")

NEUTRAL_EVAL = Path(
    "/engram/naplab/users/sd3705/emo_recog_2025s/eval"
    "/eval_neutral_sympatheia-12emo-v2-20260320-100225_ckpt2200"
)
EMOTIONAL_EVAL = Path(
    "/engram/naplab/users/sd3705/emo_recog_2025s/eval"
    "/eval_emotional_sympatheia-12emo-v2-20260320-100225_ckpt2200"
)

INTERP_DIR = Path(
    "/home/sd3705/emo_recog_2025s/sympatheia/experiments"
    "/sympatheia-12emo-v2-20260320-100225/checkpoint-2200/results_demo/neutral_19"
)
FIGURES_DIR = Path("/home/sd3705/emo_recog_2025s/sympatheia/figures")

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent  # sympatheia/
DOCS = REPO_ROOT / "docs"
AUDIO_OUT = DOCS / "audio"
IMG_OUT = DOCS / "img"

EMOTIONS = [
    "Angry", "Anxious", "Content", "Disgusted", "Excited", "Frustrated",
    "Happy", "Neutral", "Relaxed", "Sad", "Surprised", "Tired",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def copy(src, dst: Path | str) -> bool:
    if src is None:
        return False
    src, dst = Path(src), Path(dst)
    if not src.exists():
        print(f"  [MISSING] {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {src.name} -> {dst.relative_to(REPO_ROOT)}")
    return True


_PATH_REMAP = {
    "/engram/naplab/users/sd3705/emo_recog_2025s/sympatheia/": str(REPO_ROOT) + "/",
    "/share/naplab/users/sd3705/emo_recog_2025s/sympatheia/":  str(REPO_ROOT) + "/",
}

def _remap(v):
    """Fix absolute paths written on a different node's mount point."""
    if not isinstance(v, str):
        return v
    for old, new in _PATH_REMAP.items():
        if v.startswith(old):
            return new + v[len(old):]
    return v

def _remap_neutral_query(p):
    """p2v2_Neutral_XXXXX.wav -> Sympatheia-18k neutral query path."""
    p = Path(p)
    qid = p.name.replace("p2v2_", "")  # Neutral_XXXXX.wav
    return SYMPATHEIA_18K / "Neutral/audio/eval/query/neutral" / qid

def _remap_emotional_query(p):
    """new_Angry_XXXXX.wav -> Sympatheia-18k emotional query path."""
    p = Path(p)
    name = p.name[4:]                       # strip "new_" -> Angry_XXXXX.wav
    emo = name.split("_")[0].lower()
    qid = "Emotional_" + name              # Emotional_Angry_XXXXX.wav
    return SYMPATHEIA_18K / f"Emotional/audio/eval/{emo}_query" / qid

def load_manifest(path: Path) -> dict:
    entries = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d = {k: _remap(v) for k, v in d.items()}
            entries[d["id"]] = d
    return entries


# ---------------------------------------------------------------------------
# Neutral comparison — three examples per emotion
#   ex1=_00, ex2=_14, ex3=_43
# Models: Sympatheia (finetuned_va), GLM-4-Voice, Kimi Audio, Qwen3-Omni, OpenS2S
# ---------------------------------------------------------------------------
NEUTRAL_EX_INDICES = ["00", "14", "43"]

def process_neutral():
    print("\n=== Neutral eval ===")
    main  = load_manifest(NEUTRAL_EVAL / "manifest.jsonl")
    base  = load_manifest(NEUTRAL_EVAL / "manifest_base.jsonl")
    kimi  = load_manifest(NEUTRAL_EVAL / "manifest_kimiaudio.jsonl")
    qwen  = load_manifest(NEUTRAL_EVAL / "manifest_qwen3omni.jsonl")
    opens = load_manifest(NEUTRAL_EVAL / "manifest_opens2s.jsonl")

    records = []
    for emo in EMOTIONS:
        v = main.get(f"{emo.lower()}_02", {}).get("valence", 0)
        a = main.get(f"{emo.lower()}_02", {}).get("arousal", 0)
        ex_texts = {}
        for ex_num, idx in enumerate(NEUTRAL_EX_INDICES, 1):
            ex  = f"ex{ex_num}"
            eid = f"{emo.lower()}_{idx}"
            m   = main.get(eid, {})
            b   = base.get(eid, {})
            k   = kimi.get(eid, {})
            q   = qwen.get(eid, {})
            o   = opens.get(eid, {})
            if not m and not b:
                print(f"  [SKIP] {eid} not found")
                continue
            p = f"neutral/{emo.lower()}/{ex}"
            copy(_remap_neutral_query(b.get("query_audio") or m.get("query_audio", "")), AUDIO_OUT / p / "query.wav")
            copy(m.get("finetuned_va_response"), AUDIO_OUT / p / "sympatheia_v2.wav")
            copy(b.get("base_response"),         AUDIO_OUT / p / "base.wav")
            copy(k.get("kimiaudio_response"),    AUDIO_OUT / p / "kimi.wav")
            copy(q.get("qwen3omni_response"),    AUDIO_OUT / p / "qwen3omni.wav")
            copy(o.get("opens2s_response"),      AUDIO_OUT / p / "opens2s.wav")
            ex_texts[ex] = {
                "sympatheia_v2_text": m.get("finetuned_va_text", ""),
                "base_text":          b.get("base_text", ""),
                "kimi_text":          k.get("kimiaudio_text", ""),
                "qwen3omni_text":     q.get("qwen3omni_text", ""),
                "opens2s_text":       o.get("opens2s_text", ""),
            }
        if not ex_texts:
            continue
        records.append({"emotion": emo, "valence": v, "arousal": a, **ex_texts})
    return records


# ---------------------------------------------------------------------------
# Emotional comparison — three examples per emotion
# Models: Sympatheia (finetuned_na), GLM-4-Voice, Kimi Audio, Qwen3-Omni, OpenS2S
# ---------------------------------------------------------------------------
EMOTIONAL_EX_INDICES = {
    "Angry":     ["03", "10", "16"],
    "Anxious":   ["00", "09", "15"],
    "Content":   ["07", "09", "18"],
    "Disgusted": ["01", "10", "17"],
    "Excited":   ["00", "09", "15"],
    "Frustrated":["00", "04", "16"],
    "Happy":     ["04", "07", "13"],
    "Neutral":   ["01", "08", "15"],
    "Relaxed":   ["01", "06", "11"],
    "Sad":       ["00", "04", "12"],
    "Surprised": ["00", "08", "12"],
    "Tired":     ["00", "06", "13"],
}

def process_emotional():
    print("\n=== Emotional eval ===")
    main  = load_manifest(EMOTIONAL_EVAL / "manifest.jsonl")
    base  = load_manifest(EMOTIONAL_EVAL / "manifest_base.jsonl")
    kimi  = load_manifest(EMOTIONAL_EVAL / "manifest_kimiaudio.jsonl")
    qwen  = load_manifest(EMOTIONAL_EVAL / "manifest_qwen3omni.jsonl")
    opens = load_manifest(EMOTIONAL_EVAL / "manifest_opens2s.jsonl")

    records = []
    for emo in EMOTIONS:
        indices = EMOTIONAL_EX_INDICES.get(emo, ["00", "01", "02"])
        v = main.get(f"{emo.lower()}_00", {}).get("valence", 0)
        a = main.get(f"{emo.lower()}_00", {}).get("arousal", 0)
        ex_texts = {}
        for ex_num, idx in enumerate(indices, 1):
            ex  = f"ex{ex_num}"
            eid = f"{emo.lower()}_{idx}"
            m   = main.get(eid, {})
            b   = base.get(eid, {})
            k   = kimi.get(eid, {})
            q   = qwen.get(eid, {})
            o   = opens.get(eid, {})
            if not m and not b:
                print(f"  [SKIP] {eid} not found")
                continue
            p = f"emotional/{emo.lower()}/{ex}"
            copy(_remap_emotional_query(b.get("query_audio") or m.get("query_audio", "")), AUDIO_OUT / p / "query.wav")
            copy(m.get("finetuned_na_response"), AUDIO_OUT / p / "sympatheia_v2.wav")
            copy(b.get("base_response"),         AUDIO_OUT / p / "base.wav")
            copy(k.get("kimiaudio_response"),    AUDIO_OUT / p / "kimi.wav")
            copy(q.get("qwen3omni_response"),    AUDIO_OUT / p / "qwen3omni.wav")
            copy(o.get("opens2s_response"),      AUDIO_OUT / p / "opens2s.wav")
            ex_texts[ex] = {
                "sympatheia_v2_text": m.get("finetuned_na_text", ""),
                "base_text":          b.get("base_text", ""),
                "kimi_text":          k.get("kimiaudio_text", ""),
                "qwen3omni_text":     q.get("qwen3omni_text", ""),
                "opens2s_text":       o.get("opens2s_text", ""),
            }
        if not ex_texts:
            continue
        records.append({"emotion": emo, "valence": v, "arousal": a, **ex_texts})
    return records


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------
def process_interpolation():
    print("\n=== Interpolation ===")
    copy(INTERP_DIR / "input_audio.wav", AUDIO_OUT / "interpolation" / "input_audio.wav")

    happy_sad = [
        ("output_happy_v0.85_a0.35.wav",            "happy_sad/happy_100.wav",   "Happy (100%)",        0.85,  0.35),
        ("output_happy_75_sad_25_v0.45_a0.10.wav",  "happy_sad/blend_75_25.wav", "75% Happy / 25% Sad", 0.45,  0.10),
        ("output_happy_sad_mid_v0.05_a-0.15.wav",   "happy_sad/blend_50_50.wav", "50% / 50%",           0.05, -0.15),
        ("output_happy_25_sad_75_v-0.35_a-0.40.wav","happy_sad/blend_25_75.wav", "25% Happy / 75% Sad",-0.35, -0.40),
        ("output_sad_v-0.75_a-0.65.wav",            "happy_sad/sad_100.wav",     "Sad (100%)",         -0.75, -0.65),
    ]

    anxious_relaxed = [
        ("output_anxious_v-0.40_a0.65.wav",              "anxious_relaxed/anxious_100.wav",   "Anxious (100%)",              -0.40,  0.65),
        ("output_anxious_75_relaxed_25_v-0.24_a0.34.wav","anxious_relaxed/blend_75_25.wav",   "75% Anxious / 25% Relaxed",   -0.24,  0.34),
        ("output_anxious_relaxed_mid_v-0.08_a0.03.wav",  "anxious_relaxed/blend_50_50.wav",   "50% / 50%",                   -0.08,  0.03),
        ("output_anxious_25_relaxed_75_v0.09_a-0.29.wav","anxious_relaxed/blend_25_75.wav",   "25% Anxious / 75% Relaxed",    0.09, -0.29),
        ("output_relaxed_v0.25_a-0.60.wav",              "anxious_relaxed/relaxed_100.wav",   "Relaxed (100%)",               0.25, -0.60),
    ]

    records_happy_sad = []
    for src_name, dst_name, label, v, a in happy_sad:
        copy(INTERP_DIR / src_name, AUDIO_OUT / "interpolation" / dst_name)
        records_happy_sad.append({"file": dst_name, "label": label, "valence": v, "arousal": a})

    records_anxious_relaxed = []
    for src_name, dst_name, label, v, a in anxious_relaxed:
        copy(INTERP_DIR / src_name, AUDIO_OUT / "interpolation" / dst_name)
        records_anxious_relaxed.append({"file": dst_name, "label": label, "valence": v, "arousal": a})

    return {"happy_sad": records_happy_sad, "anxious_relaxed": records_anxious_relaxed}


# ---------------------------------------------------------------------------
# Dataset samples — all 12 emotions
# Emotional split: per-emotion query + response
# Neutral split:   shared query (p2v2_Neutral_00259.wav) + per-emotion response
# ---------------------------------------------------------------------------
# Neutral query index with all 12 emotion responses present in eval set
NEUTRAL_DATASET_QID = "Neutral_00259"

def process_dataset():
    print("\n=== Dataset samples (all 12 emotions) ===")

    # Per-emotion index override (0-based) for the emotional split
    EMO_DATASET_IDX = {
        "Anxious":   1,
        "Content":   16,
        "Disgusted": 1,
        "Happy":     11,
        "Neutral":   12,
        "Relaxed":   2,
        "Surprised": 1,
    }

    # Emotional split: selected entry per emotion from metadata
    emo_meta_path = SYMPATHEIA_18K / "Emotional/metadata/text_pairs_eval.jsonl"
    by_emotion = {}
    with open(emo_meta_path) as f:
        for line in f:
            d = json.loads(line)
            emo = d["user_emotion"]
            by_emotion.setdefault(emo, []).append(d)

    emotional_records = []
    for emo in EMOTIONS:
        entries = by_emotion.get(emo, [])
        idx = EMO_DATASET_IDX.get(emo, 0)
        d = entries[idx] if idx < len(entries) else (entries[0] if entries else None)
        if not d:
            print(f"  [SKIP emotional] {emo} not in metadata")
            continue
        qidx = d["query_index"]
        ridx  = d["response_index"]
        qsrc = SYMPATHEIA_18K / "Emotional/audio/eval" / f"{emo.lower()}_query"  / f"{qidx}.wav"
        rsrc = SYMPATHEIA_18K / "Emotional/audio/eval" / f"{emo.lower()}_response" / f"{ridx}.wav"
        pfx = f"dataset/emotional/{emo.lower()}"
        copy(qsrc, AUDIO_OUT / pfx / "query.wav")
        copy(rsrc, AUDIO_OUT / pfx / "response.wav")
        emotional_records.append({
            "emotion":       emo,
            "query_text":    d.get("query_text", ""),
            "response_text": d.get("response_text", ""),
        })

    # Neutral split: shared query + per-emotion responses (Neutral_00084 has all 12)
    neutral_meta_path = SYMPATHEIA_18K / "Neutral/metadata/text_pairs_eval.jsonl"
    neutral_by_emotion = {}
    with open(neutral_meta_path) as f:
        for line in f:
            d = json.loads(line)
            if d["query_index"] == NEUTRAL_DATASET_QID:
                neutral_by_emotion[d["user_emotion"]] = d

    query_src = SYMPATHEIA_18K / "Neutral/audio/eval/query/neutral" / f"{NEUTRAL_DATASET_QID}.wav"
    copy(query_src, AUDIO_OUT / "dataset/neutral/query.wav")

    neutral_records = []
    for emo in EMOTIONS:
        d = neutral_by_emotion.get(emo)
        if not d:
            print(f"  [SKIP neutral] {emo} not found for {NEUTRAL_DATASET_QID}")
            continue
        ridx = d["response_index"]  # e.g. "Neutral_00084_Angry"
        rsrc = SYMPATHEIA_18K / "Neutral/audio/eval/response" / emo.lower() / f"{ridx}.wav"
        copy(rsrc, AUDIO_OUT / f"dataset/neutral/{emo.lower()}/response.wav")
        neutral_records.append({
            "emotion":       emo,
            "query_text":    d.get("query_text", ""),
            "response_text": d.get("response_text", ""),
        })

    return {"emotional": emotional_records, "neutral": neutral_records}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def copy_figures():
    print("\n=== Figures ===")
    copy(FIGURES_DIR / "overview.png", IMG_OUT / "overview.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _load_existing_manifest():
    """Load sections from the existing manifest.json, if present."""
    out_path = AUDIO_OUT / "manifest.json"
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return {}

def _try(fn, section, existing):
    try:
        return fn()
    except Exception as e:
        print(f"  [WARNING] {section} failed ({e}); keeping existing records")
        return existing.get(section, [] if section != "interpolation" else {})

def main():
    existing  = _load_existing_manifest()
    neutral   = _try(process_neutral,       "neutral",   existing)
    emotional = _try(process_emotional,     "emotional", existing)
    interp    = _try(process_interpolation, "interpolation", existing)
    dataset   = process_dataset()
    copy_figures()

    manifest = {
        "neutral":       neutral,
        "emotional":     emotional,
        "interpolation": interp,
        "dataset":       dataset,
    }

    out_path = AUDIO_OUT / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {out_path}")
    n_neutral = len(neutral) if isinstance(neutral, list) else 0
    n_emotional = len(emotional) if isinstance(emotional, list) else 0
    print(f"Total neutral records:   {n_neutral}")
    print(f"Total emotional records: {n_emotional}")
    print(f"Dataset samples:         emotional={len(dataset['emotional'])}, neutral={len(dataset['neutral'])}")
    print("\nDone!")


if __name__ == "__main__":
    main()
