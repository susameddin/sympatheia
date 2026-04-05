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
# Sympatheia v2 (sympatheia-12emo-v2-20260320-100225, ckpt 2200)
V2_NEUTRAL_EVAL = Path(
    "/engram/naplab/users/sd3705/emo_recog_2025s"
    "/eval_neutral_sympatheia-12emo-v2-20260320-100225_ckpt2200"
)
V2_EMOTIONAL_EVAL = Path(
    "/engram/naplab/users/sd3705/emo_recog_2025s"
    "/eval_emotional_sympatheia-12emo-v2-20260320-100225_ckpt2200"
)

# Sympatheia v1 (sympatheia-12emo-20260312-100309, ckpt 3000)
V1_NEUTRAL_EVAL = Path(
    "/engram/naplab/users/sd3705/emo_recog_2025s"
    "/eval_neutral_sympatheia-12emo-20260312-100309_ckpt3000"
)
V1_EMOTIONAL_EVAL = Path(
    "/engram/naplab/users/sd3705/emo_recog_2025s"
    "/eval_emotional_sympatheia-12emo-20260312-100309_ckpt3000"
)

INTERP_DIR = Path(
    "/home/sd3705/emo_recog_2025s/sympatheia/experiments"
    "/sympatheia-12emo-v2-20260320-100225/checkpoint-2200/results_demo/neutral_19"
)

# Sympatheia v2 demo results (neutral_19 query — p2v2_Neutral_00019.wav)
NEUTRAL_19_DIR = Path(
    "/home/sd3705/emo_recog_2025s/sympatheia/experiments"
    "/sympatheia-12emo-v2-20260320-100225/checkpoint-2200/results_demo/neutral_19"
)
# Other models generated for the same neutral_19 query
NEUTRAL_19_OTHER = NEUTRAL_19_DIR / "other_models"
FIGURES_DIR = Path("/home/sd3705/emo_recog_2025s/sympatheia/figures")

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
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

def load_manifest(path: Path) -> dict:
    entries = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            d = {k: _remap(v) for k, v in d.items()}
            entries[d["id"]] = d
    return entries


# ---------------------------------------------------------------------------
# Neutral comparison — two examples per emotion
#   Ex1: p2v2_Neutral_00019.wav (neutral_19, all models from other_models/)
#   Ex2: p2v2_Neutral_00025.wav (_00 from original full eval manifests)
# ---------------------------------------------------------------------------
def process_neutral():
    print("\n=== Neutral eval ===")
    # Ex1 sources — neutral_19
    oth   = load_manifest(NEUTRAL_19_OTHER / "manifest.jsonl")
    qwen1 = load_manifest(NEUTRAL_19_OTHER / "manifest_qwen3omni.jsonl")
    open1 = load_manifest(NEUTRAL_19_OTHER / "manifest_opens2s.jsonl")
    # Ex2 sources — original full eval (_00 = p2v2_Neutral_00025.wav)
    v2m   = load_manifest(V2_NEUTRAL_EVAL / "manifest.jsonl")
    v2q   = load_manifest(V2_NEUTRAL_EVAL / "manifest_qwen3omni.jsonl")
    v2o   = load_manifest(V2_NEUTRAL_EVAL / "manifest_opens2s.jsonl")
    v1m   = load_manifest(V1_NEUTRAL_EVAL / "manifest.jsonl")

    records = []
    for emo in EMOTIONS:
        eid = f"{emo.lower()}_00"
        m1  = oth.get(eid)
        q1  = qwen1.get(eid, {})
        o1  = open1.get(eid, {})
        m2  = v2m.get(eid, {})
        q2  = v2q.get(eid, {})
        o2  = v2o.get(eid, {})
        mv1 = v1m.get(eid, {})
        if not m1:
            print(f"  [SKIP] {eid} not in neutral_19 manifest")
            continue

        v, a = m1["valence"], m1["arousal"]
        base = f"neutral/{emo.lower()}"

        # Ex1: neutral_19
        p1 = f"{base}/ex1"
        copy(NEUTRAL_19_DIR / "input_audio.wav",                              AUDIO_OUT / p1 / "query.wav")
        copy(m1.get("base_response"),                                          AUDIO_OUT / p1 / "base.wav")
        copy(NEUTRAL_19_DIR / f"output_{emo.lower()}_v{v:.2f}_a{a:.2f}.wav",  AUDIO_OUT / p1 / "sympatheia_v2.wav")
        copy(m1.get("finetuned_va_response"),                                  AUDIO_OUT / p1 / "sympatheia_v1.wav")
        copy(q1.get("qwen3omni_response"),                                     AUDIO_OUT / p1 / "qwen3omni.wav")
        copy(o1.get("opens2s_response"),                                       AUDIO_OUT / p1 / "opens2s.wav")

        # Ex2: p2v2_Neutral_00025 (from original eval manifests)
        p2 = f"{base}/ex2"
        copy(m2.get("query_audio"),            AUDIO_OUT / p2 / "query.wav")
        copy(m2.get("base_response"),          AUDIO_OUT / p2 / "base.wav")
        copy(m2.get("finetuned_va_response"),  AUDIO_OUT / p2 / "sympatheia_v2.wav")
        copy(mv1.get("finetuned_va_response"), AUDIO_OUT / p2 / "sympatheia_v1.wav")
        copy(q2.get("qwen3omni_response"),     AUDIO_OUT / p2 / "qwen3omni.wav")
        copy(o2.get("opens2s_response"),       AUDIO_OUT / p2 / "opens2s.wav")

        records.append({
            "emotion": emo, "valence": v, "arousal": a,
            "ex1": {
                "base_text":          m1.get("base_text", ""),
                "sympatheia_v2_text": m1.get("finetuned_va_text", ""),
                "sympatheia_v1_text": m1.get("finetuned_va_text", ""),
                "qwen3omni_text":     q1.get("qwen3omni_text", ""),
                "opens2s_text":       o1.get("opens2s_text", ""),
            },
            "ex2": {
                "base_text":          m2.get("base_text", ""),
                "sympatheia_v2_text": m2.get("finetuned_va_text", ""),
                "sympatheia_v1_text": mv1.get("finetuned_va_text", ""),
                "qwen3omni_text":     q2.get("qwen3omni_text", ""),
                "opens2s_text":       o2.get("opens2s_text", ""),
            },
        })
    return records


# ---------------------------------------------------------------------------
# Emotional comparison — two examples per emotion
#   Ex1: _00 query (current)
#   Ex2: _01 query (random second sample)
# All models: audio only / no VA in system prompt (finetuned_na for Sympatheia)
# ---------------------------------------------------------------------------
def process_emotional():
    print("\n=== Emotional eval (no-VA / audio only) ===")
    v2m = load_manifest(V2_EMOTIONAL_EVAL / "manifest.jsonl")
    v2q = load_manifest(V2_EMOTIONAL_EVAL / "manifest_qwen3omni.jsonl")
    v2o = load_manifest(V2_EMOTIONAL_EVAL / "manifest_opens2s.jsonl")
    v1m = load_manifest(V1_EMOTIONAL_EVAL / "manifest.jsonl")

    records = []
    for emo in EMOTIONS:
        eid1 = f"{emo.lower()}_00"
        eid2 = f"{emo.lower()}_01"
        m2_1 = v2m.get(eid1)
        m2_2 = v2m.get(eid2, {})
        m1_1 = v1m.get(eid1, {})
        m1_2 = v1m.get(eid2, {})
        q1   = v2q.get(eid1, {})
        q2   = v2q.get(eid2, {})
        o1   = v2o.get(eid1, {})
        o2   = v2o.get(eid2, {})
        if not m2_1:
            print(f"  [SKIP] {eid1} not in v2 manifest")
            continue

        base = f"emotional/{emo.lower()}"

        # Ex1: _00
        p1 = f"{base}/ex1"
        copy(m2_1.get("query_audio"),           AUDIO_OUT / p1 / "query.wav")
        copy(m2_1.get("base_response"),         AUDIO_OUT / p1 / "base.wav")
        copy(m2_1.get("finetuned_na_response"), AUDIO_OUT / p1 / "sympatheia_v2.wav")
        copy(m1_1.get("finetuned_na_response"), AUDIO_OUT / p1 / "sympatheia_v1.wav")
        copy(q1.get("qwen3omni_response"),      AUDIO_OUT / p1 / "qwen3omni.wav")
        copy(o1.get("opens2s_response"),        AUDIO_OUT / p1 / "opens2s.wav")

        # Ex2: _01
        p2 = f"{base}/ex2"
        copy(m2_2.get("query_audio"),           AUDIO_OUT / p2 / "query.wav")
        copy(m2_2.get("base_response"),         AUDIO_OUT / p2 / "base.wav")
        copy(m2_2.get("finetuned_na_response"), AUDIO_OUT / p2 / "sympatheia_v2.wav")
        copy(m1_2.get("finetuned_na_response"), AUDIO_OUT / p2 / "sympatheia_v1.wav")
        copy(q2.get("qwen3omni_response"),      AUDIO_OUT / p2 / "qwen3omni.wav")
        copy(o2.get("opens2s_response"),        AUDIO_OUT / p2 / "opens2s.wav")

        records.append({
            "emotion": emo, "valence": m2_1["valence"], "arousal": m2_1["arousal"],
            "ex1": {
                "base_text":          m2_1.get("base_text", ""),
                "sympatheia_v2_text": m2_1.get("finetuned_na_text", ""),
                "sympatheia_v1_text": m1_1.get("finetuned_na_text", ""),
                "qwen3omni_text":     q1.get("qwen3omni_text", ""),
                "opens2s_text":       o1.get("opens2s_text", ""),
            },
            "ex2": {
                "base_text":          m2_2.get("base_text", ""),
                "sympatheia_v2_text": m2_2.get("finetuned_na_text", ""),
                "sympatheia_v1_text": m1_2.get("finetuned_na_text", ""),
                "qwen3omni_text":     q2.get("qwen3omni_text", ""),
                "opens2s_text":       o2.get("opens2s_text", ""),
            },
        })
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
NEUTRAL_DATASET_QUERY = Path(
    "/engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Neutral_v2"
    "/audio/eval/query/neutral/p2v2_Neutral_00259.wav"
)
NEUTRAL_DATASET_RESPONSE_DIR = Path(
    "/engram/naplab/users/sd3705/Datasets/Sympatheia_12Emo_Neutral_v2"
    "/audio/eval/response"
)
# v2 dataset: response filename suffix matches the emotion name exactly
NEUTRAL_RESPONSE_SUFFIX = {emo: emo for emo in EMOTIONS}

def process_dataset():
    print("\n=== Dataset samples (all 12 emotions) ===")
    v2_main = load_manifest(V2_NEUTRAL_EVAL / "manifest.jsonl")

    # Emotional split: unchanged (per-emotion query + response from eval manifest)
    emotional_records = []
    for emo in EMOTIONS:
        eid = f"{emo.lower()}_00"
        m = v2_main.get(eid)
        if not m:
            print(f"  [SKIP emotional] {eid}")
            continue
        pfx = f"dataset/emotional/{emo.lower()}"
        copy(m.get("query_audio"),           AUDIO_OUT / pfx / "query.wav")
        copy(m.get("finetuned_va_response"), AUDIO_OUT / pfx / "response.wav")
        emotional_records.append({
            "emotion": emo, "valence": m["valence"], "arousal": m["arousal"],
        })

    # Neutral split: shared query + per-emotion responses from dataset
    copy(NEUTRAL_DATASET_QUERY, AUDIO_OUT / "dataset/neutral/query.wav")
    neutral_records = []
    for emo in EMOTIONS:
        eid = f"{emo.lower()}_00"
        m = v2_main.get(eid, {})
        suffix = NEUTRAL_RESPONSE_SUFFIX.get(emo, emo)
        response_src = NEUTRAL_DATASET_RESPONSE_DIR / emo.lower() / f"p2v2_Neutral_00259_{suffix}.wav"
        copy(response_src, AUDIO_OUT / f"dataset/neutral/{emo.lower()}/response.wav")
        neutral_records.append({
            "emotion": emo, "valence": m.get("valence", 0), "arousal": m.get("arousal", 0),
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
def main():
    neutral   = process_neutral()
    emotional = process_emotional()
    interp    = process_interpolation()
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
    print(f"Total neutral records:   {len(neutral)}")
    print(f"Total emotional records: {len(emotional)}")
    print(f"Interpolation steps:     {len(interp)}")
    print(f"Dataset samples:         emotional={len(dataset['emotional'])}, neutral={len(dataset['neutral'])}")
    print("\nDone!")


if __name__ == "__main__":
    main()
