#!/usr/bin/env bash
# =============================================================================
# Sympatheia Part 2 Dataset Pipeline
# =============================================================================
# Generates a dataset where each query appears with 11 different response
# emotions, teaching the model to prioritize VA labels over audio content.
#
# USAGE:
#   # Step 0 — preview (inspect a few samples before committing):
#   bash dataset_creation/run_part2_pipeline.sh preview
#
#   # Step 1 — text generation only:
#   bash dataset_creation/run_part2_pipeline.sh text
#
#   # Step 2 — audio generation only (after inspecting text):
#   bash dataset_creation/run_part2_pipeline.sh audio
#
#   # Step 3 — GLM-4-Voice conversion:
#   bash dataset_creation/run_part2_pipeline.sh convert
#
#   # Step 4 — validation:
#   bash dataset_creation/run_part2_pipeline.sh validate
#
#   # Run all steps (text → audio → convert → validate):
#   bash dataset_creation/run_part2_pipeline.sh all
# =============================================================================

set -euo pipefail

# Force Python to flush stdout/stderr immediately (no buffering)
export PYTHONUNBUFFERED=1

# ── Configuration ─────────────────────────────────────────────────────────────
CONDA_LLM_TTS="qwen3-tts4"   # text generation + TTS audio
CONDA_CONVERT="glm4voice3"   # GLM-4-Voice encoding/decoding (needs hyperpyyaml)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DATASET_DIR="/engram/naplab/users/sd3705/Datasets/Sympatheia_11Emo_17k_Part2"
METADATA_DIR="$DATASET_DIR/metadata"
AUDIO_DIR="$DATASET_DIR/audio"

LLM_MODEL="Qwen/Qwen3-32B"
TTS_MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

NUM_GPUS=4          # adjust to available GPUs
TTS_BATCH_SIZE=16   # adjust per GPU VRAM
LLM_BATCH_SIZE=8    # adjust per GPU VRAM

STEP="${1:-all}"

# ── Helpers ───────────────────────────────────────────────────────────────────
run_llm_tts() {
    echo ""
    echo ">>> $*"
    conda run -n "$CONDA_LLM_TTS" --no-capture-output "$@"
}

run_convert() {
    echo ""
    echo ">>> $*"
    conda run -n "$CONDA_CONVERT" --no-capture-output "$@"
}

# ── Step 0: Preview ───────────────────────────────────────────────────────────
if [[ "$STEP" == "preview" || "$STEP" == "all" ]]; then
    echo "============================================================"
    echo "  STEP 0: Preview — 3 query groups × 11 response emotions"
    echo "============================================================"
    mkdir -p "$METADATA_DIR"
    run_llm_tts python -u "$SCRIPT_DIR/generate_part2_text_pairs.py" \
        --llm-model "$LLM_MODEL" \
        --output-dir "$METADATA_DIR" \
        --preview 3
    echo ""
    echo "Review the preview output above."
    echo "If it looks good, run: bash dataset_creation/run_part2_pipeline.sh text"
    if [[ "$STEP" == "preview" ]]; then
        exit 0
    fi
fi

# ── Step 1: Text generation ───────────────────────────────────────────────────
if [[ "$STEP" == "text" || "$STEP" == "all" ]]; then
    echo "============================================================"
    echo "  STEP 1: Text generation (1,500 queries × 11 responses)"
    echo "  Expected output: ~16,500 (query, response) pairs"
    echo "============================================================"
    mkdir -p "$METADATA_DIR"
    run_llm_tts python -u "$SCRIPT_DIR/generate_part2_text_pairs.py" \
        --llm-model "$LLM_MODEL" \
        --output-dir "$METADATA_DIR" \
        --batch-size "$LLM_BATCH_SIZE" \
        --resume

    echo ""
    echo "Text generation complete. Files saved:"
    echo "  $METADATA_DIR/sampled_train.jsonl"
    echo "  $METADATA_DIR/sampled_eval.jsonl"
    echo "  $METADATA_DIR/unique_queries_train.jsonl"
    echo "  $METADATA_DIR/unique_queries_eval.jsonl"
fi

# ── Step 2: Audio generation ─────────────────────────────────────────────────
if [[ "$STEP" == "audio" || "$STEP" == "all" ]]; then
    echo "============================================================"
    echo "  STEP 2: Audio generation"
    echo "  Pass 1: ~1,500 query WAVs (Ryan speaker)"
    echo "  Pass 2: ~16,500 response WAVs (Vivian speaker)"
    echo "============================================================"
    mkdir -p "$AUDIO_DIR"
    run_llm_tts python -u "$SCRIPT_DIR/generate_part2_audio_multigpu.py" \
        --metadata-dir "$METADATA_DIR" \
        --output-audio-dir "$AUDIO_DIR" \
        --model-name "$TTS_MODEL" \
        --num-gpus "$NUM_GPUS" \
        --batch-size "$TTS_BATCH_SIZE" \
        --resume

    echo ""
    echo "Audio generation complete. Files saved under: $AUDIO_DIR"
fi

# ── Step 3: GLM-4-Voice conversion ───────────────────────────────────────────
if [[ "$STEP" == "convert" || "$STEP" == "all" ]]; then
    echo "============================================================"
    echo "  STEP 3: Convert to GLM-4-Voice format"
    echo "  NOTE: VA values taken from RESPONSE emotion (not query)"
    echo "============================================================"
    run_convert python -u "$SCRIPT_DIR/convert_part2_to_glm4voice.py" \
        --metadata-dir "$METADATA_DIR" \
        --audio-dir    "$AUDIO_DIR" \
        --output-dir   "$DATASET_DIR"

    echo ""
    echo "Conversion complete."
    echo "  $DATASET_DIR/train.jsonl"
    echo "  $DATASET_DIR/eval.jsonl"
fi

# ── Step 4: Validation ────────────────────────────────────────────────────────
if [[ "$STEP" == "validate" || "$STEP" == "all" ]]; then
    echo "============================================================"
    echo "  STEP 4: Dataset validation"
    echo "============================================================"
    run_convert python -u "$SCRIPT_DIR/validate_part2_dataset.py" \
        --dataset-dir "$DATASET_DIR"

    echo ""
    echo "Validation report: $DATASET_DIR/validation_report_part2.json"
fi

echo ""
echo "Pipeline complete for step: $STEP"
