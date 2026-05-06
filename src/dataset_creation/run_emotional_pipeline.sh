#!/usr/bin/env bash
# =============================================================================
# Sympatheia Emotional Split Pipeline
# =============================================================================
# Both queries and responses have emotions. Each emotion has its own
# query/response audio pair.
#
# USAGE:
#   # Step 0 — preview (inspect a few samples before committing):
#   bash dataset_creation/run_emotional_pipeline.sh preview
#
#   # Step 1 — text generation only:
#   bash dataset_creation/run_emotional_pipeline.sh text
#
#   # Step 2 — audio generation only (after inspecting text):
#   bash dataset_creation/run_emotional_pipeline.sh audio
#
#   # Step 3 — GLM-4-Voice conversion:
#   bash dataset_creation/run_emotional_pipeline.sh convert
#
#   # Step 4 — create N/A variant (masks ~1/3 of VA values):
#   bash dataset_creation/run_emotional_pipeline.sh na
#
#   # Step 5 — validation:
#   bash dataset_creation/run_emotional_pipeline.sh validate
#
#   # Run all steps (text → audio → convert → na → validate):
#   bash dataset_creation/run_emotional_pipeline.sh all
# =============================================================================

set -euo pipefail

# Force Python to flush stdout/stderr immediately (no buffering)
export PYTHONUNBUFFERED=1

# ── Configuration ─────────────────────────────────────────────────────────────
CONDA_LLM_TTS="qwen3-tts4"   # text generation + TTS audio
CONDA_CONVERT="glm4voice3"   # GLM-4-Voice encoding/decoding (needs hyperpyyaml)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

DATASET_DIR="/path/to/Sympatheia-18k/Emotional"
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
    echo "  STEP 0: Preview — 2 samples per emotion"
    echo "============================================================"
    mkdir -p "$METADATA_DIR"
    run_llm_tts python -u "$SCRIPT_DIR/emotional/generate_emotional_text_pairs.py" \
        --llm-model "$LLM_MODEL" \
        --output-dir "$METADATA_DIR" \
        --preview 2
    echo ""
    echo "Review the preview output above."
    echo "If it looks good, run: bash dataset_creation/run_emotional_pipeline.sh text"
    if [[ "$STEP" == "preview" ]]; then
        exit 0
    fi
fi

# ── Step 1: Text generation ───────────────────────────────────────────────────
if [[ "$STEP" == "text" || "$STEP" == "all" ]]; then
    echo "============================================================"
    echo "  STEP 1: Text generation (12 emotions × ~700 samples each)"
    echo "  Expected output: ~8,400 (query, response) pairs"
    echo "============================================================"
    mkdir -p "$METADATA_DIR"
    run_llm_tts python -u "$SCRIPT_DIR/emotional/generate_emotional_text_pairs.py" \
        --llm-model "$LLM_MODEL" \
        --output-dir "$METADATA_DIR" \
        --batch-size "$LLM_BATCH_SIZE" \
        --resume

    echo ""
    echo "Text generation complete. Files saved:"
    echo "  $METADATA_DIR/text_pairs_train.jsonl"
    echo "  $METADATA_DIR/text_pairs_eval.jsonl"
fi

# ── Step 2: Audio generation ─────────────────────────────────────────────────
if [[ "$STEP" == "audio" || "$STEP" == "all" ]]; then
    echo "============================================================"
    echo "  STEP 2: Audio generation"
    echo "  Query + response WAVs for each emotion (Ryan + Vivian speakers)"
    echo "============================================================"
    mkdir -p "$AUDIO_DIR"
    run_llm_tts python -u "$SCRIPT_DIR/emotional/generate_emotional_audio_multigpu.py" \
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
    echo "  NOTE: VA values from QUERY emotion"
    echo "============================================================"
    run_convert python -u "$SCRIPT_DIR/emotional/convert_emotional_to_glm4voice.py" \
        --metadata-dir "$METADATA_DIR" \
        --audio-dir    "$AUDIO_DIR" \
        --output-dir   "$DATASET_DIR"

    echo ""
    echo "Conversion complete."
    echo "  $DATASET_DIR/encoded_train.jsonl"
    echo "  $DATASET_DIR/encoded_eval.jsonl"
fi

# ── Step 4: N/A variant ───────────────────────────────────────────────────────
if [[ "$STEP" == "na" || "$STEP" == "all" ]]; then
    echo "============================================================"
    echo "  STEP 4: Create N/A variant (masks ~1/3 of VA values)"
    echo "============================================================"
    run_convert python -u "$SCRIPT_DIR/emotional/create_na_variant.py" \
        --input-dir "$DATASET_DIR"

    echo ""
    echo "N/A variant complete."
    echo "  $DATASET_DIR/encoded_train_na.jsonl"
    echo "  $DATASET_DIR/encoded_eval_na.jsonl"
fi

# ── Step 5: Validation ────────────────────────────────────────────────────────
if [[ "$STEP" == "validate" || "$STEP" == "all" ]]; then
    echo "============================================================"
    echo "  STEP 5: Dataset validation"
    echo "============================================================"
    run_convert python -u "$SCRIPT_DIR/emotional/validate_emotional.py" \
        --dataset-dir "$DATASET_DIR"

    echo ""
    echo "Validation report: $DATASET_DIR/validation_report_emotional.json"
fi

echo ""
echo "Pipeline complete for step: $STEP"
