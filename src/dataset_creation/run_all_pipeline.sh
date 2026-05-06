#!/usr/bin/env bash
# =============================================================================
# Sympatheia Full Dataset Pipeline
# =============================================================================
# Generates both Emotional and Neutral splits, then merges them into the
# final Sympatheia-18k dataset ready for GLM-4-Voice fine-tuning.
#
# USAGE:
#   bash dataset_creation/run_all_pipeline.sh
#
# What this does:
#   1. Generate Emotional split (text → audio → tokens → NA variant → validate)
#   2. Generate Neutral split   (text → audio → tokens → validate)
#   3. Merge both into Sympatheia-18k/encoded_train.jsonl + encoded_eval.jsonl
#
# Prerequisites:
#   - conda envs: qwen3-tts4 (text gen + TTS), glm4voice3 (codec encoding)
#   - LLM model: Qwen/Qwen3-32B
#   - TTS model: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (downloaded by HuggingFace)
# =============================================================================

set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_CONVERT="glm4voice3"

BASE_DIR="/path/to/Sympatheia-18k"

echo "================================================================"
echo "  Sympatheia Full Dataset Pipeline"
echo "  Output: $BASE_DIR"
echo "================================================================"

# ── Step 1: Emotional split ───────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  PHASE 1: Emotional Split"
echo "================================================================"
bash "$SCRIPT_DIR/run_emotional_pipeline.sh" all

# ── Step 2: Neutral split ─────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  PHASE 2: Neutral Split"
echo "================================================================"
bash "$SCRIPT_DIR/run_neutral_pipeline.sh" all

# ── Step 3: Merge ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  PHASE 3: Merge → Sympatheia-18k"
echo "================================================================"
conda run -n "$CONDA_CONVERT" --no-capture-output \
    python -u "$SCRIPT_DIR/merge_splits.py" \
    --emotional-dir "$BASE_DIR/Emotional" \
    --neutral-dir   "$BASE_DIR/Neutral" \
    --output-dir    "$BASE_DIR"

echo ""
echo "================================================================"
echo "  Done! Final dataset:"
echo "    $BASE_DIR/encoded_train.jsonl"
echo "    $BASE_DIR/encoded_eval.jsonl"
echo "================================================================"
