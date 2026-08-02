"""Intrinsic evaluation of TextToVAConverter on ISEAR dataset.

Evaluates how well the text-to-VA converter maps self-reported emotion
descriptions to the correct (valence, arousal) region.

Dataset: ISEAR — ~7,666 self-reported emotional situations (Scherer & Wallbott 1994).
We use 5 clean categories (joy, fear, anger, sadness, disgust); shame and guilt
are excluded because they lack well-matched speech model anchors.

Download ISEAR CSV (one-time):
    wget -O integration/cache/isear.csv \\
      https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/master/isear.csv

Usage:
    # Keyword-only (no GPU needed):
    python -m integration.eval_text_va \\
        --dataset-path integration/cache/isear.csv

    # HuggingFace 7-class classifier:
    python -m integration.eval_text_va \\
        --dataset-path integration/cache/isear.csv --method hf

    # HuggingFace zero-shot NLI (all 12 anchors):
    python -m integration.eval_text_va \\
        --dataset-path integration/cache/isear.csv --method hf_zs

    # With GLM-4 base model:
    python -m integration.eval_text_va \\
        --dataset-path integration/cache/isear.csv --llm

    # With fine-tuned LoRA checkpoint:
    python -m integration.eval_text_va \\
        --dataset-path integration/cache/isear.csv --llm \\
        --checkpoint experiments/sympatheia-12emo-YYYYMMDD-HHMMSS/checkpoint-N
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from src.integration.text_module.text_to_va import TextToVAConverter
from src.constants import SPEECH_ANCHORS

# ---------------------------------------------------------------------------
# ISEAR → speech anchor mapping (5 clean categories only)
# ---------------------------------------------------------------------------

ISEAR_TO_ANCHOR = {
    "joy":     "happy",
    "fear":    "anxious",
    "anger":   "angry",
    "sadness": "sad",
    "disgust": "disgusted",
}

ISEAR_TO_VA = {
    isear: SPEECH_ANCHORS[anchor]
    for isear, anchor in ISEAR_TO_ANCHOR.items()
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nearest_anchor_name(v: float, a: float) -> str:
    best_dist, best_name = float("inf"), "neutral"
    for name, (av, aa) in SPEECH_ANCHORS.items():
        d = (v - av) ** 2 + (a - aa) ** 2
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name


def load_isear(path: str, n_per_class: int, seed: int) -> list:
    """Load ISEAR CSV and return samples for the 5 clean emotion classes.

    Handles multiple common ISEAR CSV formats (pipe, comma, or tab separated).
    Returns a list of dicts: {emotion, anchor, text, v_gt, a_gt}.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    # Try common separators; use on_bad_lines='skip' to handle rows with
    # inconsistent field counts (common in ISEAR where text may contain the sep)
    df = None
    for sep in ["|", ",", "\t"]:
        try:
            candidate = pd.read_csv(
                path, sep=sep, encoding="latin-1",
                on_bad_lines="skip", low_memory=False,
            )
            if len(candidate.columns) >= 2:
                df = candidate
                break
        except Exception:
            continue

    if df is None:
        raise ValueError(f"Could not parse ISEAR CSV at {path}")

    # Find emotion column: first column whose values overlap with our 5 emotions
    # (handles both string labels like "joy" and integer codes like 1..7)
    known = set(ISEAR_TO_ANCHOR.keys())
    emot_col = None
    for col in df.columns:
        col_vals = set(df[col].astype(str).str.lower().str.strip().unique())
        if len(col_vals & known) >= 3:
            emot_col = col
            break

    if emot_col is None:
        raise ValueError(
            f"No emotion column found in columns: {list(df.columns)}\n"
            "Expected a column whose values include: joy, fear, anger, sadness, disgust"
        )

    # Find text column: prefer known names (SIT before Field1), skip the emotion column
    text_col = None
    for candidate_name in ["SIT", "SIT1", "Field1", "situation", "text"]:
        if candidate_name in df.columns and candidate_name != emot_col:
            text_col = candidate_name
            break
    if text_col is None:
        for col in df.columns:
            if col == emot_col:
                continue
            if df[col].dtype == object:
                avg_len = df[col].dropna().astype(str).str.len().mean()
                if avg_len > 20:
                    text_col = col
                    break

    if text_col is None:
        raise ValueError(
            f"No text column found in columns: {list(df.columns)}\n"
            "Expected a column like SIT with situation descriptions."
        )

    print(f"ISEAR columns: emotion='{emot_col}', text='{text_col}'")

    rng = random.Random(seed)
    samples = []
    for emotion in ISEAR_TO_ANCHOR:
        mask = df[emot_col].astype(str).str.lower().str.strip() == emotion
        texts = (
            df[mask][text_col]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        texts = [t for t in texts if len(t) > 10]

        if not texts:
            print(f"  WARNING: no samples found for '{emotion}' — skipping")
            continue

        chosen = rng.sample(texts, min(n_per_class, len(texts)))
        v_gt, a_gt = ISEAR_TO_VA[emotion]
        for text in chosen:
            samples.append({
                "emotion": emotion,
                "anchor": ISEAR_TO_ANCHOR[emotion],
                "text": text,
                "v_gt": v_gt,
                "a_gt": a_gt,
            })
        print(f"  {emotion:<10}: {len(chosen):>4} sampled  (total in dataset: {mask.sum()})")

    return samples


def run_eval(samples: list, converter: TextToVAConverter, method: str = "auto") -> list:
    """Run converter on all samples; return samples augmented with predictions."""
    results = []
    for s in samples:
        v_pred, a_pred, info = converter.convert(s["text"], method=method)
        results.append({
            **s,
            "v_pred": v_pred,
            "a_pred": a_pred,
            "pred_anchor": nearest_anchor_name(v_pred, a_pred),
            "info": info,
        })
    return results


def compute_metrics(results: list) -> tuple:
    """Return (per_class dict, overall dict) with RMSE, Pearson, anchor accuracy."""
    per_class = {}
    for emotion in ISEAR_TO_ANCHOR:
        subset = [r for r in results if r["emotion"] == emotion]
        if not subset:
            continue

        v_pred = np.array([r["v_pred"] for r in subset])
        a_pred = np.array([r["a_pred"] for r in subset])
        v_gt = np.array([r["v_gt"] for r in subset])
        a_gt = np.array([r["a_gt"] for r in subset])
        n = len(subset)

        rmse_v = float(np.sqrt(np.mean((v_pred - v_gt) ** 2)))
        rmse_a = float(np.sqrt(np.mean((a_pred - a_gt) ** 2)))
        rmse_2d = float(np.sqrt(np.mean((v_pred - v_gt) ** 2 + (a_pred - a_gt) ** 2)))

        def _safe_corr(x, y):
            if len(x) < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
                return 0.0
            c = float(np.corrcoef(x, y)[0, 1])
            return 0.0 if np.isnan(c) else c

        corr_v = _safe_corr(v_pred, v_gt)
        corr_a = _safe_corr(a_pred, a_gt)

        expected_anchor = ISEAR_TO_ANCHOR[emotion]
        anchor_acc = float(np.mean([r["pred_anchor"] == expected_anchor for r in subset]))

        per_class[emotion] = {
            "n": n,
            "anchor": expected_anchor,
            "rmse_v": rmse_v,
            "rmse_a": rmse_a,
            "rmse_2d": rmse_2d,
            "corr_v": corr_v,
            "corr_a": corr_a,
            "anchor_acc": anchor_acc,
        }

    v_all = np.array([r["v_pred"] for r in results])
    a_all = np.array([r["a_pred"] for r in results])
    vg_all = np.array([r["v_gt"] for r in results])
    ag_all = np.array([r["a_gt"] for r in results])

    def _safe_corr_overall(x, y):
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return 0.0
        c = float(np.corrcoef(x, y)[0, 1])
        return 0.0 if np.isnan(c) else c

    overall = {
        "n": len(results),
        "rmse_v": float(np.sqrt(np.mean((v_all - vg_all) ** 2))),
        "rmse_a": float(np.sqrt(np.mean((a_all - ag_all) ** 2))),
        "rmse_2d": float(np.sqrt(np.mean((v_all - vg_all) ** 2 + (a_all - ag_all) ** 2))),
        "corr_v": _safe_corr_overall(v_all, vg_all),
        "corr_a": _safe_corr_overall(a_all, ag_all),
        "anchor_acc": float(np.mean([r["pred_anchor"] == r["anchor"] for r in results])),
    }

    return per_class, overall


def print_table(per_class: dict, overall: dict) -> None:
    cols = f"{'Emotion':<10}  {'Anchor':<12}  {'N':>5}  {'RMSE-V':>6}  {'RMSE-A':>6}  {'RMSE-2D':>7}  {'CorrV':>6}  {'CorrA':>6}  {'AnchorAcc':>9}"
    sep = "-" * len(cols)
    print(f"\n{cols}\n{sep}")
    for emo, m in per_class.items():
        print(
            f"{emo:<10}  {m['anchor']:<12}  {m['n']:>5}  "
            f"{m['rmse_v']:>6.3f}  {m['rmse_a']:>6.3f}  {m['rmse_2d']:>7.3f}  "
            f"{m['corr_v']:>6.3f}  {m['corr_a']:>6.3f}  {m['anchor_acc']:>9.1%}"
        )
    print(sep)
    o = overall
    print(
        f"{'OVERALL':<10}  {'':12}  {o['n']:>5}  "
        f"{o['rmse_v']:>6.3f}  {o['rmse_a']:>6.3f}  {o['rmse_2d']:>7.3f}  "
        f"{o['corr_v']:>6.3f}  {o['corr_a']:>6.3f}  {o['anchor_acc']:>9.1%}"
    )


def save_plot(results: list, output_dir: str) -> None:
    """Save VA scatter + anchor confusion plot."""
    emotions = list(ISEAR_TO_ANCHOR.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(emotions)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: predicted VA scatter, colored by true emotion
    for i, emotion in enumerate(emotions):
        subset = [r for r in results if r["emotion"] == emotion]
        if not subset:
            continue
        v = [r["v_pred"] for r in subset]
        a = [r["a_pred"] for r in subset]
        ax1.scatter(v, a, c=[colors[i]], label=emotion, alpha=0.35, s=12)

    # Mark ground-truth anchors with X
    for emotion in emotions:
        v_gt, a_gt = ISEAR_TO_VA[emotion]
        ax1.scatter([v_gt], [a_gt], c="black", marker="X", s=120, zorder=5)
        ax1.annotate(ISEAR_TO_ANCHOR[emotion], (v_gt, a_gt),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax1.set_xlabel("Valence")
    ax1.set_ylabel("Arousal")
    ax1.set_xlim(-1.15, 1.15)
    ax1.set_ylim(-1.15, 1.15)
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5)
    ax1.legend(fontsize=8, markerscale=2)
    ax1.set_title("Predicted VA by True Emotion  (✕ = ground-truth anchor)")

    # Right: confusion — true emotion vs predicted anchor
    all_anchors = sorted(SPEECH_ANCHORS.keys())
    anchor_to_idx = {a: i for i, a in enumerate(all_anchors)}
    cm = np.zeros((len(emotions), len(all_anchors)), dtype=int)
    for r in results:
        row = emotions.index(r["emotion"])
        col = anchor_to_idx.get(r["pred_anchor"], anchor_to_idx.get("neutral", 0))
        cm[row, col] += 1

    ax2.imshow(cm, interpolation="nearest", cmap="Blues")
    ax2.set_yticks(range(len(emotions)))
    ax2.set_yticklabels(emotions)
    ax2.set_xticks(range(len(all_anchors)))
    ax2.set_xticklabels(all_anchors, rotation=45, ha="right", fontsize=7)
    ax2.set_xlabel("Predicted anchor")
    ax2.set_ylabel("True emotion")
    ax2.set_title("Anchor Classification  (rows = true ISEAR emotion)")
    for i in range(len(emotions)):
        for j in range(len(all_anchors)):
            if cm[i, j] > 0:
                ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "text_va_eval.png")
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Intrinsic text-to-VA evaluation on ISEAR")
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "integration" / "cache" / "isear.csv"),
        help="Path to ISEAR CSV file",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("EVAL_OUTPUT_DIR", "./eval/eval_text_va"),
    )
    parser.add_argument(
        "--n-per-class", type=int, default=200,
        help="Max samples per emotion class (default: 200)",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "keyword", "llm", "hf", "hf_zs"],
        default="hf",
        help=(
            "VA extraction method: 'hf' (7-class HuggingFace classifier, default), "
            "'keyword' (no model), 'llm' (GLM-4), "
            "'hf_zs' (zero-shot NLI over all 12 anchors), "
            "'auto' (llm if available, else keyword)."
        ),
    )
    parser.add_argument(
        "--hf-model",
        default="j-hartmann/emotion-english-distilroberta-base",
        help="HuggingFace model ID for --method hf (default: j-hartmann/emotion-english-distilroberta-base)",
    )
    parser.add_argument(
        "--hf-zs-model",
        default="facebook/bart-large-mnli",
        help="HuggingFace NLI model ID for --method hf_zs (default: facebook/bart-large-mnli)",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Use GLM-4 LLM for VA extraction (requires GPU). "
             "Equivalent to --method llm; overridden by --method if both are set.",
    )
    parser.add_argument(
        "--glm-model-path", default="THUDM/glm-4-voice-9b",
        help="HuggingFace model ID or local path for GLM-4 base model / tokenizer",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help=(
            "Path to a fine-tuned LoRA checkpoint (AutoPeftModelForCausalLM). "
            "When provided together with --llm, the fine-tuned model is used for "
            "VA extraction instead of the base model. The tokenizer is still loaded "
            "from --glm-model-path."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ISEAR from {args.dataset_path}…")
    samples = load_isear(args.dataset_path, args.n_per_class, args.seed)
    if not samples:
        print("ERROR: no samples loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"Total: {len(samples)} samples\n")

    # Resolve method: --llm is legacy alias for --method llm, only applies when
    # --method was not explicitly set (i.e. still at its default of "hf")
    method = args.method
    if args.llm and method == "hf":
        method = "llm"

    if method in ("llm", "auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"Loading tokenizer from {args.glm_model_path}…")
        tokenizer = AutoTokenizer.from_pretrained(
            args.glm_model_path, trust_remote_code=True
        )
        if args.checkpoint:
            from peft import AutoPeftModelForCausalLM
            print(f"Loading fine-tuned model from {args.checkpoint}…")
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.checkpoint, device_map="auto", trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            method_name = "llm_classify_finetuned"
        else:
            print(f"Loading base GLM-4 model from {args.glm_model_path}…")
            model = AutoModelForCausalLM.from_pretrained(
                args.glm_model_path, device_map="auto", trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            method_name = "llm_classify" if method == "llm" else "auto"
        model.eval()
        converter = TextToVAConverter(glm_model=model, glm_tokenizer=tokenizer)

    elif method == "hf":
        print(f"Using HF text-classification model: {args.hf_model}")
        converter = TextToVAConverter(
            glm_model=None, glm_tokenizer=None,
            hf_model_name=args.hf_model,
        )
        # Trigger lazy load now so download/errors surface before the eval loop
        converter._hf_classify("warm up")
        method_name = "hf_" + args.hf_model.split("/")[-1]

    elif method == "hf_zs":
        print(f"Using HF zero-shot-classification model: {args.hf_zs_model}")
        converter = TextToVAConverter(
            glm_model=None, glm_tokenizer=None,
            hf_zs_model_name=args.hf_zs_model,
        )
        # Trigger lazy load now
        converter._hf_zs_classify("warm up")
        method_name = "hf_zs_" + args.hf_zs_model.split("/")[-1]

    else:  # keyword
        import logging
        logging.getLogger("text_to_va").setLevel(logging.ERROR)
        converter = TextToVAConverter(glm_model=None, glm_tokenizer=None)
        method_name = "keyword"

    print(f"Running {method_name} evaluation on {len(samples)} samples…")
    results = run_eval(samples, converter, method=method)

    per_class, overall = compute_metrics(results)
    print_table(per_class, overall)
    save_plot(results, args.output_dir)

    out = {
        "method": method_name,
        "dataset": args.dataset_path,
        "n_per_class": args.n_per_class,
        "seed": args.seed,
        "overall": overall,
        "per_class": per_class,
    }
    json_path = os.path.join(args.output_dir, f"eval_results_{method_name}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results JSON saved → {json_path}")


if __name__ == "__main__":
    main()
