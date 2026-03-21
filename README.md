# Emotion-Aware Speech Model

Fine-tuning GLM-4-Voice with continuous Valence-Arousal (VA) conditioning for emotion-controlled speech-to-speech generation across 12 emotions, trained on the Sympatheia dataset.

## Emotions

Sad, Excited, Frustrated, Neutral, Happy, Angry, Anxious, Relaxed, Surprised, Disgusted, Tired, Content

Each emotion is mapped to a continuous (valence, arousal) coordinate, enabling interpolation between emotional states.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download decoder weights into `glm-4-voice-decoder/`:
   - `flow.pt` and `hift.pt` from the [GLM-4-Voice](https://huggingface.co/THUDM/glm-4-voice-decoder) HuggingFace model page.

3. The base model `THUDM/glm-4-voice-9b` is downloaded automatically from HuggingFace during training/inference.

## Usage

### Training
```bash
python train_sympatheia.py
```
Hyperparameters are in `config.yaml`. DeepSpeed config: `ds_config.json`.

### Inference
```bash
python inference_sympatheia.py \
    --experiment-dir experiments/<run-name> \
    --checkpoints 200 400 600
```

### Gradio Demo
```bash
python gradio_demo.py \
    --checkpoint experiments/<run-name>/checkpoint-600 \
    --port 7860
```

## Dataset Creation

The `dataset_creation/` directory contains the pipeline for creating the 12-emotion dataset:

### Part 1: Emotional queries + responses
1. `generate_new_text_pairs.py` — Generate emotion-conditioned text pairs with Qwen3-32B
2. `generate_qwen3tts_audio_12emo_multigpu.py` — Generate emotion-conditioned audio with Qwen3-TTS
3. `convert_qwen3tts_to_glm4voice_12emo.py` — Encode audio and create GLM-4-Voice VA format
4. `validate_dataset_12emo.py` — Validate the final dataset

### Part 2: Neutral queries × 12 response emotions
1. `generate_part2_text_pairs.py` — Generate neutral queries with 12 emotion response variants
2. `generate_part2_audio_multigpu.py` — Generate audio for Part 2
3. `convert_part2_to_glm4voice.py` — Convert Part 2 to GLM-4-Voice format
4. `validate_part2_dataset.py` — Validate Part 2 dataset
5. `run_part2_pipeline.sh` — Orchestration script for Part 2

### External dependency for dataset generation

`generate_qwen3tts_audio_12emo_multigpu.py` requires the [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) model package (`qwen_tts`). Install it separately or point the script's `sys.path` to your local clone.

## Emotion Recognition Modules

Multiple modalities decode (valence, arousal) from different input signals:

### Audio
Audio emotion recognition is handled natively by the GLM-4-Voice model — no separate module is needed. The model perceives emotional cues directly from the input speech signal.

### EEG (`eeg_emotion/`)
Within-subject MLP on Differential Entropy features from the DEAP dataset.
- **Model**: EEGDEModel — FC(160→256→128→64→2) with BN, ELU, Dropout
- **Accuracy**: ~70% binary (valence/arousal), 32 per-subject models
- **Train**: `python -m eeg_emotion.train --mode within_subject`
- **Inference**: `EEGVAPredictor().predict_va(eeg_array, subject_id=5)`

### Physiological signals (`physio_emotion/`)
6-stream 1D CNN with channel attention on DEAP (BVP, GSR, Resp, Temp, zEMG, tEMG).
- **Model**: PhysioMultiChannelModel — ~246K params, per-subject
- **Train**: `python -m physio_emotion.train --mode within_subject`
- **Inference**: `PhysioVAPredictor().predict_va(signals_dict, subject_id=5)`

### Face (`face_emotion/`)
ResNet18 fine-tuned on AffectNet_Balanced for 8-class emotion classification, mapped to VA via Russell's circumplex model.
- **Model**: FaceEmotionModel — ResNet18, 75.4% test accuracy (8 classes)
- **Emotions**: Anger, Contempt, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Train**: `python -m face_emotion.train`
- **Inference**: `FaceVAPredictor().predict_va(image)`

### Text (`text_to_va.py`)
Converts free-text emotion descriptions to (valence, arousal) using the GLM-4 LLM with keyword-centroid fallback.
- **Primary**: LLM prompt → structured JSON parse
- **Fallback**: Keyword-weighted centroid over 12 emotion anchors
- **Inference**: `TextToVAConverter(model, tokenizer).convert("I feel excited")`

## Project Structure

```
├── train_sympatheia.py                  # Training script (LoRA fine-tuning)
├── inference_sympatheia.py              # Batch inference
├── gradio_demo.py                       # Interactive demo
├── evaluate_model.py                    # Evaluation suite
├── compare_results.py                   # Compare base vs. fine-tuned metrics
├── text_to_va.py                        # Text → VA (GLM-4 LLM + keyword fallback)
├── constants.py                         # Shared 12-emotion VA mapping
├── config.yaml                          # LoRA training hyperparameters
├── ds_config.json                       # DeepSpeed config
├── eeg_emotion/                         # EEG → VA (DEAP, within-subject)
├── physio_emotion/                      # Physio → VA (DEAP, within-subject)
├── face_emotion/                        # Face → VA (AffectNet, ResNet18)
├── eval/                                # Baseline comparison & LLM judging
├── src/                                 # Model utilities & vocoder
├── speech_tokenizer/                    # Speech encoding/decoding
├── cosyvoice/                           # CosyVoice TTS components
├── glm-4-voice-decoder/                 # Decoder weights (flow.pt, hift.pt)
├── dataset_creation/                    # Sympatheia dataset creation pipeline
├── experiments/                         # Training checkpoints
└── results/                             # Evaluation output
```
