# Emotion-Aware Speech Model

Fine-tuning GLM-4-Voice with continuous Valence-Arousal (VA) conditioning for emotion-controlled speech-to-speech generation across 11 emotions.

## Emotions

Sad, Excited, Frustrated, Neutral, Happy, Angry, Fear, Relaxed, Surprised, Disgusted, Tired

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
python train_opens2s_qwen3tts_va_text.py
```
Hyperparameters are in `config.yaml`. DeepSpeed configs: `ds_config.json` / `ds_config_zero2.json`.

### Inference
```bash
python inference_opens2s_11emo_va_text.py \
    --experiment-dir experiments/<run-name> \
    --checkpoints 100 300 500
```

### Gradio Demo
```bash
python gradio_demo.py \
    --checkpoint experiments/<run-name>/checkpoint-700 \
    --port 7860
```

## Dataset Creation (OpenS2S_11Emo)

The `dataset_creation/` directory contains the pipeline for creating the 11-emotion OpenS2S dataset:

1. `sample_emotion_subset_11emo.py` — Sample and split data across 11 emotions
2. `generate_qwen3tts_audio_11emo_multigpu.py` — Generate emotion-conditioned audio with Qwen3-TTS (requires `qwen_tts` package, see note below)
3. `convert_qwen3tts_to_glm4voice_11emo.py` — Encode audio and create GLM-4-Voice VA format
4. `validate_dataset_11emo.py` — Validate the final dataset
5. `fix_opens2s_dataset.py` — Fix trailing silence issues
6. `add_trailing_silence.py` — Add trailing silence to audio files

### External dependency for dataset generation

`generate_qwen3tts_audio_11emo_multigpu.py` requires the [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) model package (`qwen_tts`). Install it separately or point the script's `sys.path` to your local clone.

## Project Structure

```
├── train_opens2s_qwen3tts_va_text.py   # Training script
├── inference_opens2s_11emo_va_text.py   # Batch inference
├── gradio_demo.py                       # Interactive demo
├── config.yaml                          # Training hyperparameters
├── src/                                 # Model utilities & vocoder
├── speech_tokenizer/                    # Speech encoding/decoding
├── cosyvoice/                           # CosyVoice TTS components
├── glm-4-voice-decoder/                 # Decoder config (weights downloaded separately)
└── dataset_creation/                    # Dataset creation pipeline
```
