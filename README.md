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
Hyperparameters are in `config.yaml`. DeepSpeed config: `ds_config.json`.

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

## Emotion Recognition Modules

Multiple modalities decode (valence, arousal) from different input signals:

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
- **Fallback**: Keyword-weighted centroid over 11 emotion anchors
- **Inference**: `TextToVAConverter(model, tokenizer).convert("I feel excited")`

## Project Structure

```
├── train_opens2s_qwen3tts_va_text.py   # Training script
├── inference_opens2s_11emo_va_text.py   # Batch inference
├── gradio_demo.py                       # Interactive demo
├── config.yaml                          # Training hyperparameters
├── eeg_emotion/                         # EEG → VA (DEAP, within-subject)
├── physio_emotion/                      # Physio → VA (DEAP, within-subject)
├── face_emotion/                        # Face → VA (AffectNet, ResNet18)
├── text_to_va.py                        # Text → VA (GLM-4 LLM + keyword fallback)
├── src/                                 # Model utilities & vocoder
├── speech_tokenizer/                    # Speech encoding/decoding
├── cosyvoice/                           # CosyVoice TTS components
├── glm-4-voice-decoder/                 # Decoder config (weights downloaded separately)
└── dataset_creation/                    # Dataset creation pipeline
```
