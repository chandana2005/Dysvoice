"""
inference/transcribe.py
=======================
DysVoice - Audio Pipeline (Person 2)

Takes a cleaned numpy audio array from denoise.py and returns
transcribed text using Whisper.

On Day 4: uses base Whisper model as placeholder.
On Day 5: automatically switches to fine-tuned model when Person 1
          pushes model/dysvoice_whisper.pt to GitHub.

Public API (called by Person 3's main.py):
    text = transcribe(audio_array)
"""

import os
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────
try:
    import config
    SAMPLE_RATE = config.SAMPLE_RATE    # 16000
    MODEL_NAME  = config.MODEL_NAME     # 'openai/whisper-small'
    MODEL_PATH  = config.MODEL_PATH     # 'model/dysvoice_whisper.pt'
except ModuleNotFoundError:
    SAMPLE_RATE = 16000
    MODEL_NAME  = "openai/whisper-small"
    MODEL_PATH  = "model/dysvoice_whisper.pt"

# ── Load model ────────────────────────────────────────────────────────────────
# If Person 1 has pushed the fine-tuned model, use it.
# Otherwise fall back to the base Whisper model from HuggingFace.
if os.path.exists(MODEL_PATH):
    print(f"[transcribe] Loading fine-tuned model from {MODEL_PATH}")
    _model_source = MODEL_PATH
else:
    print(f"[transcribe] Fine-tuned model not found — using base {MODEL_NAME}")
    _model_source = MODEL_NAME

_processor = WhisperProcessor.from_pretrained(MODEL_NAME)
_model     = WhisperForConditionalGeneration.from_pretrained(_model_source)
_model.eval()


def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe a numpy audio array to a text string using Whisper.

    Parameters
    ----------
    audio : np.ndarray
        1-D float32 array of audio samples at 16000 Hz.
        This is the output of denoise_audio() from audio/denoise.py.

    Returns
    -------
    str
        Transcribed text e.g. 'bring me water'.
        Returns empty string '' if audio is empty or transcription fails.
    """
    if audio.size == 0:
        return ""

    try:
        # Convert numpy array to Whisper input format
        inputs = _processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        # Run through Whisper — no gradient needed for inference
        with torch.no_grad():
            predicted_ids = _model.generate(inputs.input_features)

        # Decode token IDs back to readable text
        text = _processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return text[0].strip()

    except Exception as e:
        print(f"[transcribe] Error during transcription: {e}")
        return ""


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("=" * 50)
    print("transcribe.py  —  Whisper transcription test")
    print("=" * 50)

    # If you pass a .wav file as argument, use that
    # Otherwise record live from mic
    if len(sys.argv) > 1:
        import librosa
        wav_path = sys.argv[1]
        print(f"Loading file: {wav_path}")
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        print(f"Loaded {audio.size} samples ({audio.size / SAMPLE_RATE:.2f}s)")
    else:
        from audio.record import record_audio
        from audio.denoise import denoise_audio
        print("No file given — recording from mic...")
        audio = record_audio()
        audio = denoise_audio(audio)

    if audio.size == 0:
        print("No audio to transcribe.")
        sys.exit(1)

    print("Transcribing... (first run downloads Whisper, may take 1-2 mins)")
    result = transcribe(audio)

    print(f"\nTranscript: '{result}'")
    print("\nTest PASSED — transcribe() returned a string successfully.")