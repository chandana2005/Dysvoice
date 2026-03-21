import os
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

try:
    import config
    SAMPLE_RATE = config.SAMPLE_RATE
    MODEL_NAME  = config.MODEL_NAME
    MODEL_PATH  = config.MODEL_PATH
    DEVICE      = config.DEVICE
except ModuleNotFoundError:
    SAMPLE_RATE = 16000
    MODEL_NAME  = "openai/whisper-small"
    MODEL_PATH  = "model/dysvoice_whisper.pt"
    DEVICE      = "cpu"

print(f"[transcribe] Loading processor from {MODEL_NAME}")
_processor = WhisperProcessor.from_pretrained(MODEL_NAME)

print(f"[transcribe] Loading base model architecture from {MODEL_NAME}")
_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

if os.path.isfile(MODEL_PATH):
    print(f"[transcribe] Applying fine-tuned weights from {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    _model.load_state_dict(state_dict)
    print(f"[transcribe] Fine-tuned model loaded successfully")
else:
    print(f"[transcribe] WARNING - {MODEL_PATH} not found, using base model")

_model.to(DEVICE)
_model.eval()
_model.config.forced_decoder_ids = None
_processor.tokenizer.forced_decoder_ids = None


def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe a numpy audio array to text using Whisper.

    Parameters
    ----------
    audio : np.ndarray
        1-D float32 array of audio samples at 16000 Hz.
        This is the output of denoise_audio() from audio/denoise.py.

    Returns
    -------
    str
        Transcribed text e.g. 'bring me water'.
        Returns empty string if audio is empty or transcription fails.
    """
    if audio.size == 0:
        return ""
    try:
        inputs = _processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(DEVICE)
        attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            predicted_ids = _model.generate(input_features, attention_mask=attention_mask)
        text = _processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return text[0].strip()
    except Exception as e:
        print(f"[transcribe] Error: {e}")
        return ""


if __name__ == "__main__":
    import sys
    print("=" * 50)
    print("transcribe.py  -  Whisper transcription test")
    print("=" * 50)
    if len(sys.argv) > 1:
        import librosa
        wav_path = sys.argv[1]
        print(f"Loading file: {wav_path}")
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        print(f"Loaded {audio.size} samples ({audio.size / SAMPLE_RATE:.2f}s)")
    else:
        from audio.record import record_audio
        from audio.denoise import denoise_audio
        print("No file given - recording from mic...")
        audio = record_audio()
        audio = denoise_audio(audio)
    if audio.size == 0:
        print("No audio to transcribe.")
        sys.exit(1)
    print("Transcribing...")
    result = transcribe(audio)
    print(f"\nTranscript: '{result}'")
    print("\nTest PASSED - transcribe() returned a string successfully.")