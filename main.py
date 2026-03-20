"""
main.py
=======
DysVoice — Master Controller (Person 3)

Connects all three modules into one working pipeline:
    record_audio()  →  denoise_audio()  →  transcribe()  →  speak()  →  display_text()

Usage:
    python main.py                        # live mic mode
    python main.py --file audio.wav       # demo backup mode (file input)
"""

import argparse
import sys
import os
import config

# ── Imports ───────────────────────────────────────────────────────────────────
from audio.record import record_audio
from audio.denoise import denoise_audio
from inference.transcribe import transcribe
from output.speak import speak
from output.display import display_text


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(audio_array):
    """
    Run the full DysVoice pipeline on a numpy audio array.

    Steps:
        1. Denoise the raw audio
        2. Transcribe the cleaned audio to text
        3. Handle empty transcripts gracefully
        4. Display the transcript on screen
        5. Speak the transcript aloud

    Parameters
    ----------
    audio_array : np.ndarray
        Raw float32 audio at 16000 Hz, from record_audio() or librosa.load().

    Returns
    -------
    str
        The final transcript string (may be empty if transcription failed).
    """
    import numpy as np

    # Guard: skip if audio is too short (under 0.5 seconds)
    min_samples = int(0.5 * config.SAMPLE_RATE)
    if audio_array.size < min_samples:
        print("Audio too short — skipping this round.")
        return ""

    # Step 1: Noise reduction
    print("Processing audio...")
    clean = denoise_audio(audio_array)

    # Step 2: Transcription
    print("Transcribing...")
    text = transcribe(clean)

    # Step 3: Handle empty transcript
    if not text or not text.strip():
        fallback = "Could not understand, please repeat"
        print("Transcript: (empty) — speaking fallback message")
        display_text(fallback)
        speak(fallback)
        return ""

    # Step 4 and 5: Display and speak the result
    print(f"Transcript: {text}")
    display_text(text)
    print("Speaking output...")
    speak(text)

    return text


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DysVoice — AI-Based Dysarthric Speech Assistance System"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a .wav file for demo backup mode (skips the microphone)."
    )
    args = parser.parse_args()

    if args.file:
        # ── Demo backup mode: load a .wav file instead of recording ──────────
        import librosa
        print(f"\nDysVoice — File mode")
        print(f"Loading: {args.file}")

        if not os.path.isfile(args.file):
            print(f"Error: file not found — {args.file}")
            sys.exit(1)

        audio, _ = librosa.load(args.file, sr=config.SAMPLE_RATE, mono=True)
        print(f"Loaded {audio.size} samples ({audio.size / config.SAMPLE_RATE:.2f}s)\n")
        run_pipeline(audio)

    else:
        # ── Live mic mode: loop until Ctrl+C ─────────────────────────────────
        print("\nDysVoice ready.")
        print("Press Enter to start listening. Press Ctrl+C to quit.\n")

        while True:
            try:
                input()
                print("Listening...")
                audio = record_audio()
                run_pipeline(audio)
                print()

            except KeyboardInterrupt:
                print("\nDysVoice stopped. Goodbye.")
                sys.exit(0)


if __name__ == "__main__":
    main()