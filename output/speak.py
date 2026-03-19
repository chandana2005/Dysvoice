"""
speak.py - Text-to-Speech Output Module
DysVoice | Person 3 - Output & Integration

Converts transcribed text to spoken audio using pyttsx3.
Works offline on Windows (SAPI5), Mac (nsss), and Linux/Pi (espeak).
"""

import pyttsx3
import wave
import numpy as np
import tempfile
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _get_engine():
    """Initialise and configure the pyttsx3 TTS engine."""
    engine = pyttsx3.init()
    engine.setProperty('rate', config.TTS_RATE)
    engine.setProperty('volume', config.TTS_VOLUME)
    return engine


def speak(text: str) -> None:
    """
    Convert a text string to spoken audio output.

    Speaks the given text aloud through the system's default audio output.
    Does nothing if the text is empty or whitespace only.

    Args:
        text (str): The transcript text to speak aloud.

    Returns:
        None
    """
    if not text or not text.strip():
        return

    engine = _get_engine()
    print(f"Speaking output: {text}")
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def save_audio(text: str, output_path: str) -> str:
    """
    Save TTS output to a .wav file instead of playing it aloud.

    Useful for pre-generating demo audio or logging output.

    Args:
        text (str):        The text to convert to speech.
        output_path (str): File path where the .wav file will be saved.
                           E.g. 'output/demo_clip.wav'

    Returns:
        str: The path of the saved .wav file.
    """
    if not text or not text.strip():
        print("save_audio: empty text provided, nothing saved.")
        return output_path

    engine = _get_engine()
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    engine.stop()
    print(f"Audio saved to: {output_path}")
    return output_path


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_sentences = [
        "please bring me water",
        "turn off the lights",
        "I need help",
        "DysVoice is ready",
    ]

    print("=== speak.py test ===")
    print("Testing speak() with multiple sentences...\n")

    for sentence in test_sentences:
        print(f"  → '{sentence}'")
        speak(sentence)

    print("\nTesting save_audio()...")
    saved = save_audio("This audio was saved to a file.", "test_output.wav")
    if os.path.exists(saved):
        print(f"  ✓ File created: {saved} ({os.path.getsize(saved)} bytes)")
    else:
        print("  ✗ File was not created — check pyttsx3 save support on your OS.")

    print("\n=== All tests complete ===")