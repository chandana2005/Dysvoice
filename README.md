# DysVoice
### AI-Based Dysarthric Speech Assistance System

DysVoice is an AI system that listens to dysarthric speech (speech from people with conditions like cerebral palsy that affect how they talk) and converts it into clear text and spoken output. Think of it like a translator between dysarthric speech and normal communication.

---

## Project Structure

```
Dysvoice/
├── config.py                  ← Shared settings used by all 3 developers
├── requirements.txt           ← All libraries the project needs
├── main.py                    ← Master file that connects everything together
├── model/
│   ├── train.py               ← Code to train the AI model
│   ├── evaluate.py            ← Code to test how accurate the model is
│   ├── test_model.py          ← Script to verify the model loads and transcribes correctly
│   └── dysvoice_whisper.pt    ← Trained model file (download from Google Drive — see below)
├── audio/
│   ├── record.py              ← Code to record from microphone
│   └── denoise.py             ← Code to clean up background noise
├── inference/
│   └── transcribe.py          ← Code to convert speech to text
├── output/
│   ├── speak.py               ← Code to convert text to speech
│   └── display.py             ← Code to show text on screen
└── hardware/
    └── setup.sh               ← Setup script for Raspberry Pi
```

---

## Model Download

The trained model file `dysvoice_whisper.pt` is 967MB — too large for GitHub. Download it from Google Drive:

🔗 [Download dysvoice_whisper.pt](https://drive.google.com/drive/folders/1rtKe_JsFFLvp0zqZ8ohyRvxXvRcqMYD2?usp=sharing)

Place it in the `model/` folder before running any code:

```
Dysvoice/
└── model/
    └── dysvoice_whisper.pt  ← place here
```

---

## Accuracy Results

Model evaluated on TORGO speakers not seen during training:

| Speaker | Severity | WRA | WER | Samples Tested |
|---------|----------|-----|-----|----------------|
| M04 | Mild/Moderate | 96.30% | 3.70% | 10 |
| F03 | Moderate | 100.00% | 0.00% | 10 |

**Target was 85% — model achieved 96–100%** ✅

---

## Setup

```bash
git clone <repo_url>
cd Dysvoice
pip install -r requirements.txt
```

Then download `dysvoice_whisper.pt` from the link above and place it in `model/`.

---

## Run

```bash
# Live microphone mode
python main.py

# Demo backup mode (file input)
python main.py --file path/to/audio.wav

# Verify model is working
python -m model.test_model
```

---

## Key Libraries

| Library | Purpose |
|---------|---------|
| `torch` | Deep learning engine |
| `transformers` | Loads and runs the Whisper model |
| `librosa` | Loads audio files and resamples to 16kHz |
| `noisereduce` | Removes background noise from audio |
| `pyaudio` | Microphone access and recording |
| `pyttsx3` | Text-to-speech output (works offline) |
| `jiwer` | Calculates Word Error Rate (WER) |

---

## Developer Log

---

# Developer 1 — AI & Model Training

## Day 1

### Goal
Set up the shared GitHub repository, create the project folder structure, and write the shared configuration files so all three developers can start working immediately.

### What is this project?
DysVoice is an AI system that listens to dysarthric speech and converts it into clear text and spoken output. Think of it like a translator between dysarthric speech and normal communication.

### Steps Completed

- **Step 1:** Created the GitHub repository and shared the link with Developers 2 and 3
- **Step 2:** Created the project folder structure with placeholder files so each developer knows exactly where their files go
- **Step 3:** Wrote `config.py` with shared settings used by all 3 developers:
  1. `MODEL_NAME` — which AI model we are using (Whisper Small from OpenAI)
  2. `SAMPLE_RATE` — audio quality setting, 16000 means 16000 audio samples per second
  3. `MODEL_PATH` — where the trained model file will be saved
  4. `DEVICE` — whether to use CPU (normal laptop) or CUDA (GPU for faster training)
  5. `MAX_DURATION_SECONDS` — maximum length of audio the system will process
  6. `TTS_RATE` — how fast the text-to-speech voice speaks
  7. `TTS_VOLUME` — how loud the text-to-speech voice is
- **Step 4:** Wrote `requirements.txt` listing every library the project needs. When teammates clone the repo they run `pip install -r requirements.txt` and Python automatically installs everything:
  1. `torch` — deep learning engine
  2. `transformers` — loads the Whisper model from HuggingFace
  3. `datasets` — helps load and organise audio data
  4. `librosa` — loads audio files and converts sample rates
  5. `soundfile` — reads and writes audio files
  6. `noisereduce` — removes background noise from audio
  7. `pyaudio` — accesses the microphone
  8. `pyttsx3` — converts text to speech
  9. `evaluate` — calculates how accurate the model is
  10. `jiwer` — calculates Word Error Rate
- **Step 5:** Created `__init__.py` files in each folder so Python recognises them as importable packages
- **Step 6:** Pushed everything to GitHub and added teammates as collaborators

---

## Day 2

### Goal
Write the data loading function — the code that reads all the audio files and their matching transcripts and prepares them for AI training.

### What is the TORGO Dataset?
TORGO is a research dataset created by the University of Toronto. It contains recordings of real people with dysarthria speaking words and sentences into a microphone. Each audio file has a matching text file that says exactly what the person said. This is called a labelled dataset — the AI needs both the audio AND the correct text to learn from.

### Steps Completed

- **Step 1:** Downloaded the TORGO dataset — `F_dys.bz2` (female dysarthric speakers) and `M_dys.bz2` (male dysarthric speakers) from the University of Toronto website. Skipped `F_con` and `M_con` (control/normal speakers, not needed for training)
- **Step 2:** Extracted both `.bz2` files to create `F_dys/` and `M_dys/` folders inside the project
- **Step 3:** Explored the dataset structure — 8 dysarthric speakers total. Each `.wav` file has a matching `.txt` transcript with the same number
- **Step 4:** Discovered two types of transcripts:
  - Instruction prompts like `[say Ah-P-Eee repeatedly]` — skip these
  - Real sentence transcripts like `"Except in the winter when the ooze or snow or ice prevents"` — use these for training
  - Some transcripts had inline instructions like `"tear [as in tear up that paper]"` — cleaned by removing the bracket content
- **Step 5:** Wrote the data loading function in `model/train.py`. This function automatically processes all 8 speaker folders, finds matching audio-transcript pairs, skips instruction prompts, and returns a clean list of 2917 pairs
- **Step 6:** Created `.gitignore` to exclude the 2.5GB dataset from GitHub — teammates download the dataset separately from the TORGO website
- **Step 7:** Pushed updated `model/train.py`, `requirements.txt`, and `.gitignore` to GitHub

### Result
Successfully loaded **2917 clean audio-transcript pairs** from 8 dysarthric speakers.

---

## Day 3

### Goal
Write the audio preprocessing function, write the Whisper fine-tuning training loop, and run training on a free cloud GPU overnight.

### What is Preprocessing?
Before the AI model can learn from audio, the raw `.wav` files need to be converted into a format Whisper understands:
1. Load the audio file using `librosa`
2. Resample to 16000Hz — Whisper requires this exact sample rate
3. Extract log-Mel features using Whisper's own processor — this converts the audio waveform into a visual representation of sound frequencies that the model can learn from

### What is Fine-Tuning?
Whisper is already pre-trained on 680,000 hours of normal speech. Fine-tuning means we take this already-smart model and teach it the specific patterns of dysarthric speech using our TORGO dataset. Think of it like a doctor who already knows medicine, now specialising in a specific condition.

### What is an Epoch?
One epoch means the model has seen all 2917 samples once. We run 10 epochs — so the model sees the full dataset 10 times, getting better each time.

### Steps Completed

- **Step 1:** Added preprocessing function to `model/train.py` — loads each `.wav`, resamples to 16kHz, extracts Mel features using `WhisperProcessor`
- **Step 2:** Added training loop — loads `whisper-small` from HuggingFace, moves it to GPU, runs 10 epochs, saves checkpoints every 2 epochs so progress is not lost if the GPU disconnects
- **Step 3:** Set up Google Colab — switched runtime to T4 GPU, mounted Google Drive, uploaded TORGO dataset, ran training
- **Step 4:** Training ran epochs 1–6 on Colab. Loss dropped from 2.69 → 0.0001 showing significant learning
- **Step 5:** Hit Colab's daily GPU limit after epoch 6 — switched to Kaggle to continue
- **Step 6:** Set up Kaggle notebook — uploaded TORGO dataset and epoch 6 checkpoint, switched to T4 x2 GPU, loaded checkpoint and resumed from epoch 7
- **Step 7:** Epochs 7–10 completed on Kaggle. Final model saved as `dysvoice_whisper.pt`

---

## Day 4

### Goal
Download the trained model from Kaggle, write `model/evaluate.py` to test accuracy, and share the model file with teammates.

### What is WRA and WER?
- **WRA (Word Recognition Accuracy)** — percentage of words the model gets correct. Higher is better. Target: 85%+
- **WER (Word Error Rate)** — percentage of words the model gets wrong. Lower is better

These are calculated automatically using the `jiwer` library by comparing the model's predicted transcript against the correct transcript.

### Steps Completed

- **Step 1:** Downloaded `dysvoice_whisper.pt` (967MB) from Kaggle Output tab after all 10 epochs completed
- **Step 2:** Placed the model file in the `model/` folder
- **Step 3:** Wrote `model/evaluate.py` — loads the trained model, runs test audio from M04 and F03 speakers through it, calculates WRA and WER using `jiwer`, prints results with example transcripts
- **Step 4:** Ran evaluation on unseen test speakers — results significantly exceeded the 85% target (see Accuracy Results table above)
- **Step 5:** Uploaded `dysvoice_whisper.pt` to Google Drive and shared the link with the team

---

## Day 5

### Goal
Verify the trained model loads and transcribes correctly end-to-end. Write `model/test_model.py` as a standalone verification script. Share the model loading code with Developer 2 so they can integrate it into `inference/transcribe.py`.

### What is test_model.py?
`test_model.py` is a quick verification script written now that the trained model exists. There was no point writing it earlier since the model was not ready. It does three things:
1. Loads the fine-tuned model from `config.MODEL_PATH`
2. Loads a TORGO `.wav` file using `librosa`
3. Runs the audio through the model and prints the transcript

This confirms the entire model loading and inference chain works correctly before Developer 2 integrates it into the full pipeline.

### Steps Completed

- **Step 1:** Uploaded `dysvoice_whisper.pt` to Google Drive and shared the download link with the team on WhatsApp
- **Step 2:** Wrote `model/test_model.py` with full error handling using `try/except` so any issues print clearly instead of failing silently
- **Step 3:** Fixed a `ModuleNotFoundError` for `config` — resolved by running the script as `python -m model.test_model` instead of `python model/test_model.py` so Python can find `config.py` in the root folder
- **Step 4:** Added `language="en"` and `attention_mask` to the generate call to suppress HuggingFace warnings about multilingual mode and padding
- **Step 5:** Added `model.config.forced_decoder_ids = None` and `processor.tokenizer.forced_decoder_ids = None` to resolve a conflict between the fine-tuned model's decoder settings and the `language="en"` parameter
- **Step 6:** Confirmed successful output:
  ```
  ✅ Model loaded successfully
  ✅ Audio loaded successfully
  🎙  Input file : ...wav_headMic\0001.wav
  📝 Transcript : I'll be healthy.
  ```
- **Step 7:** Shared the model loading snippet with Developer 2 on WhatsApp so they can swap the base Whisper placeholder in `transcribe.py` for the fine-tuned model
- **Step 8:** Pulled latest code from GitHub (received Developer 3's `speak.py`, `display.py`, and `README.md` updates), then pushed `test_model.py` to GitHub

### Model Loading Code (for Developer 2)
This is the exact snippet to use in `inference/transcribe.py`:

```python
import torch
import config
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained(config.MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(config.MODEL_NAME)

state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
model.load_state_dict(state_dict)
model.to(config.DEVICE)
model.eval()
model.config.forced_decoder_ids = None
processor.tokenizer.forced_decoder_ids = None
```

---

# Developer 2 — Audio Pipeline

## Day 1

### Steps Completed

- **Step 1:** Cloned the GitHub repository once Developer 1 shared the link
- **Step 2:** Ran `pip install -r requirements.txt` to install all libraries
- **Step 3:** Downloaded 3–4 sample `.wav` files from the TORGO dataset on Kaggle — one mild speaker (M04) and one moderate speaker (F03) — for testing throughout development
- **Step 4:** Explored the audio samples — listened to the `.wav` files and read the matching `.txt` transcripts to understand what dysarthric speech looks and sounds like

---

## Day 2

### Goal
Write `audio/record.py` — the microphone recording module with Voice Activity Detection (VAD).

### What is Voice Activity Detection?
VAD monitors the microphone continuously and automatically detects when a person starts and stops speaking. Instead of pressing a button, the system begins capturing audio the moment it detects sound above a volume threshold and stops after 1.5 seconds of silence. This is important for dysarthric users because pressing buttons can be physically difficult.

### Steps Completed

- **Step 1:** Wrote `audio/record.py` with two key components:
  1. `rms()` function — calculates Root Mean Square energy of each audio chunk to measure loudness
  2. `record_audio()` function — runs in two states: WAITING (listening for speech) and RECORDING (capturing audio). Stops after 1.5 seconds of silence. Returns a `float32` numpy array normalised to the range -1.0 to 1.0
  - Key settings: `SAMPLE_RATE=16000`, `CHUNK=512`, `SILENCE_THRESHOLD=100`, `SILENCE_DURATION=1.5s`, `MAX_DURATION=10s`
- **Step 2:** Tested on laptop microphone — confirmed output numpy array with correct shape, dtype `float32`, and values between -1.0 and 1.0
- **Step 3:** Tuned silence threshold from 300 → 100 after initial run failed to detect speech
- **Step 4:** Fixed VS Code run error — always use `python -m audio.record` from terminal, not the VS Code play button
- **Step 5:** Pushed to GitHub

---

## Day 3

### Goal
Write `audio/denoise.py` — the noise reduction module.

### Why is noise reduction important for dysarthric speech?
Dysarthric speech is already harder for AI models to understand. Background noise makes it significantly worse. Even a small improvement in audio quality can make a meaningful difference in the final transcript.

### Steps Completed

- **Step 1:** Wrote `audio/denoise.py` with two processing steps inside `denoise_audio()`:
  1. Noise reduction using `noisereduce` — estimates and subtracts the background noise pattern from the recording
  2. Amplitude normalisation — scales audio so the loudest point equals 1.0, ensuring quiet speakers are amplified consistently
- **Step 2:** Tested with TORGO `.wav` files — saved `test_raw.wav` and `test_clean.wav` and confirmed audible noise reduction
- **Step 3:** Created `.gitignore` and removed `__pycache__` from GitHub with `git rm -r --cached`
- **Step 4:** Pushed to GitHub

---

## Day 4

### Goal
Write `inference/transcribe.py` — the transcription module.

### What is Whisper?
Whisper is a speech recognition model by OpenAI trained on 680,000 hours of audio. It converts audio into a log-Mel spectrogram and passes it through a transformer neural network to produce text. We use `whisper-small` which balances accuracy and speed for laptop CPU use.

### Steps Completed

- **Step 1:** Created the `inference/` folder with `__init__.py`
- **Step 2:** Wrote `inference/transcribe.py`:
  1. On import, checks if `dysvoice_whisper.pt` exists — loads it if present, falls back to base `openai/whisper-small` if not. This means the code works on Day 4 and automatically upgrades on Day 5 when the fine-tuned model arrives — no code changes needed
  2. `transcribe()` function — takes a `float32` numpy array, runs it through Whisper, returns a plain text string. Returns empty string on failure so the pipeline does not crash
- **Step 3:** Tested with TORGO `.wav` files — accuracy ~50–60% with base model (expected, will improve to 96%+ after integrating fine-tuned model on Day 5)
- **Step 4:** Pushed to GitHub

---

# Developer 3 — Output & Integration

## Day 1

### Steps Completed

- **Step 1:** Cloned the GitHub repository
- **Step 2:** Ran `pip install -r requirements.txt`
- **Step 3:** Wrote `output/speak.py` entirely on Day 1 using `pyttsx3`. Configured speech rate (`TTS_RATE=150`) and volume (`TTS_VOLUME=1.0`) from `config.py`. Tested with sentences like *"please bring me water"* and *"turn off the lights"*
- **Step 4:** Confirmed `save_audio()` successfully saves TTS output as a `.wav` file to disk
- **Step 5:** Pushed to GitHub

---

## Day 2

### Goal
Finalise `speak.py`, write `output/display.py` with terminal fallback.

### What is pyttsx3?
`pyttsx3` is a Python text-to-speech library that works completely offline. It uses the OS built-in speech engine: SAPI5 on Windows, nsss on Mac, espeak on Linux/Pi. Critical for DysVoice since the final device must work without Wi-Fi.

### Steps Completed

- **Step 1:** Polished `speak.py` — removed unused imports, tightened `speak()` to do exactly one thing: take a string, speak it, return nothing
- **Step 2:** Confirmed exact function signatures that `main.py` depends on:
  - `speak(text)` → speaks aloud, returns `None`
  - `save_audio(text, output_path)` → saves speech as `.wav`, returns the file path
- **Step 3:** Wrote `save_audio()` using `pyttsx3`'s `save_to_file()` method for demo backup pre-generation
- **Step 4:** Wrote `output/display.py` with terminal fallback, formatted as:
  ```
  ┌──────────────────────────────────────────────────┐
  │  Transcript: please bring me water               │
  └──────────────────────────────────────────────────┘
  ```
  Added an `OLED_ENABLED` flag — when hardware arrives, flipping this to `True` switches from terminal to OLED output without changing anything in `main.py`
- **Step 5:** Tested both files — all sentences spoken correctly, display box printed correctly
- **Step 6:** Pushed to GitHub

---

*Git rule: always `git pull` before `git push`. Never edit a file that belongs to another developer.*
