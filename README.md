# Dysvoice
AI-Based Dysarthric Speech Assistance System

# DEVELOPER 1:
## Day 1
**What is this project?**
DysVoice is an AI system that listens to dysarthric speech (speech from people with conditions like cerebral palsy that affect how they talk) and converts it into clear text and spoken output. Think of it like a translator between dysarthric speech and normal communication.

- Step 1: Created the GitHub Repository
- Step 2: Cloned the repo to the laptop
- Step 3: Created the project folder structure
  Created an organised folder structure so each developer knows exactly where their files go :

1. model/ — Person 1's folder, contains all AI training code
2. audio/ — Person 2's folder, contains microphone recording and noise cleaning code
3. inference/ — Person 2's folder, contains the code that runs speech through the AI model
4. output/ — Person 3's folder, contains text to speech and display code
5. hardware/ — Person 3's folder, contains setup scripts for the Raspberry Pi

We also created empty files as placeholders so everyone knows what they need to fill in:

config.py — shared settings file used by all 3 developers
requirements.txt — list of all libraries the project needs
main.py — the master file that connects everything together
model/train.py — code to train the AI model
model/evaluate.py — code to test how accurate the model is
audio/record.py — code to record from microphone
audio/denoise.py — code to clean up background noise
inference/transcribe.py — code to convert speech to text
output/speak.py — code to convert text to speech
output/display.py — code to show text on screen

Also created __init__.py files in each folder — these are empty files that tell Python "this folder is a package you can import code from."
- Step 4: Wrote config.py
  This file contains shared settings that all 3 developer's code will use:

1. MODEL_NAME — which AI model we are using (Whisper Small from OpenAI)
2. SAMPLE_RATE — audio quality setting, 16000 means 16000 audio samples per second
3. MODEL_PATH — where the trained model file will be saved
4. DEVICE — whether to use CPU (normal laptop) or CUDA (GPU for faster training)
5. MAX_DURATION_SECONDS — maximum length of audio the system will process
6. TTS_RATE — how fast the text to speech voice speaks
7. TTS_VOLUME — how loud the text to speech voice is
- Step 5: Wrote requirements.txt
  This file lists every external library the project needs. When teammates clone the repo they run pip install -r requirements.txt and Python automatically downloads and installs everything. Libraries we need:

**.** torch — the deep learning engine that powers the AI
**.** transformers — loads the Whisper model from HuggingFace
**.** datasets — helps load and organise audio data
**.** librosa — loads audio files and converts sample rates
**.** soundfile — reads and writes audio files
**.** noisereduce — removes background noise from audio
**.** pyaudio — accesses the microphone
**.** pyttsx3 — converts text to speech
**.** evaluate — calculates how accurate the model is
**.** jiwer — calculates Word Error Rate (how many words the model gets wrong)
- Step 6: Pushed everything to GitHub
- Step 7: Added teammates as collaborators

## Day 2
**Goal of Day 2**
Write the data loading function — the code that reads all the audio files and their matching transcripts and prepares them for AI training.
**What is the TORGO Dataset?**
TORGO is a research dataset created by the University of Toronto. It contains recordings of real people with dysarthria (speech impairment) speaking words and sentences into a microphone. Each audio file has a matching text file that says exactly what the person said. This is called a labelled dataset — the AI needs both the audio AND the correct text to learn from.
- Step 1: Downloaded the TORGO dataset
Went to the official TORGO website at the University of Toronto
Downloaded F_dys.bz2 (female dysarthric speakers) and M_dys.bz2 (male dysarthric speakers)
.bz2 is a compressed file format like .zip — it makes large files smaller for downloading
Skipped F_con and M_con — these are "control" speakers (normal speech), not needed for training

- Step 2: Extracted the dataset
Windows cannot open .bz2 files natively
Used the built-in extraction tool to extract both files
This created two folders: F_dys and M_dys inside the Dysvoice project folder

- Step 3: Explored the dataset structure
After extracting we found this structure:
F_dys/
├── F01/          ← Female dysarthric speaker 1
│   └── Session1/
│       ├── wav_headMic/    ← audio files (.wav)
│       └── prompts/        ← transcript files (.txt)
├── F03/          ← Female dysarthric speaker 3
└── F04/          ← Female dysarthric speaker 4

M_dys/
├── M01/          ← Male dysarthric speaker 1
├── M02/
├── M03/
├── M04/
└── M05/
Total: 8 dysarthric speakers across both folders.
Each .wav file has a matching .txt file with the same number. For example 0001.wav contains the audio of someone saying what is written in 0001.txt.

- Step 4: Discovered two types of transcripts
When opened some .txt files, found two types:

Instruction prompts like [say Ah-P-Eee repeatedly] — these tell the speaker what sound to make, not actual speech sentences. So skip these.
Real sentence transcripts like "Except in the winter when the ooze or snow or ice prevents", — these are proper sentences use for training.
Some transcripts had inline instructions like "tear [as in tear up that paper]" — so clean these by removing the part inside brackets, keeping just tear.

- Step 5: Wrote the data loading function in model/train.py
This function does the following automatically:

. Goes into every speaker folder (F01, F03, F04, M01-M05)
. Goes into every session folder inside each speaker
. Finds the wav_headMic folder (audio) and prompts folder (transcripts)
. For every transcript file, reads the text inside
. Skips it if it starts with '[' (instruction prompt)
. Removes any inline bracket instructions from the text
. Finds the matching .wav audio file with the same number
. Adds the pair (audio path, transcript text) to a list
. At the end, returns the complete list of 2917 audio-transcript pairs

- Result
Successfully loaded 2917 clean audio-transcript pairs from 8 dysarthric speakers.

- Step 6: Created .gitignore file
The dataset files are 2.5GB total — too large to upload to GitHub. We created a .gitignore file which tells Git to ignore certain files and folders. We added the dataset folders to this file so only code gets uploaded to GitHub, not the audio data. Teammates download the dataset separately from the TORGO website.

- Step 7: Pushed code to GitHub
Pushed the updated model/train.py, requirements.txt and .gitignore to GitHub with the commit message "Day 2: data loading function complete, 2917 samples".
