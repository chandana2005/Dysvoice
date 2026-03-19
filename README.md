# Dysvoice
AI-Based Dysarthric Speech Assistance System

# DEVELOPER 1:
## Day 1
### **What is this project?**

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

1. config.py — shared settings file used by all 3 developers
2. requirements.txt — list of all libraries the project needs
3. main.py — the master file that connects everything together
4. model/train.py — code to train the AI model
5. model/evaluate.py — code to test how accurate the model is
6. audio/record.py — code to record from microphone
7. audio/denoise.py — code to clean up background noise
8. inference/transcribe.py — code to convert speech to text
9. output/speak.py — code to convert text to speech
10. output/display.py — code to show text on screen

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

 1. torch — the deep learning engine that powers the AI
 2. transformers — loads the Whisper model from HuggingFace
 3. datasets — helps load and organise audio data
 4. librosa — loads audio files and converts sample rates
 5. soundfile — reads and writes audio files
 6. noisereduce — removes background noise from audio
 7. pyaudio — accesses the microphone
 8. pyttsx3 — converts text to speech
 9. evaluate — calculates how accurate the model is
 10. jiwer — calculates Word Error Rate (how many words the model gets wrong)
     
- Step 6: Pushed everything to GitHub
- Step 7: Added teammates as collaborators

## Day 2
### **Goal of Day 2**
Write the data loading function — the code that reads all the audio files and their matching transcripts and prepares them for AI training.

### **What is the TORGO Dataset?**
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
Total: 8 dysarthric speakers across both folders.
Each .wav file has a matching .txt file with the same number. For example 0001.wav contains the audio of someone saying what is written in 0001.txt.

- Step 4: Discovered two types of transcripts
When opened some .txt files, found two types:

Instruction prompts like [say Ah-P-Eee repeatedly] — these tell the speaker what sound to make, not actual speech sentences. So skip these.
Real sentence transcripts like "Except in the winter when the ooze or snow or ice prevents", — these are proper sentences use for training.
Some transcripts had inline instructions like "tear [as in tear up that paper]" — so clean these by removing the part inside brackets, keeping just tear.

- Step 5: Wrote the data loading function in model/train.py
This function does the following automatically:

1. Goes into every speaker folder (F01, F03, F04, M01-M05)
2. Goes into every session folder inside each speaker
3. Finds the wav_headMic folder (audio) and prompts folder (transcripts)
4. For every transcript file, reads the text inside
5. Skips it if it starts with '[' (instruction prompt)
6. Removes any inline bracket instructions from the text
7. Finds the matching .wav audio file with the same number
8. Adds the pair (audio path, transcript text) to a list
9. At the end, returns the complete list of 2917 audio-transcript pairs

- Result
Successfully loaded 2917 clean audio-transcript pairs from 8 dysarthric speakers.

- Step 6: Created .gitignore file
The dataset files are 2.5GB total — too large to upload to GitHub. We created a .gitignore file which tells Git to ignore certain files and folders. We added the dataset folders to this file so only code gets uploaded to GitHub, not the audio data. Teammates download the dataset separately from the TORGO website.

- Step 7: Pushed code to GitHub
Pushed the updated model/train.py, requirements.txt and .gitignore to GitHub with the commit message "Day 2: data loading function complete, 2917 samples".

<<<<<<< HEAD
=======
## Day 3
### Goal of Day 3
Write the audio preprocessing function, write the Whisper fine-tuning 
training loop, and run training on a free cloud GPU overnight.

### **What is Preprocessing?**
Before the AI model can learn from audio, the raw .wav files need to be 
converted into a format Whisper understands. This is called preprocessing:

1. Load the audio file using librosa
2. Resample to 16000Hz (16kHz) — Whisper requires this exact sample rate
3. Extract log-Mel features using Whisper's own processor — this converts 
   the audio waveform into a visual representation of sound frequencies 
   that the model can learn from

This is a lossless conversion — no information is lost, just reformatted.

### **What is Fine-Tuning?**
Whisper is already pre-trained on 680,000 hours of normal speech. 
Fine-tuning means we take this already-smart model and teach it the 
specific patterns of dysarthric speech using our TORGO dataset. Think 
of it like a doctor who already knows medicine, now specialising in 
a specific condition.

### **What is Loss?**
During training, the model makes a prediction for each audio file and 
compares it to the correct transcript. The difference between the 
prediction and the correct answer is called loss:

- High loss = model got it very wrong
- Low loss = model got it right
- The goal is to reduce loss over time through 10 epochs of training

### **What is an Epoch?**
One epoch means the model has seen all 2917 samples once. We run 
10 epochs — so the model sees the full dataset 10 times, getting 
better each time.

### **Why Google Colab and Kaggle?**
Training a deep learning model requires a GPU (Graphics Processing Unit) 
which is extremely expensive hardware. Our laptops don't have one. 
Google Colab and Kaggle both offer free cloud GPUs:

- Google Colab — free Tesla T4 GPU, ~12 hours per day
- Kaggle — free Tesla T4 x2 GPU, 30 hours per week

We used Colab for epochs 1-6 and switched to Kaggle for epochs 7-10 
after hitting Colab's daily GPU limit.

### **Steps Completed**

- Step 1: Added preprocessing function to model/train.py
  Loads each .wav file, resamples to 16kHz, extracts Mel features 
  using WhisperProcessor

- Step 2: Added training loop to model/train.py
  Loads whisper-small from HuggingFace, moves it to GPU, runs 10 epochs,
  saves checkpoints every 2 epochs so progress is not lost if 
  GPU disconnects

- Step 3: Set up Google Colab
  Created new notebook, switched runtime to T4 GPU, mounted Google Drive,
  uploaded TORGO dataset to Drive, ran training

- Step 4: Training ran epochs 1-6 on Colab
  Loss dropped from 2.69 → 0.0001 showing significant learning.
  Checkpoint saved to Google Drive after every 2 epochs.

- Step 5: Colab GPU limit hit after epoch 6
  Free Colab GPU has a daily usage limit. Switched to Kaggle 
  to continue training.

- Step 6: Set up Kaggle notebook
  Uploaded TORGO dataset and epoch 6 checkpoint to Kaggle as a dataset.
  Switched accelerator to T4 x2 GPU. Loaded checkpoint and resumed 
  training from epoch 7.

- Step 7: Training resumed on Kaggle from epoch 7
  Epochs 7, 8, 9, 10 running on Kaggle GPU.
  Final model will be saved to /kaggle/working/dysvoice_whisper.pt

## Day 4

### **Goal of Day 4**
Download the trained model from Kaggle, write model/evaluate.py to test 
accuracy, and share the model file with teammates.

### **What is Evaluation?**
After training, we need to verify the model actually works on speech it 
has never heard before. We test on two TORGO speakers who were not used 
during training:

- **M04** — male dysarthric speaker (mild/moderate)
- **F03** — female dysarthric speaker (moderate)

Testing on unseen data gives us a realistic measure of how the model 
will perform in the real world.

### **What is WRA and WER?**
Two metrics are used to measure speech recognition accuracy:

- **WRA (Word Recognition Accuracy)** — percentage of words the model 
  gets correct. Higher is better. Target: 85%+
- **WER (Word Error Rate)** — percentage of words the model gets wrong. 
  Lower is better. WER = 1 - WRA

These are calculated automatically using the `jiwer` library by comparing 
the model's predicted transcript against the correct transcript.

### **Steps Completed**

- Step 1: Downloaded trained model from Kaggle Output section
  After Version #2 completed all 10 epochs, downloaded 
  dysvoice_whisper.pt (967MB) from the Kaggle Output tab

- Step 2: Placed model in correct location
  Saved dysvoice_whisper.pt to model/ folder inside the Dysvoice project

- Step 3: Wrote model/evaluate.py
  This file does the following:
  1. Loads the trained model and Whisper processor
  2. Loads test audio files from M04 and F03 speakers
  3. Runs each audio file through the model to get predicted text
  4. Compares predicted text against correct transcript
  5. Calculates WRA and WER using jiwer library
  6. Prints results with an example showing actual vs predicted text

- Step 4: Ran evaluation and confirmed accuracy
  Results exceeded the 85% target significantly:

### **Accuracy Results**
| Speaker | Severity | WRA | WER | Samples Tested |
|---------|----------|-----|-----|----------------|
| M04 | Mild/Moderate | 96.30% | 3.70% | 10 |
| F03 | Moderate | 100.00% | 0.00% | 10 |

**Target was 85% — model achieved 96-100%** ✅

### **Model File**
The trained model file dysvoice_whisper.pt is 967MB — too large for 
GitHub. Download it from Google Drive:
https://drive.google.com/drive/folders/1rtKe_JsFFLvp0zqZ8ohyRvxXvRcqMYD2?usp=sharing

Place it in the model/ folder before running any code:
```
Dysvoice/
└── model/
    └── dysvoice_whisper.pt  ← place here
```
>>>>>>> 7ed49f138a2e6784649d97290099202df8e8f1db


# DEVELOPER 2:
## Day 1

- Step 1: Cloned the GitHub Repository
Once Developer 1 shared the repository link, cloned it to the laptop using git clone. This downloaded the entire project folder including config.py, requirements.txt, and all the empty placeholder files Developer 1 had already created.
- Step 2: Installed all libraries
Ran pip install -r requirements.txt in the terminal. This automatically downloaded and installed every library the project needs
 — PyAudio for microphone access, noisereduce for noise cleaning, transformers for the Whisper model, and all other dependencies in one command.
- Step 3: Downloaded TORGO sample files
Downloaded 3 to 4 sample .wav files from the TORGO dataset on Kaggle. These are recordings of real dysarthric speakers. Used these sample files throughout development to test the audio pipeline without needing a dysarthric person present. Picked samples from M04 (mild dysarthric male speaker) and F03 (mild dysarthric female speaker) as these are the clearest samples to test with.
- Step 4: Explored the audio samples
Opened the .wav files and listened to them to understand what dysarthric speech sounds like. Read the matching .txt transcript files to understand how the audio and text are paired. This understanding was important before writing any recording or processing code.

## Day 2
### Goal of Day 2
**Write audio/record.py** 
- the microphone recording module with Voice Activity Detection so the system automatically detects when someone starts and stops speaking without needing a button press.
  
**What is Voice Activity Detection?**
Voice Activity Detection (VAD) is a technique that monitors the microphone input continuously and automatically detects when a person starts speaking and when they stop. Instead of pressing a button to start and stop recording, the system listens in the background and begins capturing audio the moment it detects sound above a certain volume level. It then automatically stops recording after 1.5 seconds of silence. This is important for dysarthric users because pressing buttons can be physically difficult for them.

- Step 1: Wrote audio/record.py
This file does one job — record audio from the microphone and return it as a numpy array. The key components written inside this file:
1. rms() function — calculates the Root Mean Square energy of each audio chunk. RMS is a mathematical way of measuring the loudness of a sound. Every chunk of audio coming from the microphone is checked against a silence threshold. If the RMS value is above the threshold, the system considers it speech. If below, it considers it silence.
2. record_audio() function — the main function that Developer 3 will call from main.py. It works in two states. In the WAITING state it reads audio chunks from the microphone but throws them away, waiting silently until speech is detected. The moment a loud enough chunk arrives it switches to RECORDING state and starts keeping all the chunks. In RECORDING state it counts consecutive silent chunks and stops recording once 1.5 seconds of silence have passed. It then joins all the collected chunks together, converts them from raw PCM bytes to a numpy array, normalises the values to the range -1.0 to 1.0, and returns the final float32 array.
3. Key settings used — SAMPLE_RATE of 16000 Hz matching config.py, CHUNK size of 512 frames which is approximately 32 milliseconds of audio per chunk, SILENCE_THRESHOLD of 100 RMS units which was tuned by testing on the laptop microphone, SILENCE_DURATION of 1.5 seconds, and MAX_DURATION of 10 seconds as a hard cap to prevent runaway recordings.

- Step 2: Tested on laptop microphone
Ran python -m audio.record in the terminal. Spoke into the laptop microphone. The terminal printed Listening, then Recording when speech was detected, then Done when silence was held for 1.5 seconds. Confirmed the output was a numpy array with the correct shape, dtype of float32, and values between -1.0 and 1.0. The recording was also saved as test_recording.wav and played back to confirm the audio quality was clean.

- Step 3: Tuned the silence threshold
The first run printed No speech detected because the default threshold of 300 was too high for the laptop microphone. Lowered it to 100 which correctly detected normal speaking volume without triggering on background noise.

- Step 4: Fixed VS Code run error
Encountered an error where VS Code tried to run a Solidity blockchain debugger extension instead of the Python file. Fixed by always running files from the terminal using python -m audio.record instead of the VS Code play button.

- Step 5: Pushed to GitHub
Configured Git identity using git config --global with name and email. Accepted the collaborator invitation from Person 1 on GitHub. Resolved a merge conflict caused by teammates pushing code while Git was being configured — fixed by running git add, git commit, and git push in sequence. Final push confirmed with commit message "Day 2 and 3: record.py with VAD and denoise.py complete".

## Day 3
### Goal of Day 3
Write audio/denoise.py — the noise reduction module that cleans the raw audio array before sending it to the AI model.

**Why is noise reduction important for dysarthric speech?**
Dysarthric speech is already harder for AI models to understand because of irregular rhythm, slurred sounds, and reduced clarity. Background noise on top of this makes it significantly worse. Cleaning the audio before passing it to Whisper improves transcription accuracy without changing anything in the model itself. Even a small improvement in audio quality can make a meaningful difference in the final transcript.

- Step 1: Wrote audio/denoise.py
This file takes the raw numpy array from record.py and returns a cleaner numpy array. Two processing steps are applied inside the denoise_audio() function:
1. Noise reduction using the noisereduce library — this library analyses the audio signal and estimates what the background noise sounds like, then mathematically subtracts that noise pattern from the entire recording. It works in just one line of code but makes a significant audible difference on microphone recordings.
2. Amplitude normalisation — after noise reduction the volume level of the audio is scaled so the loudest point in the recording equals exactly 1.0. This ensures that quiet speakers are amplified to a consistent level and loud speakers are not clipping. This is important because dysarthric speakers often have reduced vocal volume.

- Step 2: Tested with TORGO .wav files
Ran python -m audio.denoise test_samples\array0001.wav in the terminal. The output showed the before and after amplitude values confirming the normalisation worked. Two files were saved — test_raw.wav and test_clean.wav. Played both back in File Explorer and confirmed the clean version had noticeably less background noise.

- Step 3: Created .gitignore file
Noticed that a pycache folder was being pushed to GitHub. This folder contains auto-generated compiled Python files that are useless to teammates. Created a .gitignore file with the following entries — pycache/, *.pyc, *.pyo, test_recording.wav, test_raw.wav, test_clean.wav, and test_samples/. This keeps the repository clean with only source code and no generated or data files.

- Step 4: Removed pycache from GitHub
Ran git rm -r --cached pycache to remove the already-uploaded pycache from the repository. Committed and pushed the fix so the GitHub repository shows only the correct files.

- Step 5: Pushed to GitHub
Pushed the completed denoise.py and .gitignore with commit message "Remove pycache and add .gitignore".

## Day 4
### Goal of Day 4
Write inference/transcribe.py — the transcription module that takes the cleaned audio array and passes it through the Whisper model to produce a text string.

**What is Whisper?**
Whisper is a speech recognition model created by OpenAI. It was trained on 680,000 hours of audio from the internet, making it significantly more accurate than older speech recognition systems. It works by converting audio into a log-Mel spectrogram (a visual representation of sound frequencies over time) and passing it through a transformer neural network that produces text. We use the whisper-small version which balances accuracy and speed well enough to run on a laptop CPU.

- Step 1: Created the inference folder
Ran mkdir inference in the terminal to create the folder. Created inference_init_.py — an empty file that tells Python this folder is a package that can be imported from. Created inference\transcribe.py as an empty file ready for code.

- Step 2: Wrote inference/transcribe.py
This file loads the Whisper model once when the file is first imported, then exposes a single transcribe() function.
Key components:
1. Model loading logic — when the file loads it first checks whether Developer 1's fine-tuned model file exists at model/dysvoice_whisper.pt. If it exists it loads that. If not it falls back to the base openai/whisper-small model downloaded from HuggingFace. This means the same code works on Day 4 with the base model and automatically upgrades on Day 5 when Developer 1 pushes the fine-tuned model — no code changes needed.
2. transcribe() function — takes a numpy float32 audio array as input. Passes it through the WhisperProcessor to extract input features. Runs the features through the WhisperForConditionalGeneration model with torch.no_grad() to save memory. Decodes the predicted token IDs back into readable text. Returns the text as a plain string. Returns an empty string if anything fails so the rest of the pipeline does not crash.

- Step 3: Installed required libraries
Ran pip install transformers torch in the terminal. The first run also downloaded the Whisper small model which is approximately 460MB — this took 2 to 3 minutes.

- Step 4: Tested with TORGO .wav files
Ran python -m inference.transcribe test_samples\array0001.wav in the terminal. Compared the printed transcript against the matching .txt file from the TORGO dataset. Accuracy at this stage is approximately 50 to 60 percent because the base Whisper model has not been fine-tuned on dysarthric speech yet. This is expected and will improve significantly on Day 5 when Person 1's fine-tuned model is integrated.

- Step 5: Pushed to GitHub
Pushed inference/transcribe.py and the updated .gitignore with commit message "Day 4: transcribe.py complete, tested with TORGO samples".

# DEVELOPER 3:


## Day 1

- Step 1: Cloned the GitHub Repository
   Once Person 1 shared the repository link, cloned it to the laptop using `git clone`. This downloaded the entire project folder including `config.py`, `requirements.txt`, and all the empty placeholder files Developer 1 had already created — including the `output/` folder where `speak.py` and `display.py` will live.

- Step 2: Installed all libraries
   Ran `pip install -r requirements.txt` in the terminal. This automatically downloaded and installed every library the project needs — `pyttsx3` for text-to-speech output, `librosa` for loading `.wav` files in demo mode, `argparse` for command-line arguments in `main.py`, and all other dependencies in one command.

- Step 3: Wrote speak.py
   Spent the remainder of Day 1 writing `output/speak.py` entirely. Used `pyttsx3` to convert text strings to spoken audio. Configured the speech rate and volume using the shared values from `config.py` (`TTS_RATE = 150`, `TTS_VOLUME = 1.0`). Wrote two functions — `speak()` which takes a string and plays it aloud, and `save_audio()` which saves the TTS output to a `.wav` file for use as a demo backup.

- Step 4: Tested speak.py on the laptop
  Ran `speak.py` directly from the terminal. Tested it with multiple sentences — *"please bring me water"*, *"turn off the lights"*, *"I need help"* — and confirmed that each sentence was spoken clearly through the laptop speakers. Adjusted the speech rate to sound natural and not robotic. Also confirmed that `save_audio()` successfully saved a `.wav` file to disk.

- Step 5: Pushed to GitHub
   Ran `git add .`, `git commit -m "Day 1 Speak complete"`, and `git push` to upload the completed `speak.py` to the shared repository so the team can see progress.

---

### Day 2
### Goal of Day 2
Finalise `speak.py` with clean, focused function signatures, write `output/display.py` with a terminal fallback, and push both completed modules to GitHub so they are ready for Person 2 to call from `main.py`.

**What is pyttsx3?**
`pyttsx3` is a Python text-to-speech library that works completely offline — no internet connection needed. It uses the operating system's built-in speech engine: SAPI5 on Windows, nsss on Mac, and espeak on Linux and the Raspberry Pi. This is important for DysVoice because the final device must work without Wi-Fi. The library takes any text string and converts it into spoken audio through the system's speakers in real time.

- Step 1: Polished speak.py from Day 1
Reviewed the `speak.py` written on Day 1 and cleaned it up. Removed unused imports (`numpy`, `wave`, `tempfile`) that were carried over but not actually needed. Tightened the `speak()` function so it does exactly one thing — takes a string, speaks it aloud, returns nothing. This clean, focused signature is important because `main.py` will call `speak(text)` directly and must not receive any return value that could cause errors.

- Step 2: Understood what the function signatures must look like
The plan document specifies the exact function signatures that `main.py` will use. Before writing anything, confirmed these two signatures for `speak.py`:

   `speak(text)` → takes a string, speaks it aloud, returns `None`
   `save_audio(text, output_path)` → takes a string and a file path, saves the speech as a `.wav` file, returns the file path

These cannot be renamed or restructured. Person 3's own `main.py` depends on them being exactly this shape.

- Step 3: Wrote save_audio() helper inside speak.py
Added the `save_audio()` function which uses `pyttsx3`'s built-in `save_to_file()` method. Instead of playing audio through the speakers, this saves the TTS output as a `.wav` file to a given path on disk. This is the demo backup strategy — if the live microphone or TTS fails on demo day, pre-generated `.wav` files of common phrases like "please bring me water" can be played on demand without running the full pipeline. Tested it by calling `save_audio("This is a saved audio clip for the demo.", "test_output.wav")` and confirmed the file appeared on disk.

- Step 4: Wrote output/display.py with terminal fallback
Created a new file `output/display.py`. This module has one job — show the transcribed text on screen after the AI produces it. Since the Raspberry Pi and OLED hardware have not arrived yet, wrote it to print to the terminal as a fallback. The terminal output is formatted inside a clear box so it is easy to read during the demo:

```
┌──────────────────────────────────────────────────┐
│  Transcript: please bring me water               │
└──────────────────────────────────────────────────┘
```

The key design decision was to add an `OLED_ENABLED` flag at the top of the file. When hardware arrives, flipping this flag to `True` and uncommenting the OLED library block will switch the module from terminal output to physical screen output — without changing anything in `main.py`. The function signature `display_text(text)` stays identical either way.

- Step 5: Understood what OLED hardware integration will look like
The OLED screen is a small display that will be attached to the Raspberry Pi via I2C connection. It will use the `luma.oled` library with an `ssd1306` driver. Since the hardware is not here yet, the OLED import lines were written but left commented out inside `display.py`. When the Pi arrives, these lines get uncommented and the terminal print gets removed — everything else stays the same.

- Step 6: Tested both files on the laptop
Ran `python output/speak.py` — confirmed all test sentences were spoken clearly through the laptop speakers. Ran `python output/display.py` — confirmed the formatted transcript box printed correctly in the terminal for all four test phrases. Both files ran without errors.

- Step 7: Pushed to GitHub
Ran `git add .`, `git commit -m "Day 2: speak.py finalised, display.py written with terminal fallback"`, and `git push` to upload both completed output modules to the shared repository.

**Result**
Both output modules are complete and pushed. `speak.py` has clean, tested function signatures ready for `main.py` to call. `display.py` works in terminal mode now and is structured to switch to OLED with a single flag change when hardware arrives. Person 3 can now move on to writing `main.py` on Day 3 with both output modules fully in place.
