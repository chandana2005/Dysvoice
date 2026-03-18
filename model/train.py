import os
import librosa

TORGO_PATH = r"C:\Users\chand\Desktop\Dysvoice"
SAMPLE_RATE = 16000

def load_torgo_data(base_path):
    data = []
    
    for dys_folder in ["F_dys", "M_dys"]:
        dys_path = os.path.join(base_path, dys_folder)
        print(f"Checking: {dys_path}")
        
        if not os.path.exists(dys_path):
            print(f"  NOT FOUND - skipping")
            continue
        
        print(f"  Found! Scanning speakers...")
            
        for speaker in os.listdir(dys_path):
            speaker_path = os.path.join(dys_path, speaker)
            if not os.path.isdir(speaker_path):
                continue
            print(f"    Speaker: {speaker}")
                
            for session in os.listdir(speaker_path):
                session_path = os.path.join(speaker_path, session)
                wav_path = os.path.join(session_path, "wav_headMic")
                prompt_path = os.path.join(session_path, "prompts")
                
                if not os.path.exists(wav_path) or not os.path.exists(prompt_path):
                    print(f"      Session {session}: missing wav or prompts - skipping")
                    continue
                
                print(f"      Session {session}: found wav + prompts")
                
                for txt_file in os.listdir(prompt_path):
                    if not txt_file.endswith(".txt"):
                        continue
                    
                    txt_full = os.path.join(prompt_path, txt_file)
                    with open(txt_full, "r") as f:
                        transcript = f.read().strip()
                    
                    if transcript.startswith("["):
                        continue
                    import re
                    transcript = re.sub(r'\[.*?\]', '', transcript).strip()
                    
                    if not transcript:
                        continue
                    
                    wav_file = txt_file.replace(".txt", ".wav")
                    wav_full = os.path.join(wav_path, wav_file)
                    
                    if not os.path.exists(wav_full):
                        continue
                    
                    data.append((wav_full, transcript))
    
    print(f"\nTotal samples loaded: {len(data)}")
    return data

if __name__ == "__main__":
    print("Starting TORGO data loading...\n")
    data = load_torgo_data(TORGO_PATH)
    for wav, text in data[:5]:
        print(f"Audio: {wav}")
        print(f"Transcript: {text}")
        print("---")