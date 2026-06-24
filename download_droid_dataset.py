import os
import time
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
REPO_ID = "GEAR-Dreams/DreamZero-DROID-Data"
REPO_TYPE = "dataset"
LOCAL_DIR = "./data/droid_lerobot"
COOLDOWN_MINUTES = 15  # Time to wait before retrying when quota is hit
# ---------------------

def download_with_retry():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    attempt = 1
    while True:
        print(f"\n[Attempt {attempt}] Starting/Resuming dataset download...")
        try:
            # snapshot_download mirrors the behavior of 'hf download'
            snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                local_dir=LOCAL_DIR,
                resume_download=True,  # Keeps partial downloads
                max_workers=4          
            )
            print("\n Success! Dataset download is fully complete.")
            break
            
        except Exception as e:
            # Catching the base Exception handles HuggingFaceHubRepoError,
            # HTTPError, or any token/quota timeouts seamlessly.
            print(f"\n Download interrupted or quota reached.")
            print(f"Error Details: {e}")
            print(f"Waiting {COOLDOWN_MINUTES} minutes before automatically retrying...")
            
            # Countdown timer in the terminal
            for remaining in range(COOLDOWN_MINUTES * 60, 0, -1):
                mins, secs = divmod(remaining, 60)
                print(f"Retrying in {mins:02d}:{secs:02d}...", end="\r")
                time.sleep(1)
                
            attempt += 1

if __name__ == "__main__":
    download_with_retry()