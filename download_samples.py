import os
import urllib.request
import sys
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent.resolve()
SAMPLES_DIR = PROJECT_DIR / "sample_videos"

# List of sample videos to download
VIDEO_URLS = {
    "face_walking.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4",
    "face_walking_pause.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4",
    "head_pose_detection.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female-and-male.mp4",
    "classroom.mp4": "https://github.com/intel-iot-devkit/sample-videos/raw/master/classroom.mp4"
}

def report_progress(filename, block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100, (read_so_far * 100) // total_size)
        sys.stdout.write(f"\rDownloading {filename}: {percent}% ({read_so_far // 1024}KB / {total_size // 1024}KB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloading {filename}: {read_so_far // 1024}KB")
        sys.stdout.flush()

def download_samples():
    SAMPLES_DIR.mkdir(exist_ok=True)
    print(f"Downloading practice video clips into {SAMPLES_DIR}...")
    
    for name, url in VIDEO_URLS.items():
        dest_path = SAMPLES_DIR / name
        if dest_path.exists():
            print(f"Sample clip '{name}' already exists.")
            continue
        
        print(f"\nStarting download for {name}...")
        try:
            urllib.request.urlretrieve(
                url, 
                dest_path, 
                reporthook=lambda b_num, b_size, t_size: report_progress(name, b_num, b_size, t_size)
            )
            print(f"\nSuccessfully downloaded {name}.")
        except Exception as e:
            print(f"\nError downloading {name}: {e}")
            
    print("\nAll downloads complete!")

if __name__ == "__main__":
    download_samples()
