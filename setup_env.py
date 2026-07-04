import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path

# Paths
PROJECT_DIR = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_DIR / ".venv"
BIN_DIR = PROJECT_DIR / "bin"
TEMP_DIR = PROJECT_DIR / "temp"

# Virtual environment python and pip paths
if os.name == "nt":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
    VENV_PIP = VENV_DIR / "Scripts" / "pip.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"
    VENV_PIP = VENV_DIR / "bin" / "pip"

# FFmpeg source URL for Windows
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

def run_command(args, check=True):
    print(f"Running: {' '.join(str(x) for x in args)}")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"Error executing command: {result.stderr}")
        sys.exit(result.returncode)
    return result

def setup_venv():
    if not VENV_DIR.exists():
        print("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", str(VENV_DIR)])
        print("Virtual environment created.")
    else:
        print("Virtual environment already exists.")

    # Upgrade pip
    print("Upgrading pip inside virtual environment...")
    run_command([str(VENV_PYTHON), "-m", "pip", "install", "--upgrade", "pip"])

def download_ffmpeg():
    if (BIN_DIR / "ffmpeg.exe").exists() and (BIN_DIR / "ffprobe.exe").exists():
        print("FFmpeg binaries already exist in bin/ folder.")
        return

    print("Downloading static FFmpeg build for Windows (GitHub BtbN)...")
    BIN_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    zip_path = TEMP_DIR / "ffmpeg.zip"
    
    # Download file with progress report
    def report_progress(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = min(100, (read_so_far * 100) // total_size)
            sys.stdout.write(f"\rDownloading: {percent}% ({read_so_far // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rDownloading: {read_so_far // (1024*1024)}MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(FFMPEG_URL, zip_path, reporthook=report_progress)
    print("\nDownload complete. Extracting zip file...")

    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR)
    
    # Locate ffmpeg.exe and ffprobe.exe in extracted folder
    ffmpeg_exe = None
    ffprobe_exe = None
    for root, dirs, files in os.walk(TEMP_DIR):
        for file in files:
            if file.lower() == "ffmpeg.exe":
                ffmpeg_exe = Path(root) / file
            elif file.lower() == "ffprobe.exe":
                ffprobe_exe = Path(root) / file

    if ffmpeg_exe and ffprobe_exe:
        shutil.copy(ffmpeg_exe, BIN_DIR / "ffmpeg.exe")
        shutil.copy(ffprobe_exe, BIN_DIR / "ffprobe.exe")
        print(f"Successfully installed FFmpeg to {BIN_DIR}")
    else:
        print("Error: Could not find ffmpeg.exe or ffprobe.exe in the downloaded zip file.")
        sys.exit(1)

    # Clean up temp files
    print("Cleaning up temporary directory...")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

def install_python_packages():
    print("Installing python packages in virtual environment...")
    
    # Install PyTorch (CPU-only version) first to save space and ensure compatibility
    print("Installing PyTorch (CPU build) and torchvision...")
    run_command([
        str(VENV_PYTHON), "-m", "pip", "install", 
        "torch", "torchvision", 
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    
    # Install remaining packages
    packages = ["opencv-python", "facenet-pytorch", "ffmpeg-python", "tqdm"]
    print(f"Installing packages: {', '.join(packages)}")
    run_command([str(VENV_PYTHON), "-m", "pip", "install"] + packages)
    print("Python packages installed successfully.")

def verify_setup():
    print("\n--- Verifying Installation ---")
    
    # Verify Python & Packages
    test_code = """
import cv2
import torch
import torchvision
from facenet_pytorch import MTCNN
import ffmpeg
print("Successfully imported cv2, torch, torchvision, MTCNN, and ffmpeg!")
print("PyTorch Version:", torch.__version__)
print("OpenCV Version:", cv2.__version__)
"""
    result = subprocess.run([str(VENV_PYTHON), "-c", test_code], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout.strip())
    else:
        print("Failed package import verification:")
        print(result.stderr)
        sys.exit(1)
        
    # Verify FFmpeg execution
    ffmpeg_path = BIN_DIR / "ffmpeg.exe"
    if ffmpeg_path.exists():
        ffmpeg_res = subprocess.run([str(ffmpeg_path), "-version"], capture_output=True, text=True)
        if ffmpeg_res.returncode == 0:
            first_line = ffmpeg_res.stdout.split('\n')[0]
            print(f"FFmpeg verified: {first_line}")
        else:
            print("FFmpeg execution failed.")
            sys.exit(1)
    else:
        print("ffmpeg.exe not found in bin/ directory.")
        sys.exit(1)

    print("\nEnvironment setup is complete and fully functional!")

if __name__ == "__main__":
    setup_venv()
    download_ffmpeg()
    install_python_packages()
    verify_setup()
