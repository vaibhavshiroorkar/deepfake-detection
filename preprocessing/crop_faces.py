import sys
import argparse
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

# Try importing dependencies
try:
    import torch
    from facenet_pytorch import MTCNN
    from tqdm import tqdm
    import ffmpeg
    import librosa
    import pandas as pd
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please make sure you've set up the environment (see README.md: uv sync --extra cpu, or --extra cu130 for GPU) and activated it.")
    sys.exit(1)

def crop_and_resize_face(frame_rgb, box, target_size=(224, 224), margin_percentage=0.2):
    """
    Crops the face from the frame using the bounding box, applying margin, 
    and resizes it to target_size. Handles boundary conditions properly.
    """
    height, width, _ = frame_rgb.shape
    x1, y1, x2, y2 = box
    
    # Calculate width and height of bounding box
    w = x2 - x1
    h = y2 - y1
    
    # Add margins to the box to capture full face/hairline
    margin_w = int(w * margin_percentage)
    margin_h = int(h * margin_percentage)
    
    x1_m = max(0, int(x1 - margin_w))
    y1_m = max(0, int(y1 - margin_h))
    x2_m = min(width, int(x2 + margin_w))
    y2_m = min(height, int(y2 + margin_h))
    
    # Crop the face region
    face_crop = frame_rgb[y1_m:y2_m, x1_m:x2_m]
    
    if face_crop.size == 0:
        return None
        
    # Convert face crop from RGB to BGR for OpenCV saving
    face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
    
    # Resize to target size using Cubic Interpolation for best quality
    resized_face = cv2.resize(face_crop_bgr, target_size, interpolation=cv2.INTER_CUBIC)
    return resized_face

def extract_audio(video_path, output_dir, sample_rate=16000):
    """
    Extracts the video's audio track to a mono WAV file via ffmpeg.
    Returns None (after printing a warning) if the video has no audio stream.
    """
    audio_path = output_dir / "audio.wav"
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(audio_path), ac=1, ar=sample_rate)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        print(f"Warning: audio extraction failed, continuing without audio sync.\n{stderr.strip()}")
        return None
    return audio_path

def sync_audio_to_frames(audio_duration_sec, sample_rate, frame_indices, fps, window_sec=0.35):
    """
    Maps each processed frame index to a centered audio sample window.

    A single frame's exact audio slice (1/fps seconds) is too short to carry
    usable signal for spectral features, so a fixed-duration window (default
    0.35s) is centered on each frame's timestamp instead. This is the shared
    frame<->audio mapping other streams (lip-sync, emotion) will reuse later,
    so keep this function's output format stable.

    Returns a list of dict rows: frame_idx, frame_time_sec, audio_start_sample,
    audio_end_sample, audio_start_sec, audio_end_sec.
    """
    half_window = window_sec / 2
    rows = []
    for frame_idx in frame_indices:
        frame_time_sec = frame_idx / fps
        start_sec = max(0.0, frame_time_sec - half_window)
        end_sec = min(audio_duration_sec, frame_time_sec + half_window)
        rows.append({
            "frame_idx": frame_idx,
            "frame_time_sec": round(frame_time_sec, 4),
            "audio_start_sample": int(start_sec * sample_rate),
            "audio_end_sample": int(end_sec * sample_rate),
            "audio_start_sec": round(start_sec, 4),
            "audio_end_sec": round(end_sec, 4),
        })
    return rows

def process_video(video_path, output_dir, frame_stride=1, confidence_threshold=0.90, margin=0.2,
                   sync_audio=True, audio_window_sec=0.35, sample_rate=16000):
    """
    Processes the input video, detects faces in frames based on stride,
    crops and resizes faces to 224x224, and saves them to the output directory.
    Also extracts the video's audio and syncs a centered audio window to each
    processed frame (see sync_audio_to_frames).
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        sys.exit(1)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize MTCNN
    # Run on GPU if CUDA is available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Initializing MTCNN model (weights will download on first run)...")
    mtcnn = MTCNN(keep_all=True, device=device)
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Loaded Video: {video_path.name}")
    print(f"Total Frames: {total_frames} | FPS: {fps:.2f} | Frame Stride: {frame_stride}")

    # Audio extraction + frame sync, done once up front (independent of face detection)
    if sync_audio:
        audio_path = extract_audio(video_path, output_dir, sample_rate=sample_rate)
        if audio_path is not None:
            audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
            audio_duration_sec = len(audio) / sample_rate
            frame_indices = range(0, total_frames, frame_stride)
            manifest_rows = sync_audio_to_frames(
                audio_duration_sec, sample_rate, frame_indices, fps, window_sec=audio_window_sec
            )
            manifest_path = output_dir / "audio_manifest.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            print(f"Audio extracted ({audio_duration_sec:.2f}s) and synced: "
                  f"{len(manifest_rows)} frame windows -> {manifest_path.name}")

    saved_faces_count = 0
    processed_frames_count = 0
    
    # Initialize tqdm progress bar
    pbar = tqdm(total=total_frames, desc="Processing Frames")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame based on stride
        if frame_idx % frame_stride == 0:
            processed_frames_count += 1
            
            # Convert BGR frame to RGB for MTCNN PIL image processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Detect face bounding boxes and confidence probabilities
            boxes, probs = mtcnn.detect(frame_pil)
            
            if boxes is not None:
                for face_idx, (box, prob) in enumerate(zip(boxes, probs)):
                    # Check confidence threshold
                    if prob is not None and prob >= confidence_threshold:
                        # Extract, crop, and resize face
                        face_img = crop_and_resize_face(
                            frame_rgb, box, target_size=(224, 224), margin_percentage=margin
                        )
                        
                        if face_img is not None:
                            # Save face to output folder
                            filename = f"face_frame{frame_idx:06d}_idx{face_idx}.jpg"
                            save_path = output_dir / filename
                            cv2.imwrite(str(save_path), face_img)
                            saved_faces_count += 1
        
        frame_idx += 1
        pbar.update(1)
        
    cap.release()
    pbar.close()
    
    print("\n--- Processing Complete ---")
    print(f"Total frames scanned: {processed_frames_count} / {total_frames}")
    print(f"Total face crops saved: {saved_faces_count}")
    print(f"Faces saved inside: {output_dir.resolve()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop and resize faces from a video to 224x224, "
                     "and sync a centered audio window to each processed frame."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save cropped face images.")
    parser.add_argument("--stride", type=int, default=5, help="Process every Nth frame (default: 5 to avoid duplicate faces).")
    parser.add_argument("--conf", type=float, default=0.90, help="Confidence threshold for face detection (default: 0.90).")
    parser.add_argument("--margin", type=float, default=0.25, help="Padding margin percentage around face box (default: 0.25).")
    parser.add_argument("--audio-window", type=float, default=0.35, help="Duration in seconds of the audio window centered on each frame (default: 0.35).")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate in Hz for extraction and sync (default: 16000).")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio extraction and frame-audio sync.")

    args = parser.parse_args()
    process_video(
        video_path=args.video,
        output_dir=args.output,
        frame_stride=args.stride,
        confidence_threshold=args.conf,
        margin=args.margin,
        sync_audio=not args.no_audio,
        audio_window_sec=args.audio_window,
        sample_rate=args.sample_rate,
    )
