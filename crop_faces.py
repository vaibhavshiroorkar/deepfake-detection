import os
import sys
import argparse
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

# Ensure local FFmpeg bin directory is in PATH
PROJECT_DIR = Path(__file__).parent.resolve()
BIN_DIR = PROJECT_DIR / "bin"
if BIN_DIR.exists():
    os.environ["PATH"] = str(BIN_DIR) + os.pathsep + os.environ.get("PATH", "")

# Try importing dependencies
try:
    import torch
    from facenet_pytorch import MTCNN
    from tqdm import tqdm
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please make sure you have run setup_env.py and activated the virtual environment.")
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

def process_video(video_path, output_dir, frame_stride=1, confidence_threshold=0.90, margin=0.2):
    """
    Processes the input video, detects faces in frames based on stride,
    crops and resizes faces to 224x224, and saves them to the output directory.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return
        
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
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Loaded Video: {video_path.name}")
    print(f"Total Frames: {total_frames} | FPS: {fps:.2f} | Frame Stride: {frame_stride}")
    
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
    parser = argparse.ArgumentParser(description="Crop and resize faces from a video to 224x224.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save cropped face images.")
    parser.add_argument("--stride", type=int, default=5, help="Process every Nth frame (default: 5 to avoid duplicate faces).")
    parser.add_argument("--conf", type=float, default=0.90, help="Confidence threshold for face detection (default: 0.90).")
    parser.add_argument("--margin", type=float, default=0.25, help="Padding margin percentage around face box (default: 0.25).")
    
    args = parser.parse_args()
    process_video(
        video_path=args.video,
        output_dir=args.output,
        frame_stride=args.stride,
        confidence_threshold=args.conf,
        margin=args.margin
    )
