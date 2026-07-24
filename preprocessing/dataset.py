"""
Stage 1 - PyTorch Dataset/DataLoader.

Reads one of data/{train,val,test}.csv (built by build_splits.py), and for
each row calls extract_clip() to get 16 face crops + their aligned audio
windows (cached to disk after the first access, see extract_clip.py's
docstring). Returns exactly what every later stage consumes:

    face_crop_sequence: [16, 3, 224, 224] float32, ImageNet-normalized
    audio:              [16, window_samples] float32 waveform windows
    label:              scalar int, 1 = fake / 0 = real (clip-level)

Normalization: the visual backbones (Xception/EfficientNet/DINOv2, Stage 2+)
are all ImageNet-pretrained in timm, so we normalize with ImageNet
mean/std here rather than per-stream later -- one place to get it right.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# See extract_clip.py for why: make the `preprocessing` package importable
# regardless of how this script is launched.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Run: uv sync --extra cpu (or --extra cu130 for GPU), see README.md")
    sys.exit(1)

from preprocessing.extract_clip import extract_clip
from preprocessing.ops.constants import NUM_FRAMES, AUDIO_SR, IMAGENET_MEAN, IMAGENET_STD

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# A VISUAL stream sees only frames, so its label is the video track's
# authenticity: FakeVideo-* -> fake, RealVideo-* -> real (even
# RealVideo-FakeAudio, whose fakeness is audio-only). See docs/stage-2-plan.md.
VISUAL_FAKE_TYPES = ("FakeVideo-RealAudio", "FakeVideo-FakeAudio")


class ClipDataset(Dataset):
    def __init__(self, split: str, label_mode: str = "clip"):
        """
        split: 'train', 'val', or 'test' -- matches data/{split}.csv.
        label_mode:
            'clip'   -> the overall clip label (1 = fake) straight from the
                        manifest. Used by cross-modal streams (Stages 4-5) and
                        by fusion (Stage 6), which DO see audio.
            'visual' -> the video-track label derived from manipulation_type,
                        for visual-only streams (Stages 2-3).
        """
        if label_mode not in ("clip", "visual"):
            raise ValueError(f"label_mode must be 'clip' or 'visual', got '{label_mode}'")
        manifest_path = DATA_DIR / f"{split}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} not found -- run preprocessing/build_splits.py first."
            )
        self.manifest = pd.read_csv(manifest_path)
        self.split = split
        self.label_mode = label_mode
        # video_path in the manifest is relative to data/ (see audit_dataset.py)
        self.video_root = DATA_DIR
        # Precompute the label array once, honoring label_mode. Exposed via
        # labels() so a WeightedRandomSampler can be built without re-reading.
        self._labels = self._compute_labels()

    def _compute_labels(self) -> np.ndarray:
        if self.label_mode == "visual":
            return self.manifest["manipulation_type"].isin(VISUAL_FAKE_TYPES).astype(np.int64).to_numpy()
        return self.manifest["label"].astype(np.int64).to_numpy()

    def labels(self) -> np.ndarray:
        """Per-sample labels (in manifest order) under the active label_mode."""
        return self._labels

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        video_path = self.video_root / row["video_path"]
        clip_id = row["clip_id"]

        try:
            extracted = extract_clip(video_path, clip_id)
        except Exception as e:
            # A single unreadable/corrupt clip shouldn't crash an entire
            # training run -- surface it clearly and let the caller decide
            # (DataLoader will raise; a training loop can catch per-batch).
            raise RuntimeError(f"Failed to load clip_id={clip_id} at {video_path}: {e}") from e

        frames = extracted["frames"].astype(np.float32) / 255.0   # [16, 224, 224, 3] in [0,1]
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        frames = np.transpose(frames, (0, 3, 1, 2))                # -> [16, 3, 224, 224]

        face_seq = torch.from_numpy(frames.copy()).float()
        audio = torch.from_numpy(extracted["audio"].copy()).float()  # [16, window_samples]
        label = torch.tensor(int(self._labels[idx]), dtype=torch.long)

        return face_seq, audio, label, clip_id


def make_dataloader(split: str, batch_size: int = 4, shuffle: bool = None,
                    num_workers: int = 0, label_mode: str = "clip") -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"
    dataset = ClipDataset(split, label_mode=label_mode)
    # clip_id (a str) can't go through torch's default collate as a tensor,
    # so batch it as a plain list explicitly.
    def collate(batch):
        face_seqs, audios, labels, clip_ids = zip(*batch)
        return torch.stack(face_seqs), torch.stack(audios), torch.stack(labels), list(clip_ids)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate)


if __name__ == "__main__":
    # Stage-1 verification artifact: load one batch, print shapes.
    loader = make_dataloader("train", batch_size=4)
    face_seq, audio, label, clip_ids = next(iter(loader))
    print(f"face_crop_sequence shape: {tuple(face_seq.shape)}  (expected [B, {NUM_FRAMES}, 3, 224, 224])")
    print(f"audio shape:              {tuple(audio.shape)}  (expected [B, {NUM_FRAMES}, window_samples @ {AUDIO_SR}Hz])")
    print(f"label shape:              {tuple(label.shape)}  values={label.tolist()}")
    print(f"clip_ids:                 {clip_ids}")
    print(f"face_seq dtype/range:     {face_seq.dtype}, min={face_seq.min():.3f}, max={face_seq.max():.3f}")
    print(f"audio dtype/range:        {audio.dtype}, min={audio.min():.3f}, max={audio.max():.3f}")
