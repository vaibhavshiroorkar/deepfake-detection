"""The top line of an answer: a vision-language model says what the still shows and what the heatmap covers.

Qwen3-VL-2B-Instruct (Apache 2.0) runs on this computer; nothing is sent anywhere. It is asked only to
describe what is visible, never to judge real or fake: tested 11-19, a prompt that asked it to explain the
evidence made up findings ("unusual coloration") and contradicted the checks. The verdict stays the checks'.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
from pathlib import Path

from deepfake_detection.webapp import outside

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models" / "pretrained" / "qwen3-vl-2b-instruct"
SYSTEM = (
    "You describe images for a result page. Plain English, short. Never name or guess who a person is. "
    "Describe only what is visible. Do not judge whether anything is real or fake."
)
WITH_HEAT = (
    "Image 1 is a still from a video. Image 2 is the face from it with a heatmap; the warm (yellow and red) "
    "area is where a detector looked. Write exactly two sentences. Sentence 1: what image 1 shows (person, "
    "setting). Sentence 2: name the part of the face the warm area covers in image 2 as the suspicious "
    "area, and say what it looks like in image 1, for example 'The mouth and jaw look suspicious: the "
    "skin there is smoother than the rest of the face.' Say 'the video'. Never say 'heatmap', 'warm "
    "area', 'image 1' or 'image 2'."
)
STILL_ONLY = "This is a still from a video. In one sentence, say what it shows (person, setting)."
log = logging.getLogger(__name__)


class Describer:
    """Loads the model on first use, on the GPU only (on a processor it would take about a minute)."""

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.model_dir, self._model, self._processor = model_dir, None, None
        self._lock = threading.Lock()

    def available(self) -> bool:
        import torch

        device = os.environ.get("DDF_SCORER_DEVICE", "cpu")
        return (
            device.startswith("cuda")
            and torch.cuda.is_available()
            and self.model_dir.is_dir()
        )

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        with self._lock:
            if self._model is None:
                self._processor = AutoProcessor.from_pretrained(self.model_dir)
                self._model = (
                    AutoModelForImageTextToText.from_pretrained(
                        self.model_dir, dtype=torch.bfloat16
                    )
                    .to("cuda")
                    .eval()
                )

    def __call__(self, still, heat=None) -> str:
        import torch

        self._load()
        images = [{"type": "image", "image": still}] + (
            [{"type": "image", "image": heat}] if heat else []
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {
                "role": "user",
                "content": [
                    *images,
                    {"type": "text", "text": WITH_HEAT if heat else STILL_ONLY},
                ],
            },
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=70, do_sample=False)
        text = self._processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return tidy(" ".join(text.split()))


def tidy(text: str, most: int = 2) -> str:
    """At most `most` sentences, and none that repeats an earlier one's words (found 11-20: the model
    restated the heatmap sentence)."""
    import re

    kept, seen = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = set(re.findall(r"[a-z']+", sentence.lower()))
        if not words or any(len(words & s) / len(words) > 0.6 for s in seen):
            continue
        kept.append(sentence)
        seen.append(words)
        if len(kept) == most:
            break
    return " ".join(kept)


_describer = Describer()


def _image(data_url: str):
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert(
        "RGB"
    )


def describe(upload: Path, time: float, explain: dict) -> str | None:
    """The detector's hook: a still of the upload at `time` and the first cited heatmap, described."""
    from PIL import Image

    if not _describer.available():
        return None
    target = upload.with_name(upload.stem + "-describe.jpg")
    try:
        with Image.open(outside.frame(upload, time, target)) as still:
            frames = explain.get("frames") or []
            heat = _image(frames[0]["image"]) if frames else None
            return _describer(still.convert("RGB"), heat)
    finally:
        target.unlink(missing_ok=True)
