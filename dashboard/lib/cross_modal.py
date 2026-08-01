"""The Lip-Sync and Emotion pages: two encoders compared by cross-attention.

Both streams are the same shape. One modality asks the questions (Query), the
other supplies the evidence (Key and Value), and how cleanly the evidence answers
is the signal. Lip-sync sends audio against mouth motion; emotion sends voice
against facial expression. One renderer serves both, differing by the spec passed
in.

Neither pair of encoders exists yet: AV-HuBERT and Whisper are Stage 4,
HSEmotions and Wav2Vec2 are Stage 5. So the page shows the two things that are
real today, and is explicit that the third is neither:

  1. The inputs. The mouth crops, face crops and audio windows are produced by
     preprocessing.ops, the same code the batch pipeline runs. These are exactly
     the tensors the encoders will read.
  2. The encoders, greyed out, named, with the stage that lands them.
  3. The attention mechanism, computed over RANDOM projections of those inputs.
     It shows the arithmetic and the shapes. It measures nothing. Every caption
     on it says so, because an attention heatmap with a bright diagonal looks
     like a finding whether or not it is one.
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from preprocessing.ops import audio as AF
from preprocessing.ops.constants import AUDIO_SR

# The width the demo projects both modalities into. Small on purpose: it is a
# stand-in for an encoder's output, not a hyperparameter anyone should tune here.
DEMO_DIM = 64
DEMO_SEED = 0


def random_projection(sequence: np.ndarray, dim: int, seed: int) -> np.ndarray:
    """[T, F] -> [T, dim] through a fixed random Gaussian matrix.

    Stands in for an encoder that has not been built. Fixed seed so the picture
    is stable between reruns; random weights so it cannot be mistaken for
    something that learned anything.
    """
    flat = np.asarray(sequence, dtype=np.float32).reshape(len(sequence), -1)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0 / np.sqrt(flat.shape[1]), size=(flat.shape[1], dim))
    return (flat - flat.mean()) @ matrix.astype(np.float32)


def scaled_dot_product_attention(query: np.ndarray, key: np.ndarray,
                                 value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """softmax(QK^T / sqrt(d_k)) V, returning (weights, output).

    The divisor is not cosmetic: the dot product of two d-dimensional vectors
    with unit-variance components has variance d, so without it the softmax
    saturates towards one-hot as the width grows.
    """
    scores = query @ key.T / np.sqrt(query.shape[1])
    shifted = scores - scores.max(axis=1, keepdims=True)     # overflow guard only
    weights = np.exp(shifted)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, weights @ value


def diagonal_mass(weights: np.ndarray, band: int = 1) -> float:
    """Fraction of attention mass within `band` steps of the diagonal.

    In a genuine recording, sound arrives at a near-fixed offset from the
    articulation that produced it, so a trained model's attention concentrates
    near the diagonal. The number is defined here so the eventual Stage-4 stream
    and this page measure the same thing.
    """
    rows, cols = weights.shape
    r = np.arange(rows)[:, None]
    c = np.arange(cols)[None, :]
    return float(weights[np.abs(r * (cols / rows) - c) <= band].sum() / rows)


def clip_settings(st) -> tuple[int, float]:
    """(frames, audio window seconds) as chosen on the Preprocessing page."""
    return (int(st.session_state.get("pp_n_frames", 16)),
            float(st.session_state.get("pp_window", 0.35)))


def decode_inputs(video_path: str, num_frames: int, window_sec: float, conf: float = 0.9,
                  margin: float = 0.2) -> dict:
    """The real preprocessed tensors both cross-modal streams read.

    Faces and mouths come from one detect call, exactly as the Visual tab of the
    Preprocessing page does it, so the crops on this page are the crops the
    pipeline stores.
    """
    from dashboard.lib import media

    duration, fps = media.frame_meta(video_path)
    timestamps = media.sample_timestamps(duration, num_frames, window_sec)
    frames = media.decode_frames(video_path, timestamps)

    detector, _device = media.get_detector()
    faces, mouths, detected = [], [], []
    for frame in frames:
        face, mouth, found = media.detect_face_and_mouth(frame, detector, conf, margin)
        faces.append(face)
        mouths.append(mouth)
        detected.append(found)

    raw, native_sr = media.decode_audio(video_path)
    if raw.size:
        wav = AF.resample(AF.downmix(raw), native_sr, AUDIO_SR)
        windows = AF.extract_windows(wav, AUDIO_SR, timestamps, window_sec)
    else:
        windows = np.zeros((num_frames, int(window_sec * AUDIO_SR)), dtype=np.float32)

    return {"faces": faces, "mouths": mouths, "windows": windows,
            "detected": sum(detected), "timestamps": timestamps,
            "duration": duration, "fps": fps, "has_audio": bool(raw.size)}


def render(st, spec: dict, key: str, video_source: str):
    """The whole page body. `video_source` is "mouths" or "faces"."""
    from dashboard.lib import stream_pages, stream_ui, trace_ui

    st.title(spec["title"])
    st.caption(spec["note"])
    st.info(spec["status"])

    video_path = stream_pages.render_inherited_clip(st)
    num_frames, window_sec = clip_settings(st)

    st.subheader("1 · Inputs")
    st.caption("Produced by `preprocessing.ops`, the same code the batch pipeline runs. These "
               "are the tensors the encoders will read once they exist.")
    if video_path is None:
        st.stop()

    with st.spinner("Decoding the clip…"):
        data = decode_inputs(video_path, num_frames, window_sec)

    visual = data[video_source]
    visual_label = ("Mouth crops, 96 pixels, derived from the two mouth-corner landmarks"
                    if video_source == "mouths" else
                    "Face crops, 224 pixels, margin-padded bounding box")

    with st.container(border=True):
        left, right = st.columns([1, 2])
        left.markdown(f"**Video** ({spec['keyvalue']})")
        left.metric("Faces detected", f"{data['detected']}/{num_frames}")
        left.caption(f"`[{num_frames}, 3, {visual[0].shape[0]}, {visual[0].shape[1]}]`")
        trace_ui.show_frames(right, visual, visual_label)

    with st.container(border=True):
        left, right = st.columns([1, 2])
        windows = data["windows"]
        left.markdown(f"**Audio** ({spec['query']})")
        left.metric("Windows", f"{windows.shape[0]} x {windows.shape[1]}")
        left.caption(f"One {window_sec:.2f}s window per frame timestamp, at {AUDIO_SR} Hz.")
        if not data["has_audio"]:
            right.warning("This clip carries no audio stream, so the windows below are silence.")
        right.pyplot(trace_ui.matrix_fig(
            np.abs(windows).T, f"Audio windows {windows.shape} (magnitude)",
            "window, one per frame", "sample"))

    st.subheader("2 · Encoders")
    st.caption(f"Not built. Stage {spec['stage']}.")
    for name, role in spec["models"]:
        stream_ui.render_disabled_model_box(
            st, name, role, f"Lands in Stage {spec['stage']}.", key=f"{key}_{name}")

    st.subheader("3 · Cross-attention")
    st.latex(r"\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^{\top}}"
             r"{\sqrt{d_k}}\right)V")
    st.warning(
        f"**This is the mechanism, not a measurement.** The encoders above do not exist, so the "
        f"{spec['query']} and {spec['keyvalue']} sequences below are projected into "
        f"{DEMO_DIM} dimensions by a fixed *random* matrix. The arithmetic and the shapes are "
        f"real. The attention pattern says nothing whatsoever about whether this clip is "
        f"synchronised, and will not until Stage {spec['stage']} lands.")

    query = random_projection(data["windows"], DEMO_DIM, DEMO_SEED)
    evidence = random_projection(np.stack([v.astype(np.float32) / 255.0 for v in visual]),
                                 DEMO_DIM, DEMO_SEED + 1)
    weights, out = scaled_dot_product_attention(query, evidence, evidence)

    left, right = st.columns([2, 3])
    with left:
        st.code(
            f"Q  {query.shape}   <- {spec['query']}\n"
            f"K  {evidence.shape}   <- {spec['keyvalue']}\n"
            f"V  {evidence.shape}\n"
            f"QK^T/sqrt(d)  {weights.shape}\n"
            f"attended      {out.shape}\n"
            f"mismatch vec  ({out.shape[1]},)  -> fusion",
            language="text")
        st.metric("Mass near the diagonal", f"{diagonal_mass(weights):.2f}",
                  help="Fraction of each row's attention within one step of the diagonal. A "
                       "trained model on a genuine clip should concentrate here. With random "
                       "projections it is close to what uniform attention would give.")
        st.caption("The stream hands fusion the attended vector, not a decision. Nothing is "
                   "transcribed and no emotion label is produced; everything stays vectors.")
    with right:
        st.pyplot(trace_ui.attention_fig(
            weights, "Attention weights (random projections)",
            f"{spec['keyvalue']} step", f"{spec['query']} step"))
