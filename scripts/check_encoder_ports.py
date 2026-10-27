"""Hold the AV-HuBERT and HSEmotion ports to the released code, and pick the lip crop.

Both encoders are run here through code written for this repository
(`streams/avhubert.py`, `streams/hsemotion.py`), because their releases need a
fairseq or a timm this environment cannot run. A port that loads every weight
can still compute something else, so each is compared with the original on the
same inputs:

    export     (here)  cached FakeAVCeleb clips to lips, audio and face crops
    reference  (the environment scripts/setup_encoder_reference.sh builds)
                       the official AV-HuBERT and HSEmotion on those inputs,
                       plus python_speech_features filterbanks on the raw audio
    compare    (here)  the ports on the same inputs, against the reference

`tune-crop` (here) chooses where the lip crop sits in the cached mouth view. For
each candidate it scores how well AV-HuBERT's lips-only and audio-only features
agree on genuine training clips when aligned versus when the audio is shifted
by half a second, and keeps the crop that separates the two best. No test or
validation clip is read.

    uv run python scripts/check_encoder_ports.py export
    python scripts/check_encoder_ports.py reference \
        --av-hubert ~/avhubert-ref/av_hubert --fairseq ~/avhubert-ref/fairseq-src
    uv run python scripts/check_encoder_ports.py compare
    uv run python scripts/check_encoder_ports.py tune-crop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROGRAM = Path("runs/program-20260906")
DEFAULT_DIR = Path("runs/encoder-ports")


def _clips(manifest: Path, count: int, *, genuine_only: bool, seed: int):
    import csv

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if genuine_only:
        rows = [row for row in rows if row["clip_fake"] in ("0", "False", "false")]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    return [rows[i] for i in order[: count * 3]]


def _load_views(rows, count: int):
    from deepfake_detection.data.cache_index import read_cache_index
    from deepfake_detection.views.cache_store import CacheStore

    index = read_cache_index(PROGRAM / "cache-index.csv")
    store = CacheStore(Path("."))
    out = []
    for row in rows:
        path = index.get(row["clip_id"])
        if path is None:
            continue
        prepared = store.load(
            path, views=("visual_view", "sync_video_view", "sync_audio_view")
        )
        if prepared.sync_video_view is None or prepared.sync_audio_view is None:
            continue
        if prepared.visual_view is None:
            continue
        out.append(prepared)
        if len(out) == count:
            break
    return out


def export(arguments: argparse.Namespace) -> int:
    import torch

    from deepfake_detection.streams.avhubert import audio_features, lip_frames

    rows = _clips(PROGRAM / "split" / "train-usable.csv", 6, genuine_only=False, seed=7)
    prepared = _load_views(rows, 6)
    mouth = torch.from_numpy(np.stack([p.sync_video_view for p in prepared]))
    wave = torch.from_numpy(np.stack([p.sync_audio_view for p in prepared]))
    lips = lip_frames(mouth).numpy()
    port_audio = audio_features(wave).numpy()
    # Two frames from each clip, as HSEmotion would see them.
    faces = np.concatenate([p.visual_view[[0, 8]] for p in prepared])
    arguments.dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.dir / "inputs.npz",
        lips=lips.astype(np.float32),
        wave=wave.numpy().astype(np.float32),
        port_audio=port_audio.astype(np.float32),
        faces=faces.astype(np.float32),
    )
    print(f"{len(prepared)} clips -> {arguments.dir / 'inputs.npz'}")
    return 0


def reference(arguments: argparse.Namespace) -> int:
    """Runs where fairseq and timm 0.9 are installed. Imports nothing from this
    repository, so it needs only the official code and the inputs file."""
    from argparse import Namespace

    import torch
    import torch.nn.functional as F
    from python_speech_features import logfbank

    sys.path[:0] = [str(arguments.fairseq), str(arguments.av_hubert)]
    from fairseq import checkpoint_utils, utils

    utils.import_user_module(Namespace(user_dir=str(arguments.av_hubert / "avhubert")))

    inputs = np.load(arguments.dir / "inputs.npz")
    lips = torch.from_numpy(inputs["lips"])
    frames = lips.shape[2]

    # The official audio path: logfbank, stack four frames, layer-norm.
    official_audio = []
    for wave in inputs["wave"]:
        feats = logfbank(wave, samplerate=16000).astype(np.float32)
        feats = torch.from_numpy(_stack(feats, 4))
        feats = F.layer_norm(feats, feats.shape[1:])
        official_audio.append(feats.T[:, :frames])
    official_audio = torch.stack(official_audio)

    from dataclasses import fields

    from fairseq import tasks
    from omegaconf import OmegaConf

    # load_model_ensemble_and_task, unrolled: the released checkpoint names
    # task fields (input_modality) that later versions of the code dropped, so
    # its config is pruned to the fields the current task declares.
    state = checkpoint_utils.load_checkpoint_to_cpu(str(arguments.checkpoint))
    task_cfg = state["cfg"]["task"]
    task_class = tasks.TASK_DATACLASS_REGISTRY[task_cfg["_name"]]
    known = {f.name for f in fields(task_class)} | {"_name"}
    pruned = {k: v for k, v in OmegaConf.to_container(task_cfg).items() if k in known}
    task = tasks.setup_task(OmegaConf.create(pruned))
    task.load_state_dict(state["task_state"])
    # fairseq 0.12's wav2vec2 encoder reads fields AV-HuBERT's config predates.
    # These values select the layout the checkpoint was trained with: one
    # positional convolution and plain transformer layers.
    from avhubert.hubert import AVHubertConfig

    # The dataclass defaults first (relu_type and others the checkpoint never
    # wrote), then the checkpoint's own values over them.
    saved = OmegaConf.to_container(state["cfg"]["model"])
    declared = {f.name for f in fields(AVHubertConfig)}
    model_cfg = OmegaConf.merge(
        OmegaConf.structured(AVHubertConfig),
        {k: v for k, v in saved.items() if k in declared},
    )
    OmegaConf.set_struct(model_cfg, False)
    for key, value in {
        "required_seq_len_multiple": 1,
        "pos_conv_depth": 1,
        "layer_type": "transformer",
        "checkpoint_activations": False,
        "offload_activations": False,
        "min_params_to_wrap": 100_000_000,
        "max_positions": 100_000,
    }.items():
        if key not in model_cfg:
            model_cfg[key] = value
    from avhubert.hubert import AVHubertModel

    # Built directly rather than through task.build_model, which would merge
    # the config back into the dataclass and reject the fields added above.
    model = AVHubertModel(model_cfg, task.cfg, task.dictionaries)
    model.load_state_dict(state["model"], strict=True)
    model = model.eval()
    with torch.no_grad():

        def run(video, audio):
            source = {"video": video, "audio": audio}
            features, _ = model.extract_finetune(source=source, padding_mask=None)
            return features.numpy()

        both = run(lips, official_audio)
        lips_only = run(lips, None)
        audio_only = run(None, official_audio)

    import timm

    faces = torch.from_numpy(inputs["faces"])
    expression = torch.load(arguments.hsemotion, map_location="cpu", weights_only=False)
    expression.eval()
    with torch.no_grad():
        logits = expression(faces).numpy()

    np.savez(
        arguments.dir / "reference.npz",
        official_audio=official_audio.numpy(),
        both=both,
        lips_only=lips_only,
        audio_only=audio_only,
        hsemotion_logits=logits,
        versions=np.array([torch.__version__, timm.__version__]),
    )
    print(f"reference -> {arguments.dir / 'reference.npz'}")
    return 0


def _stack(feats: np.ndarray, order: int) -> np.ndarray:
    """`AVHubertDataset.stacker`, copied verbatim from the official loader."""
    feat_dim = feats.shape[1]
    if len(feats) % order != 0:
        res = order - len(feats) % order
        res = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res], axis=0)
    return feats.reshape((-1, order, feat_dim)).reshape(-1, order * feat_dim)


def compare(arguments: argparse.Namespace) -> int:
    import torch

    from deepfake_detection.paths import AVHUBERT_CHECKPOINT, HSEMOTION_WEIGHTS
    from deepfake_detection.streams.avhubert import load_avhubert
    from deepfake_detection.streams.hsemotion import build_hsemotion

    inputs = np.load(arguments.dir / "inputs.npz")
    ref = np.load(arguments.dir / "reference.npz")
    print("reference torch, timm:", list(ref["versions"]))
    failures = 0

    def report(name, ours, theirs, tolerance):
        nonlocal failures
        error = float(np.abs(ours - theirs).max())
        scale = float(np.abs(theirs).max())
        ok = error <= tolerance * max(scale, 1.0)
        failures += not ok
        print(
            f"{name:28} max abs error {error:.2e} (scale {scale:.2f}) "
            f"{'ok' if ok else 'MISMATCH'}"
        )

    frames = inputs["lips"].shape[2]
    report(
        "filterbank features",
        inputs["port_audio"][:, :, :frames],
        ref["official_audio"],
        1e-4,
    )

    model = load_avhubert(AVHUBERT_CHECKPOINT)
    lips = torch.from_numpy(inputs["lips"])
    audio = torch.from_numpy(ref["official_audio"])
    with torch.no_grad():
        report(
            "AV-HuBERT lips and audio",
            model(video=lips, audio=audio).numpy(),
            ref["both"],
            1e-3,
        )
        report("AV-HuBERT lips only", model(video=lips).numpy(), ref["lips_only"], 1e-3)
        report(
            "AV-HuBERT audio only", model(audio=audio).numpy(), ref["audio_only"], 1e-3
        )

    expression = build_hsemotion(HSEMOTION_WEIGHTS, with_classifier=True)
    with torch.no_grad():
        logits = expression(torch.from_numpy(inputs["faces"])).numpy()
    report("HSEmotion logits", logits, ref["hsemotion_logits"], 1e-3)
    agree = (logits.argmax(1) == ref["hsemotion_logits"].argmax(1)).mean()
    print(f"HSEmotion top class agrees on {agree:.0%} of faces")
    return 1 if failures else 0


def tune_crop(arguments: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F

    from deepfake_detection.paths import AVHUBERT_CHECKPOINT
    from deepfake_detection.streams.avhubert import (
        audio_features,
        lip_frames,
        load_avhubert,
    )

    rows = _clips(
        PROGRAM / "split" / "train-usable.csv",
        arguments.clips,
        genuine_only=True,
        seed=arguments.seed,
    )
    prepared = _load_views(rows, arguments.clips)
    mouth = torch.from_numpy(np.stack([p.sync_video_view for p in prepared]))
    wave = torch.from_numpy(np.stack([p.sync_audio_view for p in prepared]))
    device = arguments.device
    model = load_avhubert(AVHUBERT_CHECKPOINT).to(device)
    # Half a second, 8000 samples, rolled: the same sound, out of step.
    shifted = torch.roll(wave, shifts=8000, dims=1)
    with torch.no_grad():
        frames = mouth.shape[1]
        sound = audio_features(wave.to(device))[:, :, :frames]
        late = audio_features(shifted.to(device))[:, :, :frames]
        heard = model(audio=sound)
        heard_late = model(audio=late)
        results = []
        for crop in arguments.crops:
            for center in arguments.centers:
                seen = []
                for start in range(0, len(mouth), 8):
                    lips = lip_frames(
                        mouth[start : start + 8].to(device), crop=crop, center_y=center
                    )
                    seen.append(model(video=lips))
                seen = torch.cat(seen)
                aligned = F.cosine_similarity(seen, heard, dim=-1).mean(1)
                off = F.cosine_similarity(seen, heard_late, dim=-1).mean(1)
                # Paired: the fraction of clips that agree better aligned.
                wins = float((aligned > off).float().mean())
                gap = float((aligned - off).mean())
                results.append((wins, gap, crop, center))
                print(
                    f"crop {crop:.2f} center {center:.2f}  aligned beats shifted "
                    f"on {wins:.0%} of clips, mean gap {gap:+.4f}",
                    flush=True,
                )
    best = max(results)
    print(
        f"best: crop {best[2]:.2f}, center {best[3]:.2f} "
        f"({best[0]:.0%} of {len(mouth)} genuine training clips)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "reference", "compare", "tune-crop"):
        command = commands.add_parser(name)
        command.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    reference_parser = commands.choices["reference"]
    reference_parser.add_argument("--av-hubert", type=Path, required=True)
    reference_parser.add_argument("--fairseq", type=Path, required=True)
    reference_parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/avhubert/base_vox_iter5.pt")
    )
    reference_parser.add_argument(
        "--hsemotion",
        type=Path,
        default=Path("models/hsemotion/enet_b0_8_best_afew.pt"),
    )
    tune = commands.choices["tune-crop"]
    tune.add_argument("--clips", type=int, default=96)
    tune.add_argument("--seed", type=int, default=17)
    tune.add_argument("--device", default="cuda")
    tune.add_argument("--crops", type=float, nargs="+", default=[0.45, 0.6, 0.75, 1.0])
    tune.add_argument("--centers", type=float, nargs="+", default=[0.55, 0.62, 0.7])
    arguments = parser.parse_args(argv)
    handler = {
        "export": export,
        "reference": reference,
        "compare": compare,
        "tune-crop": tune_crop,
    }[arguments.command]
    return handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
