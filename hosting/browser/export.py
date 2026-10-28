"""Export the served visual-best path to ONNX for the in-browser demo.

Run against the demo runtime, so the weights are the ones the web app serves:

    $env:DDF_PROJECT_ROOT = "runs/hosting/<demo>/runtime"
    uv run --with onnx --with onnxruntime --with onnxscript python hosting/browser/export.py out/

Writes MTCNN's three nets, the visual-best branch, the DINOv3 encoder the
familiarity gate was fitted on, and the gate's density as JSON. Each ONNX file
is checked against PyTorch on random input before the script reports success.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

OPSET = 18


class _Logit(nn.Module):
    """The branch returns a dataclass; ONNX needs a plain tensor."""

    def __init__(self, branch: nn.Module) -> None:
        super().__init__()
        self.branch = branch

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.branch(frames).logits


def _export(model, example, path, names, dynamic) -> None:
    model.eval()
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["input"],
        output_names=names,
        dynamic_axes=dynamic,
        opset_version=OPSET,
        dynamo=False,
    )


def _check(model, example, path) -> float:
    import onnxruntime

    with torch.inference_mode():
        expected = model(example)
    expected = expected if isinstance(expected, tuple) else (expected,)
    session = onnxruntime.InferenceSession(str(path))
    actual = session.run(None, {"input": example.numpy()})
    return max(
        float(np.abs(a - e.numpy()).max())
        for a, e in zip(actual, expected, strict=True)
    )


def main(out: Path) -> int:
    from facenet_pytorch import MTCNN

    from deepfake_detection.inference import authenticity, frozen_features
    from deepfake_detection.webapp.engines import BEST_SINGLE, DENSITY_PATH, Registry

    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    report = {}

    mtcnn = MTCNN(keep_all=True, device="cpu")
    nets = {
        "pnet": (mtcnn.pnet, torch.rand(1, 3, 120, 160) * 2 - 1, ["reg", "prob"]),
        "rnet": (mtcnn.rnet, torch.rand(4, 3, 24, 24) * 2 - 1, ["reg", "prob"]),
        "onet": (
            mtcnn.onet,
            torch.rand(4, 3, 48, 48) * 2 - 1,
            ["reg", "landmarks", "prob"],
        ),
    }
    for name, (net, example, outputs) in nets.items():
        dynamic = {"input": {0: "batch"}} | {o: {0: "batch"} for o in outputs}
        if name == "pnet":
            dynamic["input"] |= {2: "height", 3: "width"}
            dynamic["reg"] |= {2: "h", 3: "w"}
            dynamic["prob"] |= {2: "h", 3: "w"}
        path = out / f"{name}.onnx"
        _export(net, example, path, outputs, dynamic)
        report[name] = _check(net, example, path)

    registry = Registry(device="cpu")
    branch = _Logit(dict(registry.ensemble().models)[BEST_SINGLE])
    example = torch.randn(1, 16, 3, 224, 224)
    path = out / "visual-best.onnx"
    _export(branch, example, path, ["logit"], None)
    report["visual-best"] = _check(branch, example, path)

    density = authenticity.load(DENSITY_PATH)
    encoder, size = frozen_features.load_encoder(density.encoder, "cpu")
    example = torch.rand(4, 3, size, size)
    path = out / "dinov3.onnx"
    _export(encoder, example, path, ["features"], {"input": {0: "batch"}})
    report["dinov3"] = _check(encoder, example, path)

    gate = {
        "encoder": density.encoder,
        "size": size,
        "windows": frozen_features.WINDOWS,
        "window_length": frozen_features.WINDOW_LENGTH,
        "threshold": density.threshold,
        "reference": density.reference,
        "mean": density.mean.tolist(),
        "components": density.components.tolist(),
        "centre": density.centre.tolist(),
        "inverse": density.inverse.tolist(),
    }
    (out / "gate.json").write_text(json.dumps(gate), encoding="utf-8")

    for name, error in report.items():
        print(f"{name}: max abs difference {error:.2e}")
    worst = max(report.values())
    if worst > 1e-3:
        print(f"ONNX differs from PyTorch by {worst:.2e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1])))
