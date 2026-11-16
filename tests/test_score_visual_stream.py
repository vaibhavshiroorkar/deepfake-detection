"""score_visual_stream.py rebuilds a stream from its history so a saved checkpoint loads into it."""

import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_visual_stream.py"


def _module():
    spec = importlib.util.spec_from_file_location("score_visual_stream", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HISTORY = {"config": {"stream": "visual-test", "model": {
    "backbone": "resnet18", "global_pool": "avg", "temporal": "gru", "temporal_hidden": 32,
    "common_dim": 16, "frozen_backbone": True}}}


def test_rebuilt_stream_matches_the_trained_shape_and_loads_its_weights():
    module = _module()
    trained = module.stream_from_history(HISTORY)
    rebuilt = module.stream_from_history(HISTORY)
    rebuilt.load_state_dict(trained.state_dict())          # strict: same architecture
    frames = torch.randn(1, 4, 3, 64, 64)
    trained.eval()
    rebuilt.eval()
    logit_a, _ = trained(frames)
    logit_b, _ = rebuilt(frames)
    assert torch.allclose(logit_a, logit_b)
