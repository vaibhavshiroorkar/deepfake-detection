import math

import pytest

torch = pytest.importorskip("torch")

from deepfake_detection.training.selection import BestEpoch  # noqa: E402


def _model(value: float) -> torch.nn.Module:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


def test_keeps_the_highest_score_and_restores_its_weights() -> None:
    selection = BestEpoch(patience=5)
    model = _model(0.0)
    for epoch, score in enumerate((0.6, 0.9, 0.7), start=1):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        selection.update(epoch=epoch, score=score, model=model)

    assert selection.restore(model) == 2
    assert float(model.weight.detach()) == 2.0


def test_stops_after_patience_epochs_without_improvement() -> None:
    selection = BestEpoch(patience=2)
    model = _model(0.0)

    assert not selection.update(epoch=1, score=0.8, model=model)
    assert not selection.update(epoch=2, score=0.8, model=model)
    assert selection.update(epoch=3, score=0.7, model=model)


def test_an_undefined_score_keeps_the_first_epoch_until_a_real_one_arrives() -> None:
    selection = BestEpoch(patience=5)
    model = _model(1.0)
    selection.update(epoch=1, score=math.nan, model=model)
    with torch.no_grad():
        model.weight.fill_(2.0)
    selection.update(epoch=2, score=math.nan, model=model)

    assert selection.restore(model) == 1
    assert float(model.weight.detach()) == 1.0

    selection.update(epoch=3, score=0.6, model=_model(3.0))
    restored = _model(0.0)
    assert selection.restore(restored) == 3
    assert float(restored.weight) == 3.0
