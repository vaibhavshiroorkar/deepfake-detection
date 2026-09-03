from deepfake_detection.dashboard.workflow import StepState, workflow_state


def test_empty_workflow_waits_for_video() -> None:
    state = workflow_state(
        has_upload=False, has_prepared=False, has_prediction=False
    )
    assert state.video is StepState.READY
    assert state.preprocessing is StepState.WAITING
    assert state.visual_model is StepState.READY
    assert state.prediction is StepState.WAITING


def test_prepared_video_unlocks_prediction() -> None:
    state = workflow_state(
        has_upload=True, has_prepared=True, has_prediction=False
    )
    assert state.video is StepState.COMPLETE
    assert state.preprocessing is StepState.COMPLETE
    assert state.visual_model is StepState.READY
    assert state.prediction is StepState.READY


def test_prediction_completes_the_workflow() -> None:
    state = workflow_state(
        has_upload=True, has_prepared=True, has_prediction=True
    )
    assert state.prediction is StepState.COMPLETE


def test_step_status_renders_the_workflow_state() -> None:
    import pytest

    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string(
        """
from deepfake_detection.dashboard.components import render_step_status
from deepfake_detection.dashboard.workflow import StepState

render_step_status(StepState.WAITING)
"""
    ).run()

    assert not app.exception
    assert '<div class="step-state waiting">waiting</div>' in [
        item.value for item in app.markdown
    ]
