# Learning path

This handbook builds the knowledge needed to read the platform and judge its
research claims. Chapters 00 through 04 are current. Chapters 05 through 14
are planned.

## Who this is for

The reader should know basic Python, Git, functions, classes, and command-line
use. Prior machine learning, signal processing, and research methods are not
required.

After the live foundation chapters, you should be able to:

- State the research question without turning a detector score into a fact.
- Trace visual and audio tensor shapes through the current code.
- Calculate the current window sizes and synchronization offsets.
- Explain why the split groups clips by source identity.
- Find the tests that protect each implemented contract.

## Reading order

Follow this 15-chapter sequence. Allow 45 to 75 minutes for each foundation
chapter. Allow 60 to 90 minutes for each later implementation chapter once it
is available.

| Chapter | Topic | Status | Main outcome |
|---|---|---|---|
| 00 | Learning path | Current | Plan a repeatable study cycle. |
| 01 | Problem and research question | Current | Define the claim and its limits. |
| 02 | Deep learning foundations | Current | Read tensors, loss, gradients, and training code. |
| 03 | Audio-video foundations | Current | Reason about samples, frames, and time. |
| 04 | Data and leakage | Current | Audit labels and identity-safe splits. |
| 05 | Preprocessing pipeline | Planned | Trace decoding, tracking, crops, and cache gates. |
| 06 | Visual branch | Planned | Trace frame features and temporal aggregation. |
| 07 | Audio branch | Planned | Trace speech tokens and attentive pooling. |
| 08 | Sync branch | Planned | Trace correspondence and offset learning. |
| 09 | Fusion and calibration | Planned | Explain calibrated late fusion and abstention. |
| 10 | Training system | Planned | Run and inspect staged training. |
| 11 | Evaluation and statistics | Planned | Read metrics, intervals, and paired comparisons. |
| 12 | Inference and dashboard | Planned | Trace one prediction and its provenance. |
| 13 | Reproducing the project | Planned | Repeat the frozen workflow. |
| 14 | Viva preparation | Planned | Defend choices, limits, and negative results. |

Do not skip chapter 04. A correct model trained on a leaking split supports a
wrong conclusion.

## How to study each chapter

Use this practice cycle:

1. Read the learning goals and required background.
2. Copy the equations by hand and define every symbol.
3. Recalculate the worked example without looking at the answer.
4. Open each linked source file and follow the named code path.
5. Run the supporting fixture tests. Fixtures use small synthetic records, not
   private datasets or model artifacts.
6. Complete one exercise by changing only a local fixture.
7. Answer the viva questions aloud in two minutes each.

For every implementation chapter, run its named tests after reading. A passing
test shows the repository still matches the explanation. It does not prove a
research hypothesis.

## Learning checks

Record progress with evidence, not a simple read marker.

- Concept check: define the key terms without the chapter.
- Calculation check: reproduce every numeric result.
- Code check: name the input, output, and failure path of each linked function.
- Test check: run the focused tests and explain one assertion.
- Research check: state what the code or test cannot establish.

After chapters 01 through 04, run:

```powershell
uv run pytest tests\test_documentation.py tests\test_training.py `
  tests\test_views.py tests\test_manifest.py tests\test_protocols.py -v
```

A useful progress note is specific: "I can derive the two-sample BCE gradient
and explain why `video_fake` differs from `clip_fake`." A test score or reading
percentage alone does not show that understanding.
