---
title: Deepfake Generalization
emoji: 🎞️
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
pinned: false
short_description: Why deepfake detectors fail on new data, in your browser
---

# Deepfake generalization demo

This page runs the project's best single face model entirely in the browser.
The video never leaves the visitor's device.

It mirrors the served Python path: 16 uniformly sampled frames, MTCNN face
detection, the greedy IoU face track, a 0.2-margin square crop, then
EfficientNet-B0 with a GRU. A familiarity check embeds three 32-frame windows
with DINOv3 and withholds the verdict when the clip sits too far from the
genuine training footage.

## Files

- `pipeline.js`: the port. Each function names the Python it mirrors.
- `worker.js`: loads the ONNX models with onnxruntime-web and runs the pipeline.
- `app.js`, `index.html`, `style.css`: the page.
- `export.py`: writes the ONNX models and `gate.json` from the demo runtime.

The models load from
[vaibhavshiroorkar/deepfake-generalization-demo](https://huggingface.co/vaibhavshiroorkar/deepfake-generalization-demo/tree/main/browser).

## Matching the Python results

On the same decoded frames, the port matches Python on detections, tracks and
familiarity distance, and the logit to within 0.011 on six of seven test
clips. The seventh differs by 0.16 because two MTCNN boxes tie at a score of
1.0 and NumPy's sort breaks the tie differently. Browser decoding adds small
differences for HD video, whose colours browsers convert with BT.709 where
OpenCV uses BT.601. Verdicts matched on every test clip.

Code, paper and results:
[github.com/vaibhavshiroorkar/deepfake-detection](https://github.com/vaibhavshiroorkar/deepfake-detection).
