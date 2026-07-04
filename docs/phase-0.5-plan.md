# Phase 0.5: Foundations — Day-by-Day Plan (Days 1 to 5)

The core idea: each of you owns one architecture's math start to finish. The person-to-architecture split runs alongside your normal role work, so the math gets written in parallel instead of one at a time.

Note on who owns what math: normally Research would write all the math, but to hit Day 5 we're splitting it three ways. So each person wears two hats this phase: their usual role (Research / Data / ML) plus one architecture writeup.

**Architecture ownership:**

- Person 1 (Research lead): Xception math
- Person 2 (Data lead): EfficientNet math
- Person 3 (ML lead): DINOv2 math

Xception goes to Research because it's the gentlest read and Research is already deepest in the papers. DINOv2 is the hardest, so it goes to the ML lead who's most comfortable with the training mechanics. EfficientNet sits in the middle, which fits the Data lead picking up a second hat.

---

## Day 1

**Person 1 (Research): Basics + Survey Paper 1**

- Half a day on the shared glossary: training/validation/test splits, loss functions, overfitting, and what AUC-ROC, accuracy, and LogLoss mean. This is for the whole team, so write it clearly.
- Start Hashmi et al. (2024): abstract, intro, and the detection-type breakdown (spatial, temporal, cross-modal).
- Done when: shared glossary exists, and Survey Paper 1's method types are outlined.

**Person 2 (Data): Tools + first face-cropping script**

- Get Python working with OpenCV, ffmpeg, and facenet-pytorch.
- Grab 3 to 5 short practice clips.
- Write a script that turns a video into a folder of cropped 224x224 faces.
- Done when: video in, folder of face crops out.

**Person 3 (ML): Repo, environment, GPU check**

- Set up the shared GitHub repo, kept lean:
  - `data/`
  - `preprocessing/`
  - `models/baseline/`
  - `evaluation/`
  - `notebooks/`
  - `docs/`
- Write `requirements.txt` with fixed versions (PyTorch, torchvision, torchaudio, fixed CUDA, OpenCV, librosa, ffmpeg-python, transformers, timm, facenet-pytorch).
- Confirm the GPU works with PyTorch and note VRAM.
- Done when: repo up, dependencies pinned, GPU confirmed.

---

## Day 2

**Person 1 (Research): Finish Survey Paper 1 + start Xception math**

- Finish Hashmi et al. (2024). Sort detection methods into types with 2 to 3 examples each. Pull the human-factors findings into their own section for the motivation angle.
- Start reading the Xception paper. Take notes on depthwise separable convolution (per-channel spatial convolution, then a 1x1 convolution across channels, and why it uses fewer parameters).
- Done when: Survey Paper 1 notes are clean, and Xception reading has started.

**Person 2 (Data): Datasets + access + start EfficientNet math**

- Shortlist datasets, FakeAVCeleb as primary. Request FakeAVCeleb (academic email) and Deepfake-Eval-2024 today.
- Note FakeAVCeleb's four categories (RealVideo-RealAudio, RealVideo-FakeAudio, FakeVideo-RealAudio, FakeVideo-FakeAudio) and log file sizes.
- Start reading the EfficientNet paper. Take notes on compound scaling (one coefficient balancing depth, width, and resolution).
- Done when: both access requests in, category note written, and EfficientNet reading started.

**Person 3 (ML): PyTorch practice exercise + start DINOv2 math**

- Build the tiny real/fake PyTorch classifier on Day 1's practice crops (mock labels are fine). Cover Dataset/DataLoader, a small model with a new head, training loop, validation loop. Comment it well so the others can read it.
- Start reading the DINOv2 paper. Take notes on the self-distillation idea (a student network learning to match a slowly-updated teacher, no labels).
- Done when: the notebook runs start to finish, and DINOv2 reading has started.

---

## Day 3

**Person 1 (Research): Xception math writeup + method papers**

- Write up the Xception math section properly: the depthwise separable convolution formula, why it cuts parameters, and the entry/middle/exit flow structure.
- Read Bohacek & Farid (2024) and Mittal et al. (2020) enough to summarize their inputs, mismatch scoring, and results. (These are shared reading, so coordinate with the others on splitting them if time is tight.)
- Done when: Xception math is a solid first draft, and both cross-modal papers are summarized.

**Person 2 (Data): Audio extraction + EfficientNet math writeup**

- Extend the script to pull audio with ffmpeg and sync it to video frames (match each frame to its audio window using fps and sample rate). This syncing gets reused everywhere later, so keep it clean.
- Write up the EfficientNet math section: the compound scaling coefficient and why it gave a strong accuracy-per-parameter tradeoff.
- Done when: one video gives both face crops and synced audio, and EfficientNet math is a solid first draft.

**Person 3 (ML): Team learns the notebook + DINOv2 math writeup**

- Walk the other two through the training notebook so all three can run and tweak it. Then add real metrics (accuracy, AUC-ROC) and a confusion matrix.
- Write up the DINOv2 math section: the self-distillation objective, multi-crop strategy, and why self-supervised features generalize better to unseen fakes.
- Done when: all three can run the notebook with metrics showing, and DINOv2 math is a solid first draft.

---

## Day 4

**Person 1 (Research): Polish Xception + consolidate the three sections**

- Tighten the Xception writeup based on your own second read.
- Start pulling all three architecture sections into one consistent document (same notation, same structure), so the professor sees one clean piece, not three mismatched styles.
- Done when: Xception is polished and the combined math doc is taking shape.

**Person 2 (Data): Stress-test pipeline + polish EfficientNet**

- Combine face and audio scripts into one preprocessing module with adjustable settings and a short README. Run it on 10 to 20 clips to find breakages (missed faces, multiple faces, corrupt files) and add error handling.
- Polish the EfficientNet writeup and hand it to Person 1 for the combined doc.
- Done when: one solid preprocessing module exists, and EfficientNet math is polished.

**Person 3 (ML): Fine-tune a real model + polish DINOv2**

- Swap the toy model for a real pretrained backbone from timm (start with ResNet before Xception, just to practice). Practice loading weights, swapping the head, freezing/unfreezing, and a short training run on practice crops.
- Polish the DINOv2 writeup and hand it to Person 1.
- Done when: a documented fine-tuning run exists with metrics, and DINOv2 math is polished.

---

## Day 5

**Person 1 (Research): Finalize the combined math doc + draft intro**

- Merge all three architecture sections into one clean, consistently formatted document ready to show the professor.
- Draft the report's Intro and Related Work from the survey notes and method summaries.
- Done when: one combined math doc is review-ready, and Intro/Related Work drafts exist.

**Person 2 (Data): Lock the pipeline + storage plan**

- Freeze the preprocessing module's interface (inputs, outputs, folder naming, manifest format). Write the manifest spec (clip ID to label and manipulation type). Draft the full-dataset scaling plan for when FakeAVCeleb lands.
- Do a final read of your EfficientNet section for errors before the review.
- Done when: pipeline frozen, manifest spec written, scaling plan drafted.

**Person 3 (ML): Full dry run + final math check**

- Run the whole slice end to end on practice data: preprocessing in, model fine-tuned, metrics and confusion matrix out. This becomes the template for the first real stream.
- Do a final read of your DINOv2 section before the review.
- Done when: the full practice pipeline runs end to end, ready for real data.

---

**End of Day 5, ready for the professor:**

- One combined, consistently formatted math document covering all three architectures (Xception, EfficientNet, DINOv2)
- Draft Intro and Related Work
- A working preprocessing pipeline and a proven end-to-end training slice
