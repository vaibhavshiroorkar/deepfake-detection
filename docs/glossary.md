# Shared Glossary & Prerequisites

**Owner:** Research workstream.

**Canonical living copy:** the "Research" doc in the team's Google Drive research folder — <https://drive.google.com/drive/folders/1rWDoC-Qm73uaeEU8-x_svNoLr3s3fnHP>. That copy keeps evolving (survey abstracts, math, etc.); this file is a synced Markdown snapshot of the glossary sections for anyone working in the repo.

---

## Understanding

### What is a deepfake?

Media (image, video, audio, etc.) that has been altered by AI to show something that never happened in reality:

1. **Face swapped** onto another person's body to frame them.
2. **Mouth movement** changed to make them appear to say words they never said.
3. **Cloned voice** speaking sentences the real person never spoke.

### What are we looking at?

Some fakes are simple — the whole face is swapped, which gives plenty of information to check. But harder fakes make only a **tiny change** that keeps the majority of the media real.

An example is the **lip-sync fake**: only the person's mouth is altered to match a fake voice track. Everything else — features, background, lighting — stays real. This kind of fake often cannot be detected from the video alone or the audio alone; we have to look at the **mismatch between them** and evaluate them together instead of standalone.

### What can we use and why?

We use a **supervised binary classification** model. It learns to make decisions by looking at labelled examples during training rather than following fixed rules. The dataset is split into three stages. The final output is a probability **between 0 and 1**, produced by a **sigmoid** function, representing how likely the clip is fake.

1. **Training** — feed the model thousands of clips labelled real/fake. It figures out the pattern itself, updating its parameters as it sees the data.
2. **Validation** — a held-out set the model does not learn from. Periodically during training we check against it to see how the model does on unseen data, which lets us tune hyperparameters and catch overfitting.
3. **Testing** — after training is complete, we feed brand-new clips the model has never seen and measure its predictions **with no further change to parameters or hyperparameters**.

### Why multiple models instead of one?

A single model has blind spots. Take the lip-sync fake: only the mouth is altered, everything else is real, so a model that only looks at pixels calls it real — because visually it *is* real. The alteration isn't in the pixels; it's the mismatch between mouth movement and sound. A visual-only model cannot detect it.

So instead of one all-rounder, we use multiple models that are each good at catching different kinds of fakes. If one is blind to a trick, another can cover it. Each of these is a **stream**.

### What is a stream, and the types we use?

In ML, a stream (often called a pipeline) is a dedicated path that processes one specific type of data from start to finish.

How it works:

1. **Single input** — takes exactly one data type (e.g. just audio, or just video frames).
2. **Isolated processing** — runs that data through its own independent neural network layers.
3. **Feature extraction** — converts raw data into a mathematical summary called an embedding vector.
4. **Independent output** — produces its own prediction/feature map before talking to any other part of the system.

Types:

- **Visual streams** — look at video frames only. They catch fakes where the pixels are wrong: blending errors around a swapped face, colour mismatch, artifacts left by the AI. Examples: **Xception**, **EfficientNet**, **DINOv3**.
- **Cross-modal streams** — take audio and video *together* to catch the mismatch between them, using **cross-attention on embeddings** (no transcription or lip-reading-to-text anywhere; everything is vectors).
  - **Lip-sync stream** — a video encoder embeds mouth-region motion (Key/Value), an audio encoder embeds the audio track (Query), and cross-attention compares them. Mismatch → suspicion score rises.
  - **Emotion stream** — a face-emotion encoder embeds facial expression (Key/Value), a voice-emotion encoder embeds vocal affect (Query), and cross-attention compares them. Mismatch → suspicion score rises.

### What is fusion?

Each stream produces its own embedding vector, so we have multiple independent views of the same clip — but the system needs one final answer. **Fusion** combines them. This project uses **feature-level fusion**: all stream embeddings are concatenated into one long vector, passed through a small MLP (a few fully-connected layers), and squashed to a `[0,1]` fake-probability by a **sigmoid**.

This is a deliberate choice over the simpler alternative, **late (score-level) fusion** — where each stream would output its own probability and those probabilities get averaged, or combined by a small learned model (logistic regression). Late fusion is easier to build and its learned weights are directly interpretable ("how much does the system trust each stream"); feature-level fusion is more powerful but loses that particular interpretability — which is why an **ablation** (running subsets of streams and comparing the fused result) matters more here, not less.

---

## Concepts

- **Parameters** — the values the model itself learns. It continuously updates them using optimization to minimize prediction error. Examples: **weights**, **biases**.
- **Hyperparameters** — external configuration set manually *before* a training run; fixed during the run, changed only between runs. Examples:
  - **Learning rate** — how much the model adjusts its parameters (e.g. weights) in response to prediction errors during training.
  - **Batch size** — how many clips the model looks at together before updating its parameters once.
  - **Epoch** — one full pass of the model through the entire training dataset.
- **Overfitting** — the model gets too comfortable with its training data and underperforms on unseen data, because it memorized specific training examples instead of learning the general pattern.
- **Underfitting** — even worse: the model wasn't trained well enough even on its own dataset, so it underperforms on both seen and unseen data.
- **Loss function** — a number saying how wrong the model is after a guess (higher = more loss). Measured per prediction during training; it's the thing the model actively tries to minimize. Examples: Mean Squared Error, Mean Absolute Error, **Binary Cross-Entropy (Log Loss)**, Categorical Cross-Entropy.

### Metrics (measuring how good the model is)

- **Accuracy** — how often the model is correct (e.g. 90 out of 100 clips). Misleading on imbalanced data.
- **AUC-ROC** — for binary classification, measures the model's ability to distinguish the classes (0 vs 1). 0.5 is worst (a coin flip), 1.0 is best. Threshold-free; our primary metric.
- **LogLoss** — measures the *certainty* of predictions and punishes confidently-wrong ones harder. E.g. predicting 0.99-fake on a real clip is punished more than 0.55-fake on a real clip.
- **Precision** — of the clips the model flagged as fake, how many actually were. E.g. 40 flagged, 30 truly fake → 75%. Low precision = more false alarms.
- **Recall** — of the clips that were actually fake, how many the model caught. E.g. 50 real fakes, 35 caught → 70%. Low recall = the model misses fakes.
- **F1 Score** — the balance between precision and recall, a single score from 0 (worst) to 1 (best). Precision 75% + recall 70% → F1 ≈ 0.724.
- **Confusion matrix** — a table evaluating a classifier by comparing predictions against actual labels, revealing exactly where it succeeds and where it makes mistakes.
- **EER (Equal Error Rate)** — the threshold where the model's False Acceptance Rate equals its False Rejection Rate.

---

## Reading index

Maintained in full on the Drive doc; summarized here for orientation.

- **Surveys:** Hashmi et al. (2024); Khan, Khan & Ahmad (2025).
- **Datasets:** FaceForensics++ (Rössler et al. 2019); FakeAVCeleb (Khalid et al. 2022); Celeb-DF (Li et al. 2020); Deepfake-Eval-2024 (Chandra et al. 2026).
- **Models:** Xception (Chollet 2016); EfficientNet; DINOv3 (Siméoni et al. 2025).
- **Methods (implemented):** Vaswani et al. (2017), cross-attention; Chung & Zisserman (2016, SyncNet), lip-sync embedding lineage; Mittal et al. (2020), emotion mismatch concept.
- **Methods (read-only comparison):** Zhou & Lim (2021); AVFF (Oorloff et al. 2024); Bohacek & Farid (2024) — their transcription-based semantic lip-sync method is not implemented here; our lip-sync stream uses embeddings and cross-attention instead.

> **Survey Paper 1 & 2 outlines** and the math/novelty sections are in progress on the Drive doc.
