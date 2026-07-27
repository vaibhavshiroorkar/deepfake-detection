"""Static architecture text for the Streams / Fusion / Explainability pages.

All three pages are currently locked — they describe what lands there rather
than stubbing controls, so there is no half-working UI to mistake for the real
thing. The machinery behind the Streams page (dashboard/lib/stream_pages.py and
stream_ui.py, both still unit-tested) is intact and unreferenced, so unlocking
it is a matter of calling the render functions again.
"""

STREAMS = {
    "title": "Streams",
    "status": "Locked while the visual stream is re-validated. Five-point face alignment "
              "changed the cached pixels, so the precached splits have to be rebuilt and "
              "EfficientNet-B0 re-measured against its AUC 0.994 bar before the model "
              "boxes here mean anything.",
    "note": "Three feature streams feed fusion. Each one emits a 256-d clip embedding and "
            "never a standalone score, which is what lets fusion learn interactions between "
            "them rather than averaging opinions.",
    "views": [
        ("Visual", "EfficientNet-B0, Xception and DINOv2 over the face-crop sequence — each a "
                   "configurable box (temporal model, hidden size, embedding dim, freeze) with "
                   "Train, which emits the background-trainer command, and Run, which "
                   "forward-passes one clip to check shapes, device and speed."),
        ("Lip-Sync", "AV-HuBERT over the mouth crop against Whisper over the audio, compared by "
                     "cross-attention — the synchronisation-mismatch vector. Stage 4."),
        ("Emotions", "HSEmotions over the face against Wav2Vec2 over the voice, same mechanism — "
                     "an affect-consistency mismatch vector. Stage 5."),
    ],
}

LIPSYNC_STREAM = {
    "title": "Lip-sync stream",
    "status": "Not built yet — Stage 4. This page scaffolds its model box.",
    "models": [
        ("AV-HuBERT", "video encoder — reads mouth-region motion into an embedding (Key/Value)"),
        ("Whisper", "audio encoder — reads the audio track into an embedding (Query)"),
    ],
    "note": "Scaled dot-product cross-attention (audio attends to video) yields a "
            "synchronization-mismatch vector. No transcription anywhere — everything is vectors.",
}

EMOTION_STREAM = {
    "title": "Emotion stream",
    "status": "Not built yet — Stage 5. This page scaffolds its model box.",
    "models": [
        ("HSEmotions", "face-emotion encoder — reads expression into an embedding (Key/Value)"),
        ("Wav2Vec2", "voice-emotion encoder — reads vocal affect into an embedding (Query)"),
    ],
    "note": "Cross-attention (voice attends to face) yields an emotional-consistency "
            "mismatch vector — a fixed-size feature for fusion, not a standalone score.",
}

FUSION = {
    "title": "Fusion",
    "status": "Not built yet — Stage 6, when the fusion MLP is trained.",
    "note": "Feature-level fusion (NOT score averaging): each enabled stream's clip "
            "embedding is projected to common_dim=256, written to the feature store, then "
            "concatenated and passed through an MLP + sigmoid for the fake probability. "
            "Which streams to include is a Stage-7 ablation decision.",
}

EXPLAINABILITY = {
    "title": "Explainability",
    "status": "Not built yet — Stage 10, after the streams and fusion are trained.",
    "views": [
        ("Grad-CAM", "where each visual backbone looks on a frame when it calls fake"),
        ("Embedding shift", "which stream's embedding moves most on which manipulation type"),
        ("Per-category / per-method", "accuracy broken down by FakeAVCeleb category and method"),
    ],
}
