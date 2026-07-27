"""Static text for the locked Streams, Fusion and Explainability pages.

All three describe what will land there rather than stubbing controls, so there
is no half-working UI to mistake for the real thing. The machinery behind the
Streams page (stream_pages.py and stream_ui.py, both still unit-tested) is intact
and unreferenced, so unlocking it means calling the render functions again.

Copy rules for this file, since it is the source for three pages: say what
unlocks the page and what it will contain, and nothing else. Claims that apply to
the whole system (embeddings rather than scores, no training in this dashboard)
are stated once on the Overview page.
"""

STREAMS = {
    "title": "Streams",
    "status": "Locked while the visual stream is re-validated. Five-point face alignment changed "
              "the cached pixels, so the precached splits have to be rebuilt and EfficientNet-B0 "
              "re-measured against its AUC 0.994 bar before the model boxes here mean anything.",
    "note": "Three feature streams feed fusion.",
    "views": [
        ("Visual", "EfficientNet-B0, Xception and DINOv2 over the face-crop sequence. Each is a "
                   "configurable box (temporal model, hidden size, embedding dim, freeze) with "
                   "Train, which emits the background-trainer command, and Run, which "
                   "forward-passes one clip to check shapes, device and speed."),
        ("Lip-Sync", "AV-HuBERT over the mouth crop against Whisper over the audio, compared by "
                     "cross-attention. Produces the synchronisation-mismatch vector. Stage 4."),
        ("Emotions", "HSEmotions over the face against Wav2Vec2 over the voice, same mechanism. "
                     "Produces an affect-consistency mismatch vector. Stage 5."),
    ],
}

LIPSYNC_STREAM = {
    "title": "Lip-sync stream",
    "status": "Not built yet. Stage 4. This page scaffolds its model box.",
    "models": [
        ("AV-HuBERT", "video encoder. Reads mouth-region motion into an embedding (Key/Value)"),
        ("Whisper", "audio encoder. Reads the audio track into an embedding (Query)"),
    ],
    "note": "Scaled dot-product cross-attention, audio attending to video, yields a "
            "synchronisation-mismatch vector. Nothing is transcribed; everything stays vectors.",
}

EMOTION_STREAM = {
    "title": "Emotion stream",
    "status": "Not built yet. Stage 5. This page scaffolds its model box.",
    "models": [
        ("HSEmotions", "face-emotion encoder. Reads expression into an embedding (Key/Value)"),
        ("Wav2Vec2", "voice-emotion encoder. Reads vocal affect into an embedding (Query)"),
    ],
    "note": "Cross-attention, voice attending to face, yields an emotional-consistency mismatch "
            "vector for fusion to read.",
}

FUSION = {
    "title": "Fusion",
    "status": "Not built yet. Stage 6, when the fusion MLP is trained.",
    "note": "Feature-level fusion, not score averaging. Each enabled stream's clip embedding is "
            "projected to common_dim=256, written to the feature store, then concatenated and "
            "passed through an MLP and a sigmoid for the fake probability. Which streams to "
            "include is a Stage-7 ablation decision.",
    "views": [
        ("Stream selection", "which streams are concatenated into the fusion input"),
        ("Fusion MLP", "hidden sizes, dropout, and the resulting input dimension (streams x 256)"),
        ("Ablation view", "fused metrics across stream subsets, the Stage-7 result"),
    ],
}

EXPLAINABILITY = {
    "title": "Explainability",
    "status": "Not built yet. Stage 10, after the streams and fusion are trained.",
    "note": "Explainability needs a trained model to explain, so these views populate after fusion "
            "is trained.",
    "views": [
        ("Grad-CAM", "where each visual backbone looks on a frame when it calls fake"),
        ("Embedding shift", "which stream's embedding moves most on which manipulation type"),
        ("Per-category and per-method", "accuracy by FakeAVCeleb category and method"),
    ],
}
