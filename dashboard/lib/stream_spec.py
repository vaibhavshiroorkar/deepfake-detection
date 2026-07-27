"""Static architecture text for the Streams / Fusion / Explainability pages.

The visual page builds real models; the cross-modal stream tabs are read-only
scaffolds, and the Fusion and Explainability pages are locked, for work still in
the plans (no compute).
"""

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
