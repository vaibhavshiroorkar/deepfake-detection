"""Static text for the Streams section and for the two sections still locked.

Fusion and Explainability describe what will land there rather than stubbing
controls, so there is no half-working UI to mistake for the real thing. Streams
is live and its entry describes what the section *is*.

Copy rules for this file: a locked entry says what unlocks the page and what it
will contain, and nothing else. Claims that apply to the whole system (embeddings
rather than scores, no training in this dashboard) are stated once on the
Overview page.
"""

STREAMS = {
    "title": "Streams",
    "note": "Three feature streams feed fusion. Each has its own page, where one clip is walked "
            "through the model a stage at a time.",
    "views": [
        ("Visual", "Xception, EfficientNet-B0 and DINOv3 over the face-crop sequence. Configure "
                   "the temporal model, hidden size, embedding dim and freezing, then run a clip "
                   "and watch the activations at every backbone stage."),
        ("Lip-Sync", "AV-HuBERT over the mouth crop against Whisper over the audio, compared by "
                     "cross-attention. Produces the synchronisation-mismatch vector. Stage 4."),
        ("Emotion", "HSEmotions over the face against Wav2Vec2 over the voice, same mechanism. "
                    "Produces an affect-consistency mismatch vector. Stage 5."),
    ],
}

LIPSYNC_STREAM = {
    "title": "Lip-sync stream",
    "status": "The two encoders are Stage 4 and are not built. What this page shows today is the "
              "real input they will read and the mechanism that will compare them.",
    "models": [
        ("AV-HuBERT", "video encoder. Reads mouth-region motion into an embedding (Key/Value)"),
        ("Whisper", "audio encoder. Reads the audio track into an embedding (Query)"),
    ],
    "note": "Scaled dot-product cross-attention, audio attending to video, yields a "
            "synchronisation-mismatch vector. Nothing is transcribed; everything stays vectors.",
    "stage": 4,
    "query": "audio",
    "keyvalue": "mouth motion",
}

EMOTION_STREAM = {
    "title": "Emotion stream",
    "status": "The two encoders are Stage 5 and are not built. What this page shows today is the "
              "real input they will read and the mechanism that will compare them.",
    "models": [
        ("HSEmotions", "face-emotion encoder. Reads expression into an embedding (Key/Value)"),
        ("Wav2Vec2", "voice-emotion encoder. Reads vocal affect into an embedding (Query)"),
    ],
    "note": "Cross-attention, voice attending to face, yields an emotional-consistency mismatch "
            "vector for fusion to read.",
    "stage": 5,
    "query": "voice",
    "keyvalue": "facial expression",
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
