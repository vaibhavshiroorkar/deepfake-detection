"""Static architecture text for the read-only Streams scaffolds (no compute)."""

VISUAL_STREAM = {
    "title": "Visual stream",
    "status": "Not trained yet — this page fills in after Stage 2/3 training.",
    "architecture": [
        "Backbones: Xception, EfficientNet, DINOv2 (config-driven, one at a time)",
        "Temporal model: BiLSTM over per-frame embeddings",
        "Output: one clip-level embedding (projected to common_dim=256 for fusion)",
        "Never sees audio — labels are the video track's authenticity.",
    ],
}

AUDIOVISUAL_STREAM = {
    "title": "Audiovisual stream",
    "status": "Not trained yet — this page fills in after Stage 4/5 training.",
    "architecture": [
        "Lip-sync: AV-HuBERT (video) + Whisper (audio), scaled dot-product cross-attention",
        "Emotion: HSEmotions (video) + Wav2Vec2 (audio), cross-attention",
        "Each outputs a fixed-size mismatch feature vector (not a standalone score)",
        "Cross-modal by construction — catches audio/video disagreement.",
    ],
}
