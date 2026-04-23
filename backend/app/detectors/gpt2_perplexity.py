"""GPT-2 perplexity scoring for AI-generated text detection.

Key insight: language models assign higher probability (lower perplexity) to
text produced by similar models. AI-generated prose is "unsurprising" to GPT-2,
resulting in low, uniform perplexity. Human writing is burstier — some phrases
are predictable, others are novel, producing higher variance.

Two complementary signals are extracted:
  1. Overall perplexity — lower ⇒ more likely AI-generated
  2. Per-sentence perplexity variance ("burstiness") — lower variance ⇒ AI

No fine-tuning is required; this uses the pretrained GPT-2 weights directly.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def compute_perplexity(
    text: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    stride: int = 512,
) -> float:
    """Sliding-window perplexity over the full text.

    Uses a stride of 512 tokens with context windows of up to 1024,
    so long documents are handled correctly without truncation.
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0]
    seq_len = input_ids.size(0)

    if seq_len < 2:
        return 0.0

    max_length = min(1024, model.config.n_positions)
    nlls: list[float] = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # only score the new tokens

        ids = input_ids[begin:end].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(ids, labels=ids)

        # Extract per-token losses for just the newly-seen tokens.
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        # Only count the non-overlapping tail.
        offset = max(0, token_losses.size(0) - target_len)
        nlls.extend(token_losses[offset:].cpu().tolist())

        prev_end = end
        if end >= seq_len:
            break

    if not nlls:
        return 0.0
    return float(math.exp(sum(nlls) / len(nlls)))


def compute_sentence_perplexities(
    sentences: list[str],
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> list[float]:
    """Per-sentence perplexity. Returns a list parallel to *sentences*."""
    ppls: list[float] = []
    for sent in sentences:
        tokens = tokenizer(sent, return_tensors="pt", truncation=True, max_length=1024)
        ids = tokens.input_ids.to(device)
        if ids.size(1) < 2:
            ppls.append(0.0)
            continue
        with torch.no_grad():
            loss = model(ids, labels=ids).loss
        ppls.append(float(math.exp(loss.item())))
    return ppls


def perplexity_signal(
    text: str,
    sentences: list[str],
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> tuple[float, str, dict]:
    """Returns (score, detail_string, raw_stats).

    score is 0..1 where higher = more suspicious (more likely AI).
    """
    overall_ppl = compute_perplexity(text, model, tokenizer, device)

    sent_ppls = compute_sentence_perplexities(sentences, model, tokenizer, device)
    valid_ppls = [p for p in sent_ppls if p > 0]

    if not valid_ppls:
        return 0.3, "Insufficient text for perplexity analysis.", {"overall_ppl": 0}

    ppl_mean = float(np.mean(valid_ppls))
    ppl_std = float(np.std(valid_ppls))
    ppl_cv = ppl_std / (ppl_mean + 1e-9)

    # --- Scoring ---
    # Low overall perplexity → AI-generated.  Thresholds calibrated on
    # informal GPT-2 measurements: human text typically 40–120 ppl,
    # GPT-3/4/Claude output typically 15–40 ppl on GPT-2.
    score = 0.0

    # Overall perplexity component (0–0.55)
    if overall_ppl < 20:
        score += 0.55
    elif overall_ppl < 35:
        score += 0.40
    elif overall_ppl < 55:
        score += 0.25
    elif overall_ppl < 80:
        score += 0.10

    # Per-sentence variance component (0–0.45)
    # AI text has very even per-sentence perplexity (low CV).
    if ppl_cv < 0.25:
        score += 0.45
    elif ppl_cv < 0.45:
        score += 0.30
    elif ppl_cv < 0.65:
        score += 0.15
    elif ppl_cv < 0.85:
        score += 0.05

    score = float(np.clip(score, 0.0, 1.0))

    if score > 0.55:
        detail = (
            f"GPT-2 perplexity {overall_ppl:.1f} (sentence CV {ppl_cv:.2f}). "
            "Text is highly predictable to a language model — consistent with AI generation."
        )
    elif score > 0.35:
        detail = (
            f"GPT-2 perplexity {overall_ppl:.1f} (sentence CV {ppl_cv:.2f}). "
            "Moderately predictable. Could be polished human writing or lightly edited AI output."
        )
    else:
        detail = (
            f"GPT-2 perplexity {overall_ppl:.1f} (sentence CV {ppl_cv:.2f}). "
            "Text surprises the model in ways typical of human writing."
        )

    stats = {
        "overall_perplexity": round(overall_ppl, 2),
        "sentence_perplexity_mean": round(ppl_mean, 2),
        "sentence_perplexity_std": round(ppl_std, 2),
        "sentence_perplexity_cv": round(ppl_cv, 3),
    }
    return score, detail, stats
