# Xception: Architecture Math

**Owner:** Research workstream
**Paper:** Chollet, F. (2017). *Xception: Deep Learning with Depthwise Separable Convolutions.* CVPR 2017. <https://arxiv.org/abs/1610.02357>
**Status:** built now, first visual stream (Stage 2).

> Draft here, keep notation consistent with the other writeups — merge into one combined architecture document once all are drafted.

## 1. Depthwise separable convolution: the formula

<!-- Standard convolution: one filter mixes space AND channels at once.
     Depthwise separable: per-channel spatial convolution (depthwise),
     then a 1x1 convolution across channels (pointwise). Write both as equations. -->

## 2. Parameter count: separable vs standard

<!-- Count parameters for a standard KxK conv with C_in -> C_out channels,
     then for depthwise + pointwise. Show the ratio and plug in real numbers
     (e.g. K=3, C_in=C_out=256) to make the saving concrete. -->

## 3. Entry / middle / exit flow structure

<!-- The three stages of the Xception network, what each does, and where
     the residual (skip) connections sit. A diagram helps. -->

## 4. Why this matters for deepfake detection

<!-- Why an artifact-focused CNN like this catches blending edges and
     colour inconsistencies. One short paragraph. -->

> Implementation note for later: in our pinned timm version the model is named
> `legacy_xception` (`timm.create_model("legacy_xception", ...)`) — the old
> `"xception"` name is deprecated.
