# Gemini-generated image detection: data and provenance feasibility (2026-10-27)

Status: assessment and proposal only. No Gemini data was downloaded, no model was trained, nothing was evaluated. The user's
instruction of 2026-10-27 evening: do not train a Gemini-specific model yet; first a data and provenance feasibility assessment and a
preregistered proposal; document the blocker if suitable data cannot be obtained. Visible logo or watermark detection stays excluded
(D-077); SynthID verification is an external Google service with no local detector (docs/FEASIBILITY_AND_TARGET_ARCHITECTURE.md) and
is not part of this proposal.

## 1. What the project already has (checked in the repository)

| Question | Answer | Evidence |
|---|---|---|
| Gemini-generated images on disk | None | grep of docs/, src/, scripts/, tests/ for "gemini"; DATA_LOG has no Gemini row; docs/PRODUCT_REDESIGN_AND_MODEL_ROADMAP.md section 9 lists no modern text-to-image generator on disk before E-028 |
| Any experiment on Gemini | None. The nearest measured generator is Google Imagen 3, one of the 21 generators of the Community Forensics-Eval subset (203 generated against 688 paired real images) | E-028, E-029 result files |
| Imagen 3, frozen T-I (E-028) | AUC 0.7691 [0.7352, 0.8020], missed 0.8719 of the generated images at the frozen threshold | data/_results/modern/e028/e028_results.json, gen:Imagen3 |
| Imagen 3, E-029 head (seed 0) | AUC 0.8646 [0.8360, 0.8917], missed 0.0345, but flagged 0.5654 of the paired real images | data/_results/modern/e029/e029_results.json, gen:Imagen3 |
| Real-image controls | COCO, FFHQ, VISION (Community Forensics-Small, training split); LAION, RAISE, COCO, ImageNet, FFHQ (Community Forensics-Eval, test only) | DATA_LOG I6, I7 |
| Pipeline for training and evaluation | Yes: scripts/modern_image (features, linear heads, one-look evaluation with guards), dfd.eval | E-029 |
| Near-duplicate tooling | No perceptual-hash package installed (imagehash absent); a DCT hash can be written with numpy and OpenCV, which are present | checked 2026-10-27 |

Reading of the existing evidence: Imagen 3 is not Gemini, and Gemini image models are not one model. The E-029 pattern (good ranking,
threshold that flags most real photographs) is the failure the user wants to avoid, and nothing about it is specific to Gemini.

## 2. Candidate external data (checked 2026-10-27 from the dataset cards; nothing downloaded)

| Dataset | What the card states | Problems for this experiment |
|---|---|---|
| bitmind/nano-banana (Hugging Face) | 9,457 images, "Nano-Banana (Google Gemini 2.5 Flash Image Preview)"; licence tag MIT; columns id, image, format, mode, width, height, uploadtime; visible rows PNG 1024 x 1024; one train split | No prompts, no generation dates (only upload times, 2025-08-30), no collection procedure, no real images; the MIT tag is the uploader's, and whether an uploader can license Gemini outputs under MIT is not established; uniform 1024 x 1024 PNG is exactly the format and size cue that E-028 measured (format-only AUC 0.9222) |
| zr-zhang/MLLM-Generated-Image-Detection-Dataset | 45,103 rows, real and generated, "GPT Image2 and Nano Banana2"; tag apache-2.0 plus a research-only restriction; paper arXiv:2608.01258 | Generator per image not stated (sample paths sit under a "FLUX.2" folder, which conflicts with the card); real-image sources not stated; no per-class counts; 50.3 GB |
| ash12321/nano-banana-pro-generated-1k | 200 images, "Nano Banana Pro", MIT tag (search result, card not opened) | Very small; provenance not checked |
| Images the user generates with Gemini | Possible, with a recorded procedure | Terms must be read by the user: Google's Terms of Service, the Generative AI Prohibited Use Policy and, for the API, the Gemini API Additional Terms. A web search did not find Google's primary text on using outputs to train a detector; secondary summaries conflict. NR until the user reads the primary terms |

Sources of the checks: the two dataset cards above (fetched 2026-10-27), and Google's help page on Gemini image generation, which
names the Terms of Service, the Generative AI Service Specific Terms and the Prohibited Use Policy
(https://support.google.com/bard/answer/14286560). No claim about what those terms allow is made here.

## 3. Feasibility verdict

Not feasible as a trustworthy experiment today. The blockers, in the order they must be resolved:

1. **Licence and terms (user decision).** Confirm from Google's primary terms that generated images may be used to train and evaluate a
   detector for non-commercial research, and accept or reject the uploader licences of the candidate sets.
2. **Provenance.** No candidate set records prompts, generation dates or the exact model build. Without them, "Gemini" cannot be
   separated from "one uploader's batch of one preview model".
3. **Matched real controls.** No candidate set includes real photographs matched in content, size and compression. Training Gemini PNG
   files against JPEG photographs would learn the format, as E-028 showed for the Community Forensics fakes.
4. **Version coverage.** Gemini 2.5 Flash Image, Gemini 3 Pro Image ("Nano Banana Pro") and Imagen-based outputs in the Gemini app are
   different generators; one preview model does not represent "Gemini-generated images".

## 4. Preregistration proposal (E-031, PROPOSED; to be hashed only after blockers 1 to 3 are resolved)

The text below is the proposal. It becomes a preregistration in docs/EXPERIMENTS.md, hashed before any image is decoded, only after
the user approves the data.

- Question: can a head on frozen CLIP ViT-L/14 features separate images of one named Gemini image model from real photographs of
  sources it was not trained or calibrated on, at a false-positive rate fixed in advance, after both classes are put through the same
  resize and compression?
- Data. Positive class, development: one Gemini image model with recorded provenance (dataset revision and SHA-256 per file, or the
  user's own generation log: tool, model name shown in the interface, date, prompt, output file hash). Positive class, untouched test:
  a different collection of the same model (other prompts, other dates, ideally another collector) plus, as a separate block, a second
  Gemini model version (generalisation within the family). Real class: training and validation from Community Forensics-Small reals
  (COCO, FFHQ, VISION) and at least one more source to give the threshold three sources; test reals from sources not used for training
  or calibration (LAION, RAISE and ImageNet reals of Community Forensics-Eval, test-only as licensed).
- Control of obvious differences: every image of both classes is decoded, resized to 512 px on the short side and re-encoded as JPEG
  quality 90 by one function before features are extracted; format-only and size-only scores are reported as baselines; a content
  control matches real images to the Gemini prompts by caption where captions exist (COCO), reported separately.
- Splits: by prompt (development) and by collection (test); exact duplicates by SHA-256 and near duplicates by a 64-bit DCT hash with a
  Hamming distance of at most 6 removed across splits before any feature is extracted; the audit result is logged.
- Models compared on the same images and the same transformations: A, frozen T-I (E-028, product head); B, the frozen E-029 candidate
  (IMG-13 seed 0, its own validation threshold); C, the E-029 recipe (linear head, class-balanced loss, early stopping on validation)
  trained on the Gemini development data, three seeds.
- Threshold rule fixed in advance: for C, the lowest threshold whose false-positive rate on the pooled validation reals of three
  sources is at most 0.02; for A and B, their frozen thresholds. Frozen and hashed before the test look; one look.
- Metrics on the test: ROC AUC, TPR (recall) and FPR at the frozen threshold, specificity, precision at the stated class ratio, F1,
  confusion matrix, all with bootstrap intervals over images (and over prompts where known); FPR per real source; robustness blocks:
  the test set additionally resized by 0.5 and re-encoded at JPEG quality 75; threshold sensitivity curve (TPR and FPR at thresholds
  around the frozen value, reported, not used for any choice).
- Lines fixed in advance. Usable as a research detector of that Gemini model only if: pooled test FPR upper bound at most 0.05 and every
  unseen real source at most 0.10; TPR lower bound at least 0.80 on the untouched same-model test; AUC drop under the robustness blocks at
  most 0.05. Not usable if the pooled FPR lower bound is above 0.10 or any robustness AUC is below 0.80. The second Gemini version is
  reported separately; no claim covers other Gemini models without passing its own block.
- Product: none. A pass would make C a candidate for an experimental card under the roadmap rules; it would not enter APPROVED_HEADS or
  any finding without a separate decision.
- Compute: about 10,000 to 20,000 images; feature extraction at the E-029 rate fits in under an hour on the RTX 5070; head training in
  minutes on the CPU.

## 4a. Sample size against the lines (computed 2026-10-27, Wilson 95 percent intervals, before any data exists)

The lines in section 4 are judged on interval bounds, so the sample sizes decide whether a good detector can pass at all. Wilson
intervals for a proportion (images treated as independent; bootstrap over prompts, which the proposal also requires, gives wider
intervals because images of one prompt are correlated):

| Quantity | Line | What the sample size allows |
|---|---|---|
| TPR on the untouched same-model test | lower bound at least 0.80 | 200 images: a true TPR of 0.85 gives a lower bound of 0.794 (fails); at least 230 images are needed at 0.85, and 400 give 0.812. At TPR 0.90, 70 images suffice |
| Pooled FPR on unseen real sources | upper bound at most 0.05 | 200 reals: a true FPR of 0.02 gives 0.050 (borderline); 500 reals give 0.036; the 5,791 Eval reals are ample |
| FPR per unseen real source | upper bound at most 0.10 | RAISE has only 121 photographs in the Eval subset: its line can be met only if its true FPR is about 0.04 or lower (0.05 needs about 130 images) |
| Threshold calibration on validation reals | FPR at most 0.02 | the point estimate on, for example, 2,161 validation reals has a Wilson half-width of about 0.006, so the realised test FPR can be expected to differ from 0.02 by more than sampling alone when the real sources shift (E-029) |

Consequences for the proposal (to be written into the preregistration): (1) the untouched Gemini test needs at least 400 images from
at least 100 distinct prompts, and the development set at least 600, so about 1,000 generated images in total, not 600 as section 5
first suggested; (2) the per-source line is reported for RAISE with its small n stated, and a RAISE failure that rests only on interval
width is reported as "not distinguished" rather than as a pass or a fail; (3) intervals use the bootstrap over prompts (and over real
source items), so the effective sample size is the number of prompts, which is why 100 prompts is the floor; (4) five lines are tested
on one look, so the family-wise error is higher than 5 percent; the result reports this instead of claiming each line at 95 percent.

## 5. Smallest next step

The user (a) reads Google's primary terms and records the conclusion in DATA_LOG, and (b) chooses the positive-class source: either
generate about 1,000 images with one named Gemini model with a written log (prompt, date, model name, file hash; about 600
development and at least 400 test images from at least 100 distinct prompts, split by prompt; section 4a), or accept one community set with its gaps written down. With (a) and (b) done, E-031 can be preregistered and run in
one working session.
