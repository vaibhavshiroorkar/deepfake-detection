"""Regenerate the result registry from the artifacts it points at.

`docs/research/result-traceability.md` requires every reported number to resolve
to an analysis command, content hashes, and MLflow runs. Written by hand, that
table goes stale the first time a run is repeated, and a stale hash is worse
than none: it still looks like evidence.

So the rows are declared here, next to the command that produces each artifact,
and the hashes are computed from the files at the moment of writing. An artifact
that is missing produces a `pending` row naming what to run, rather than being
quietly dropped.

Status follows the acceptance rules in that document, and the seed rule is the
one that bites: the frozen protocol asks for three training seeds and the
Design B streams were trained on one, so their rows are `provisional` however
well the hashes resolve.

    uv run python scripts/update_result_registry.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY = Path("docs/research/result-traceability.md")
BEGIN = "<!-- BEGIN GENERATED REGISTRY -->"
END = "<!-- END GENERATED REGISTRY -->"


@dataclass(frozen=True, slots=True)
class Result:
    """One reported number and everything needed to get back to it."""

    result_id: str
    paper_location: str
    command: str
    report: Path
    # The per-clip scores behind the report. Design A writes a predictions CSV;
    # the Design B streams write a feature store instead, which carries the same
    # per-clip logits alongside the embeddings fusion reads.
    predictions: Path | None
    mlflow: tuple[tuple[str, str], ...] = ()
    decision: str = ""
    seeds: int = 1
    notes: str = ""
    # Set where a result is quoted from a run whose artifact was never written
    # to `runs/`. It stays in the table as `pending` with the command that would
    # produce it, because a number in the paper with no file behind it is the
    # thing this registry exists to make visible.
    missing_reason: str = ""
    extra: tuple[str, ...] = field(default_factory=tuple)


PROGRAM = Path("runs/program-20260906")
DESIGN_B = Path("runs/design-b-20260910")
STREAMS = Path("runs/streams-20260905")
FFPP = Path("runs/ffpp-20260913")

RESULTS: tuple[Result, ...] = (
    Result(
        result_id="A-visual-dfdc",
        paper_location="Results, cross-corpus baseline",
        command="ddf evaluate branch --branch visual --dataset DFDC",
        report=PROGRAM / "evaluation" / "visual-dfdc-metrics.json",
        predictions=PROGRAM / "evaluation" / "visual-dfdc-predictions.csv",
        mlflow=(
            ("program-20260906", "final-visual-seed17"),
            ("program-20260906", "visual-dfdc-evaluation"),
        ),
        decision="Design A visual baseline, the number Design B is measured against",
        seeds=1,
    ),
    Result(
        result_id="A-fusion-dfdc",
        paper_location="Results, late fusion cross-corpus",
        command="ddf evaluate fusion --dataset DFDC",
        report=PROGRAM / "evaluation" / "fusion-dfdc-metrics.json",
        predictions=PROGRAM / "evaluation" / "fusion-dfdc-predictions.csv",
        mlflow=(("program-20260906", "fusion-dfdc-evaluation"),),
        decision="Late fusion does not beat its own visual branch cross-corpus",
        seeds=1,
    ),
    Result(
        result_id="A-fusion-in-domain",
        paper_location="Results, late fusion in-domain",
        command="ddf evaluate fusion --dataset FakeAVCeleb --partition test",
        report=PROGRAM / "evaluation" / "fusion-test-metrics.json",
        predictions=PROGRAM / "evaluation" / "fusion-test-predictions.csv",
        mlflow=(("program-20260906", "fusion-test-evaluation"),),
        decision="In-domain ceiling for the Design A pipeline",
        seeds=1,
    ),
    Result(
        result_id="A-visual-mnw",
        paper_location="Results, how generalization fails",
        command="ddf evaluate branch --branch visual --dataset MNW",
        report=PROGRAM / "evaluation" / "visual-mnw-metrics.json",
        predictions=PROGRAM / "evaluation" / "visual-mnw-predictions.csv",
        mlflow=(("program-20260906", "visual-mnw-evaluation"),),
        decision=(
            "Detection is 25 of 85 and not uniform across generators: 0 of 10 on "
            "vasa_1 against 7 of 8 in the wild"
        ),
        seeds=1,
        notes="Fake-only, so no ranking metric exists. Detection rate only.",
    ),
    Result(
        result_id="A-authenticity-density",
        paper_location="Results, abstaining instead of accusing",
        command=(
            "python scripts/fit_authenticity_density.py --features "
            "runs/probe-20260914/ood2-dinov3.npz --reference real"
        ),
        report=Path("runs/probe-20260914/ood2-density-0.10.json"),
        predictions=Path("runs/probe-20260914/ood2-dinov3.json"),
        mlflow=(),
        decision=(
            "Intercepts 74.4 percent of unfamiliar genuine clips at 8.7 percent "
            "abstention on familiar ones; 67.3 percent averaged over twenty "
            "reference splits, 49.4 to 83.1"
        ),
        seeds=1,
    ),
    Result(
        result_id="A-gate-encoder",
        paper_location="Results, abstaining instead of accusing",
        command=(
            "python scripts/extract_probe_features.py --folder data/ood-probe "
            "--encoder pe --output runs/probe-20260914/ood2-pe.npz; "
            "python scripts/fit_authenticity_density.py --features "
            "runs/probe-20260914/ood2-pe.npz --reference real --flag-rate 0.10 "
            "--encoder pe --output runs/probe-20260914/ood2-pe-density-0.10.json "
            "--save-model runs/probe-20260914/authenticity-density-pe.npz"
        ),
        report=Path("runs/probe-20260914/ood2-pe-density-0.10.json"),
        predictions=Path("runs/probe-20260914/ood2-pe.json"),
        mlflow=(),
        decision=(
            "Perception Encoder intercepts 32.8 percent over twenty splits "
            "against DINOv3's 67.3 and abstains on more familiar clips; the "
            "gate keeps DINOv3"
        ),
        seeds=1,
    ),
    Result(
        result_id="A-false-alarms",
        paper_location="Results, false alarms on genuine video",
        command="ddf evaluate branch, then threshold sweep over the predictions",
        report=PROGRAM / "evaluation" / "visual-celebdf-metrics.json",
        predictions=PROGRAM / "evaluation" / "visual-celebdf-predictions.csv",
        mlflow=(("program-20260906", "visual-celebdf-evaluation"),),
        decision=(
            "Genuine clips called fake: 0.00 percent in-domain, 4.14 on DFDC, "
            "90.97 on Celeb-DF, where no threshold gives both zero false alarms "
            "and any detection"
        ),
        seeds=1,
    ),
    Result(
        result_id="A-motion",
        paper_location="Results, motion hides manipulation",
        command="python scripts/motion_vs_error.py --dataset dfdc",
        report=PROGRAM / "evaluation" / "motion-dfdc.json",
        predictions=PROGRAM / "evaluation" / "visual-dfdc-predictions.csv",
        mlflow=(),
        decision=(
            "Motion lowers the fake score on manipulated clips in both corpora, "
            "so moving video produces missed detections rather than false alarms"
        ),
        seeds=1,
    ),
    Result(
        result_id="A-operating-point",
        paper_location="Results, calibration and the paired comparison",
        command="ddf evaluate fusion --dataset DFDC",
        report=PROGRAM / "evaluation" / "fusion-dfdc-metrics.json",
        predictions=PROGRAM / "evaluation" / "fusion-dfdc-predictions.csv",
        mlflow=(("program-20260906", "fusion-dfdc-evaluation"),),
        decision=(
            "Paired difference -0.0083 [-0.0185, 0.0024], so fusion is not "
            "separated from the visual baseline cross-corpus; calibration error "
            "rises from 0.0022 to 0.6366"
        ),
        seeds=1,
    ),
    Result(
        result_id="B-stream-scores",
        paper_location="Results, per-stream ROC-AUC with intervals",
        command="python scripts/score_streams.py --run-dir runs/design-b-20260910",
        report=DESIGN_B / "evaluation" / "stream-scores.json",
        predictions=DESIGN_B / "features" / "dfdc.parquet",
        mlflow=(
            ("design-b-frozen-bn", "visual-efficientnet"),
            ("design-b-frozen-bn", "visual-dinov3"),
            ("design-b-frozen-bn", "stream-lipsync"),
            ("design-b-frozen-bn", "stream-emotion"),
        ),
        decision="Every stream except emotion reads near chance cross-corpus",
        seeds=1,
    ),
    Result(
        result_id="B-lavdf-lipsync",
        paper_location="Results, attention at chance on a shortcut-controlled corpus",
        command="python scripts/score_lavdf_stream.py",
        report=STREAMS / "lavdf-lipsync-score.json",
        predictions=STREAMS / "lavdf-val-usable.csv",
        mlflow=(),
        decision=(
            "0.9969 ROC-AUC with diagonal mass at chance, on matched windows "
            "from the same file, so the mechanism is absent even where the "
            "corpus forbids every other cue"
        ),
        seeds=1,
    ),
    Result(
        result_id="B-lavdf-transfer",
        paper_location="Results, capture conditions bind generalization",
        command=(
            "python scripts/score_lavdf_stream.py --cache-run "
            "runs/program-20260906 --dataset DFDC --manifest "
            "runs/program-20260906/split/dfdc-sync.csv --label dfdc"
        ),
        report=STREAMS / "lavdf-lipsync-score-dfdc.json",
        predictions=STREAMS / "lavdf-lipsync-score-fakeavceleb.json",
        mlflow=(),
        decision=(
            "A matched-pair-trained stream transfers freely inside VoxCeleb2, "
            "0.9969 to 0.7407, and reads chance outside it at 0.5059"
        ),
        seeds=1,
    ),
    Result(
        result_id="B-training-histories",
        paper_location="Results, attention that never moved; lip-sync collapse",
        command="pwsh scripts/train_design_b.ps1",
        report=DESIGN_B / "checkpoints" / "stream-emotion-history.json",
        predictions=DESIGN_B / "checkpoints" / "stream-lipsync-lowlr-history.json",
        mlflow=(
            ("design-b-frozen-bn", "stream-emotion"),
            ("design-b-frozen-bn", "stream-lipsync"),
        ),
        decision=(
            "Both cross-modal streams sit on chance diagonal mass for every "
            "epoch of every run"
        ),
        seeds=1,
    ),
    Result(
        result_id="B-deep-ablation",
        paper_location="Results, does fusing streams beat the best single stream",
        command="python scripts/run_deep_ablation.py --run-dir runs/design-b-20260910",
        report=DESIGN_B / "evaluation" / "deep-ablation.json",
        predictions=DESIGN_B / "features" / "holdout.parquet",
        mlflow=(),
        decision=(
            "A subset beats the full fusion cross-corpus, so more streams is not "
            "the lever"
        ),
        seeds=1,
    ),
    Result(
        result_id="B-stream-correlation",
        paper_location="Results, redundancy between streams",
        command="python scripts/stream_correlation.py --run-dir runs/design-b-20260910",
        report=DESIGN_B / "evaluation" / "stream-correlation.json",
        predictions=DESIGN_B / "features" / "in-domain.parquet",
        mlflow=(),
        decision="Redundancy is measurable and was not where it was predicted",
        seeds=1,
    ),
    Result(
        result_id="B-batchnorm-arms",
        paper_location="Results, freezing BatchNorm trades transfer for accuracy",
        command=(
            "python scripts/score_batchnorm_arms.py"
        ),
        report=DESIGN_B / "evaluation" / "batchnorm-arms.json",
        predictions=None,
        mlflow=(
            ("design-b-frozen-bn", "visual-efficientnet"),
            ("design-b-live-bn", "visual-efficientnet"),
        ),
        decision=(
            "Live BatchNorm buys 0.1276 on DFDC and costs 0.5430 in-domain. The "
            "frozen arm ships; the trade is the result"
        ),
        seeds=1,
    ),
    Result(
        result_id="B-batchnorm-trade",
        paper_location="Results, freezing BatchNorm trades transfer for accuracy",
        command=(
            "python scripts/batchnorm_sweep.py --run-dir runs/design-b-20260910"
        ),
        report=DESIGN_B / "evaluation" / "batchnorm-sweep.json",
        predictions=None,
        mlflow=(),
        decision=(
            "With the epoch budget held fixed, BatchNorm moves DFDC by +0.0177, "
            "not the +0.1276 of the uncontrolled full-scale comparison, and "
            "epochs alone move it by 0.0042. The in-domain cost survives the "
            "control: validation falls 0.9072 to 0.5500"
        ),
        seeds=1,
        notes=(
            "1,400-clip arms, so read the difference between arms rather than "
            "the absolute numbers."
        ),
    ),
    Result(
        result_id="C-df26-logo",
        paper_location="Results, leave-one-generator-out on DF26",
        command=(
            "python scripts/train_logo_probe.py --features "
            "runs/probe-20260914/df26-dinov3.npz"
        ),
        report=Path("runs/probe-20260914/df26-dinov3-logo.json"),
        predictions=Path("runs/probe-20260914/df26-dinov3.json"),
        mlflow=(),
        decision=(
            "Macro 0.6942 across seven unseen generators before the frame-rate "
            "control. Real clips were the only ones at 25 or 29.97 fps, and "
            "C-df26-logo-controlled is the same probe with that removed"
        ),
        seeds=1,
    ),
    Result(
        result_id="C-df26-logo-controlled",
        paper_location="Results, leave-one-generator-out on DF26, frame rate held fixed",
        command=(
            "python scripts/normalize_media.py --source data/df26-scored "
            "--target data/df26-norm --fps 24; python "
            "scripts/extract_probe_features.py --folder data/df26-norm "
            "--encoder dinov3; python scripts/train_logo_probe.py --features "
            "runs/probe-20260914/df26-norm-dinov3.npz"
        ),
        report=Path("runs/probe-20260914/df26-norm-dinov3-logo.json"),
        predictions=Path("runs/probe-20260914/df26-norm-dinov3.json"),
        mlflow=(),
        decision=(
            "Macro 0.6786 with every clip re-encoded at 24 fps. Wan 2.6 fell "
            "0.094 and HunyuanVideo 0.045, so frame rate carried part of the "
            "uncontrolled number"
        ),
        seeds=1,
    ),
    Result(
        result_id="C-df26-logo-pe",
        paper_location="Results, leave-one-generator-out on DF26, the headline",
        command=(
            "python scripts/extract_probe_features.py --folder data/df26-norm "
            "--encoder pe; python scripts/train_logo_probe.py --features "
            "runs/probe-20260914/df26-norm-pe.npz"
        ),
        report=Path("runs/probe-20260914/df26-norm-pe-logo.json"),
        predictions=Path("runs/probe-20260914/df26-norm-pe.json"),
        mlflow=(),
        decision=(
            "Macro 0.7527 across seven unseen generators, frame-rate controlled, "
            "range 0.6542 on LTX 2.3 to 0.8496 on Veo 3.1. The headline, since "
            "A-encoder-comparison separates Perception Encoder from DINOv3"
        ),
        seeds=1,
    ),
    Result(
        result_id="C-df26-audio-audit",
        paper_location="Results, DF26 cannot support audiovisual work",
        command="ffprobe over 30 clips per DF26 group",
        report=Path("data/df26-scored/df26-index.csv"),
        predictions=None,
        mlflow=(),
        decision=(
            "Audio presence predicts the label: 30 of 30 for every commercial "
            "generator, 0 of 30 for every open-source one, 3 of 30 for real"
        ),
        seeds=1,
    ),
    Result(
        result_id="C-ffpp-logo",
        paper_location="Results, the region of interest decides the result",
        command=(
            "python scripts/extract_probe_features.py --folder "
            "data/FaceForensics++/FaceForensics++_C23 --limit 150; "
            "python scripts/train_logo_probe.py --features "
            "runs/probe-20260914/ffpp-dinov3.npz"
        ),
        report=Path("runs/probe-20260914/ffpp-dinov3-logo.json"),
        predictions=Path("runs/probe-20260914/ffpp-dinov3.json"),
        mlflow=(),
        decision=(
            "A frozen full-frame probe is at chance on face manipulation, "
            "macro 0.5387 with every interval containing 0.5"
        ),
        seeds=1,
    ),
    Result(
        result_id="C-ffpp-zeroshot",
        paper_location="Results, trained on FF++ and scored on unseen corpora",
        command="python scripts/score_ffpp_zeroshot.py",
        report=FFPP / "evaluation" / "zero-shot.json",
        predictions=FFPP / "checkpoints" / "visual-efficientnet-history.json",
        mlflow=(("ffpp-20260913", "visual-efficientnet"),),
        decision=(
            "The only number in this project trained under the protocol the "
            "published cross-dataset tables use"
        ),
        seeds=1,
    ),
    Result(
        result_id="C-ffpp-family-holdout",
        paper_location="Results, the region of interest decides the result",
        command=(
            "sh scripts/train_family_holdout.sh; "
            "python scripts/summarize_family_holdout.py"
        ),
        report=FFPP / "evaluation" / "family-holdout.json",
        predictions=FFPP / "checkpoints" / "holdout-ffpp-faceswap-history.json",
        mlflow=(),
        decision=(
            "Holding a family out costs 0.1231 macro ROC-AUC, 0.8743 to 0.7511; "
            "faceswap falls to chance at 0.5195 and two drops overlap"
        ),
        seeds=1,
        notes=(
            "Visual stream only: FaceForensics++ has no audio. Unpaired "
            "comparison against the all-family model on the same test clips."
        ),
    ),
    Result(
        result_id="B-fusion-ablations",
        paper_location="Results, late against deep and abstention against fallback",
        command=(
            "python scripts/run_fusion_ablations.py --run-dir runs/design-b-20260910"
        ),
        report=DESIGN_B / "evaluation" / "fusion-ablations.json",
        predictions=DESIGN_B / "features" / "dfdc.parquet",
        mlflow=(),
        decision=(
            "The embedding carries nothing the logit does not, and abstention "
            "buys no accuracy"
        ),
        seeds=1,
    ),
    Result(
        result_id="A-corruption-visual",
        paper_location="Results, corruption robustness",
        command=(
            "python scripts/run_corruption_eval.py "
            "--run-dir runs/program-20260906 --branch visual"
        ),
        report=PROGRAM / "evaluation" / "corruption-visual.json",
        predictions=None,
        mlflow=(("program-20260906", "final-visual-seed17"),),
        decision=(
            "JPEG 75 costs 0.0054 of ROC-AUC and takes false alarms from 0 to 76 "
            "of 150 genuine clips; by JPEG 50 every genuine clip is called fake "
            "while AUC still reads 0.9187. Ranking survives degradation, the "
            "decision does not"
        ),
        seeds=1,
        notes=(
            "Corruption applied to the cached crop with the face track held "
            "fixed, so this bounds the classifier only. Corrupting the source "
            "would also break detection. Optimistic bound, not an estimate."
        ),
    ),
    Result(
        result_id="A-corruption-audio",
        paper_location="Results, corruption robustness",
        command=(
            "python scripts/run_corruption_eval.py "
            "--run-dir runs/program-20260906 --branch audio"
        ),
        report=PROGRAM / "evaluation" / "corruption-audio.json",
        predictions=None,
        mlflow=(("program-20260906", "final-audio-seed17"),),
        decision=(
            "Noise at 20 dB SNR costs 0.0227 of ROC-AUC and misses 857 of 859 "
            "fakes, the opposite failure direction to the visual branch. A 0.5 "
            "second silence shift changes nothing, so the silence shortcut the "
            "threat model names is absent from this checkpoint"
        ),
        seeds=1,
    ),
    Result(
        result_id="A-error-analysis",
        paper_location="Results, what the locked system gets wrong",
        command=(
            "python scripts/run_error_analysis.py --run-dir runs/program-20260906"
        ),
        report=PROGRAM / "evaluation" / "error-analysis.json",
        predictions=PROGRAM / "evaluation" / "fusion-dfdc-predictions.csv",
        mlflow=(),
        decision=(
            "Cross-corpus the model is more confident when wrong than when "
            "right, 0.8742 against 0.8299 on DFDC, so no confidence threshold "
            "can abstain its way out. Fusion repairs 314 of the visual branch's "
            "DFDC errors and breaks 5, while 1,585 are beyond any head"
        ),
        seeds=1,
        notes=(
            "Reads the frozen evaluation predictions. No model is loaded and no "
            "threshold is selected, as Phase 5 requires."
        ),
    ),
    Result(
        result_id="A-sync-label-mode",
        paper_location="Results, the sync label-mode ablation",
        command=(
            "ddf train sync --label-mode global-fake, against the shipped "
            "authentic-offset checkpoint"
        ),
        report=PROGRAM / "checkpoints" / "final-sync-globalfake-seed17-history.json",
        predictions=None,
        mlflow=(),
        decision=(
            "Required ablation, never run until now. Read against each arm's own "
            "chance loss: the shipped authentic-offset arm sits at or above its "
            "2.079 at 2.1842, so it did not learn its objective, while the "
            "rejected global-fake arm reaches 0.0558 against its 0.693. The "
            "design's reason for rejecting the fake label is better supported, "
            "not reversed: a binary head separates whole-clip manipulation by "
            "generator artifacts"
        ),
        seeds=1,
        notes=(
            "The two losses are different tasks and are not comparable as "
            "numbers. No ROC-AUC for the global-fake arm: sync_anomaly_logit "
            "assumes offset semantics and is meaningless for a binary head."
        ),
    ),
    Result(
        result_id="A-research-design-audit",
        paper_location="Limitations, what the design asked for and did not get",
        command=(
            "read docs/research-design.md against the registry and the sync "
            "training history"
        ),
        report=PROGRAM / "checkpoints" / "final-sync-seed17-history.json",
        predictions=None,
        mlflow=(("program-20260906", "final-sync-seed17"),),
        decision=(
            "The exit criterion 'the sync offset confidence interval must exclude "
            "chance' has never been measured: the history records only total "
            "loss, and the loss combines offset cross-entropy with a contrastive "
            "term so it cannot be decomposed after the fact. Two of eight "
            "required ablations, the sync label mode and leading silence, were "
            "never run"
        ),
        seeds=3,
        notes=(
            "The artifact is the history file, cited as evidence of what it does "
            "not contain. Both unrun ablations are expressible today."
        ),
    ),
    Result(
        result_id="A-encoder-comparison",
        paper_location="Method, which frozen encoder for the open-world probe",
        command=(
            "bash scripts/run_encoder_comparison.sh; python "
            "scripts/compare_probe_encoders.py --features "
            "runs/probe-20260914/df26-norm-dinov3.npz "
            "runs/probe-20260914/df26-norm-pe.npz "
            "runs/probe-20260914/df26-norm-clip.npz --output "
            "runs/probe-20260914/df26-norm-encoder-comparison.json"
        ),
        report=Path("runs/probe-20260914/df26-norm-encoder-comparison.json"),
        predictions=None,
        mlflow=(),
        decision=(
            "Perception Encoder beats DINOv3 by +0.0741 [+0.0121, +0.1317] macro, "
            "paired and resampled by source video, and leads on all seven arms. "
            "CLIP beats DINOv3 by +0.0668 [+0.0049, +0.1289]. PE against CLIP is "
            "not separated, -0.0073 [-0.0603, +0.0463]. The probe switches to PE. "
            "The earlier unverified claim, 72.7 against 57.0, had the right "
            "direction and twice the gap"
        ),
        seeds=1,
        notes=(
            "Source-video intervals are wider than the per-arm clip intervals in "
            "each -logo.json, because every arm shares its real clips."
        ),
    ),
    Result(
        result_id="A-audio-crosscorpus-gap",
        paper_location="Results, the audio branch has no cross-corpus test",
        command=(
            "score the audio rows of runs/program-20260906/features/dfdc.parquet "
            "against the clip-level label, identity-clustered"
        ),
        report=PROGRAM / "features" / "dfdc-report.json",
        predictions=PROGRAM / "features" / "dfdc.parquet",
        mlflow=(),
        decision=(
            "The audio branch is at chance cross-corpus: 0.5054 [0.4747, 0.5371] "
            "on DFDC against 0.7739 in-domain, clip-level label. A cue-specific "
            "audio evaluation is impossible there because DFDC does not publish "
            "which modality it manipulated, but the clip-level one was available "
            "in the feature store all along"
        ),
        seeds=1,
        notes=(
            "Supersedes an earlier reading of audio_fake=0 on DFDC as evidence "
            "the audio is genuine. It is a placeholder for cue unknown, as the "
            "DFDC ingestion docstring states."
        ),
    ),
    Result(
        result_id="A-visual-convnext",
        paper_location="Results, the visual challenger",
        command=(
            "ddf train visual --architecture convnext-tiny, then "
            "ddf evaluate branch --branch visual --architecture convnext-tiny"
        ),
        report=PROGRAM / "evaluation" / "visual-convnext-dfdc-metrics.json",
        predictions=PROGRAM / "evaluation" / "visual-convnext-dfdc-predictions.csv",
        mlflow=(),
        decision=(
            "VIS-01 decided for EfficientNet-B0 on stability, not accuracy. "
            "ConvNeXt is worse on DFDC at all three seeds, mean 0.6783 against "
            "0.7151, and diverged outright at seed 43 under an identical "
            "budget: validation loss 0.218, 0.6473, 0.7172, leaving a "
            "never-fine-tuned frozen-backbone checkpoint"
        ),
        seeds=3,
        notes=(
            "The only seed whose paired interval separates is the failed one. "
            "On the two healthy seeds the difference is -0.0340 and -0.0187, "
            "neither separated."
        ),
    ),
    Result(
        result_id="A-audio-wavlm",
        paper_location="Results, the audio challenger",
        command=(
            "ddf train audio --audio-model microsoft/wavlm-base, then "
            "ddf evaluate branch --branch audio --audio-model microsoft/wavlm-base"
        ),
        report=PROGRAM / "evaluation" / "audio-wavlm-in-domain-test-metrics.json",
        predictions=(
            PROGRAM / "evaluation" / "audio-wavlm-in-domain-test-predictions.csv"
        ),
        mlflow=(),
        decision=(
            "AUD-01 undecided: both encoders read ROC-AUC 1.0000 in-domain, so "
            "the partition cannot separate them and the incumbent stays by "
            "scope control rather than evidence. The audio branch has no "
            "cross-corpus evaluation to decide it"
        ),
        seeds=1,
        notes=(
            "Before this run the flag built a truncated Wav2Vec2 from the WavLM "
            "checkpoint, 94,371,712 parameters against 94,381,936."
        ),
    ),
    Result(
        result_id="A-fusion-seed-spread",
        paper_location="Results, what a seed is worth cross-corpus",
        command=(
            "python scripts/run_fusion_seed_spread.py "
            "--run-dir runs/program-20260906"
        ),
        report=PROGRAM / "evaluation" / "fusion-seed-spread.json",
        predictions=PROGRAM / "features" / "dfdc-seed29.parquet",
        mlflow=(),
        decision=(
            "Fused DFDC moves 0.7500, 0.6814, 0.6442 with the branch seed, a "
            "spread of 0.1057 against 0.0001 in-domain. The reported 0.7500 is "
            "the best of three and the mean is 0.6919, so the headline needs "
            "the range beside it"
        ),
        seeds=3,
        notes=(
            "Branch checkpoints vary; the head is the seed-17 fit reused, "
            "because the fold models exist at seed 17 only."
        ),
    ),
    Result(
        result_id="A-sync-seeds",
        paper_location="Results, the synchronization branch",
        command=(
            "python scripts/score_sync_branch.py --run-dir runs/program-20260906 "
            "--checkpoint runs/program-20260906/checkpoints/final-sync-seed17.pt"
        ),
        report=PROGRAM / "evaluation" / "sync-final-sync-seed17-metrics.json",
        predictions=PROGRAM / "evaluation" / "sync-final-sync-seed17-predictions.csv",
        mlflow=(("program-20260906", "final-sync-seed17"),),
        decision=(
            "The branch's first measured number, and it is below chance on all "
            "three seeds: 0.4364, 0.4570, 0.4931, with two intervals excluding "
            "0.5. The scalar fusion reads from this branch carries no usable "
            "signal at any seed"
        ),
        seeds=3,
        notes=(
            "Scored on clip_fake, matching the fusion store. A re-synchronised "
            "fake may be genuinely aligned, so this is a floor rather than a "
            "measurement of alignment detection."
        ),
    ),
    Result(
        result_id="A-visual-seeds",
        paper_location="Results, what a seed is worth cross-corpus",
        command=(
            "pwsh scripts/run_seeds.ps1 -Branches visual -SkipTraining"
        ),
        report=PROGRAM / "evaluation" / "visual-dfdc-seed29-metrics.json",
        predictions=PROGRAM / "evaluation" / "visual-dfdc-seed29-predictions.csv",
        mlflow=(),
        decision=(
            "Seed alone spreads DFDC across 0.7611, 0.6963 and 0.6878 while "
            "in-domain holds to three decimals. The spread of 0.0733 is four and "
            "a half times the in-domain fusion gain, and the reported seed 17 is "
            "the best of the three against a mean of 0.7151"
        ),
        seeds=3,
        notes=(
            "Checkpoints trained 2026-09-08 by run_program.ps1 and never scored, "
            "because its stage 5 names the seed-17 checkpoint directly."
        ),
    ),
    Result(
        result_id="A-branch-ablation",
        paper_location="Results, does fusing branches beat the best single branch",
        command="python scripts/run_ablation.py --run-dir runs/program-20260906",
        report=PROGRAM / "evaluation" / "fusion-ablation.json",
        predictions=PROGRAM / "features" / "test.parquet",
        mlflow=(),
        decision=(
            "All three branches beat every single branch in-domain, "
            "+0.0159 [+0.0129, +0.0190] against visual, and do not cross-corpus, "
            "-0.0085 [-0.0196, +0.0027]"
        ),
        seeds=1,
        notes=(
            "Clip-level label throughout, so the visual row is 0.9742 and not "
            "the 0.9988 of the cue-labelled branch report."
        ),
    ),
    Result(
        result_id="A-fusion-model-ablation",
        paper_location="Results, which fusion head and what its inputs are worth",
        command=(
            "python scripts/run_fusion_model_ablation.py "
            "--run-dir runs/program-20260906"
        ),
        report=PROGRAM / "evaluation" / "fusion-model-ablation.json",
        predictions=PROGRAM / "features" / "dfdc.parquet",
        mlflow=(),
        decision=(
            "FUS-01 settled for logistic: the MLP is indistinguishable in-domain "
            "and loses 0.1508 [0.0995, 0.2052] cross-corpus. Calibration and the "
            "quality features buy no ranking, and an unweighted mean of the "
            "calibrated branches is better calibrated cross-corpus than the "
            "learned head, 0.0554 against 0.6366"
        ),
        seeds=1,
    ),
    Result(
        result_id="A-generalization-slices",
        paper_location="Results, where the fused system fails",
        command=(
            "python scripts/run_generalization_slices.py "
            "--run-dir runs/program-20260906"
        ),
        report=PROGRAM / "evaluation" / "generalization-slices.json",
        predictions=PROGRAM / "features" / "test.parquet",
        mlflow=(),
        decision=(
            "No method or subgroup gap is visible in-domain, at a ceiling that "
            "could not show one. Dropping rtvc from the head's fit misses 73 of "
            "its 75 clips while ROC-AUC only falls to 0.9563, so the head's "
            "audio weighting is method-specific and AUC hides it"
        ),
        seeds=1,
        notes=(
            "The method holdout refits the head only. Branch checkpoints saw "
            "every method, so this is not an unseen-manipulation result."
        ),
    ),
    Result(
        result_id="B-gate-thresholds",
        paper_location="Method, per-media-kind decision thresholds",
        command="python scripts/fit_gate_fusion.py --run-dir runs/design-b-20260910",
        report=DESIGN_B / "checkpoints" / "gate-fusion.json",
        predictions=DESIGN_B / "features" / "holdout.parquet",
        mlflow=(),
        decision="One threshold cannot serve video, image and audio",
        seeds=1,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_cell(path: Path | None) -> str:
    if path is None:
        return "not applicable"
    if not path.is_file():
        return "missing"
    return f"`{_sha256(path)[:16]}`"


def _mlflow_ids(entries, tracking_uri: str) -> dict[tuple[str, str], str]:
    """Run IDs for every (experiment, run name) the registry cites."""
    try:
        import mlflow
    except ImportError:
        return {}
    mlflow.set_tracking_uri(tracking_uri)
    found: dict[tuple[str, str], str] = {}
    for experiment in sorted({name for name, _ in entries}):
        try:
            runs = mlflow.search_runs(
                experiment_names=[experiment], output_format="list", max_results=5000
            )
        except Exception as error:
            # An experiment that is not in the store is the ordinary case on a
            # fresh checkout, and it must not stop the rest of the table.
            print(f"  cannot read experiment {experiment}: {error}")
            continue
        for run in runs:
            key = (experiment, run.info.run_name or "")
            # Newest wins: a retrained stream logs a second run under the same
            # name, and the registry should point at the current one.
            found.setdefault(key, run.info.run_id)
    return found


def _status(result: Result, resolved: bool) -> str:
    if not resolved:
        return "pending"
    if result.seeds < 3:
        # The frozen protocol asks for three training seeds. One seed is a real
        # result and not an accepted one.
        return "provisional (1 seed)"
    return "accepted"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    arguments = parser.parse_args(argv)

    cited = [entry for result in RESULTS for entry in result.mlflow]
    run_ids = _mlflow_ids(cited, arguments.tracking_uri)

    lines = [
        "| Result ID | Paper location | Analysis command | Report SHA-256 | "
        "Prediction SHA-256 | MLflow run IDs | Decision | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    pending = 0
    for result in sorted(RESULTS, key=lambda item: item.result_id):
        report_cell = _hash_cell(result.report)
        prediction_cell = _hash_cell(result.predictions)
        resolved = report_cell not in {"missing"} and prediction_cell != "missing"
        if not resolved:
            pending += 1
        ids = [
            f"`{run_ids[key][:12]}`" if key in run_ids else "not logged"
            for key in result.mlflow
        ]
        decision = result.decision
        if result.missing_reason and not resolved:
            decision = f"{decision}. {result.missing_reason}"
        lines.append(
            f"| `{result.result_id}` | {result.paper_location} | "
            f"`{result.command}` | {report_cell} | {prediction_cell} | "
            f"{', '.join(ids) if ids else 'none'} | {decision} | "
            f"{_status(result, resolved)} |"
        )

    text = arguments.registry.read_text(encoding="utf-8")
    block = BEGIN + "\n" + "\n".join(lines) + "\n" + END
    if BEGIN in text and END in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        text = head + block + tail
    else:
        text = text.replace(
            "## Acceptance rules", block + "\n\n## Acceptance rules", 1
        )
    arguments.registry.write_text(text, encoding="utf-8")

    print(f"{len(RESULTS)} rows, {pending} pending -> {arguments.registry}")
    for result in RESULTS:
        for key in result.mlflow:
            if key not in run_ids:
                print(f"  no MLflow run for {key[0]} / {key[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
