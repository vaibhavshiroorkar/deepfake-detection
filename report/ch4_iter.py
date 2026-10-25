"""Chapter 4, results of the iterations I-1 to I-7 and of the re-run baseline.
Every number is copied from the logged result blocks of the experiments (all exploratory: validation and development slice, mean of
three seeds, 95 percent group bootstrap interval of 2,000 resamples) and from the result files written by the experiment scripts."""

M = "−"   # minus sign


def results(B):
    B.h2("Results of the Iterations", trace="EXPERIMENTS E-005 to E-013; DECISIONS D-051, D-054 to D-061; DOC_FACTS section 15")
    B.p("All seven experimental iterations after the working version were run on 25 October 2026, each from a hypothesis, a success line and a failure line written before the run. "
        "The lever of each slot was chosen from the diagnostic battery of the preceding step, so the order in which levers were used differs from the order of the candidate table above. "
        "Every number below is exploratory: it is measured on the validation split or on the development slice, is the mean of three seeds with a 95 percent group bootstrap interval, "
        "and every verdict is the mechanical evaluation of the criteria that were written before the run. None of the sealed test sets was read. <ref tab:ladderres> summarises the ladder as it was measured.")
    B.table("ladderres", "The iteration ladder as measured (exploratory; validation split and development slice).",
            ["Step", "Lever and task", "Validation AUC", "Development slice AUC", "Mechanical verdict"],
            [["0", "Working version: frozen encoder and linear head (face video, generated images, speech)", "0.7985, 1.0000, 0.9942", "0.6933, 0.9935, 0.6100", "baseline"],
             ["1", "Full fine-tuning of the face-video encoder (SigLIP base)", "0.9899 [0.9809, 0.9967]", "0.9096 [0.8776, 0.9388]", "supported"],
             ["2", "Speech: trim silence and normalise the peak", "0.9889 [0.9831, 0.9933]", "0.6723 [0.6126, 0.7584]", "supported"],
             ["3", "Speech: augmentation of the training clips", "0.9869 [0.9818, 0.9911]", "0.7455 [0.7052, 0.8121]", "supported"],
             ["4", "Speech: average of three encoders", "0.9964 [0.9943, 0.9982]", "0.8130 [0.7523, 0.8808]", "supported"],
             ["5", "Frozen heads on LAV-DF, pre-check then fusion", "visual 0.3838, audio 0.6476 (LAV-DF validation, own label)", "not applicable", "not tested: the visual head is below chance"],
             ["6", "Heads trained on LAV-DF, late fusion; visual head on produced videos only", "visual 0.7491, audio 0.8604, fused 0.9935 (LAV-DF validation)", "not applicable", "fusion supported, with the limit below; produced-only visual head not supported"],
             ["7", "Pulse signal check", "not applicable", "not applicable", "fails: signal not recoverable from these data"]],
            [0.9, 5.0, 3.9, 3.3, 3.3], ["center", "left", "center", "center", "left"], size=8, keep=False,
            trace="E-004 to E-012, E-006; exploratory")
    B.figure("ladderres", "fig08_ladder_results.png", "Measured ladder on the development slice: face video (top) and speech (bottom), with 95 percent intervals. The sealed test results are not yet available.", 15.5,
             "Two panels of AUC with intervals on the development slice, face video from the working version to fine-tuning and the baseline, speech from the working version to the three-encoder average",
             tag="exploratory", trace="E-004, E-005, E-007, E-008, E-009, E-013 (exploratory)")

    # ---- iteration 1
    B.h3("Iteration 1: fine-tuning depth on the face-video task", trace="E-005; D-051")
    B.p("The diagnostic battery of the working version had pointed to bias for the face task: the training AUC of the frozen feature with a linear head was only 0.837. The lever was therefore "
        "the depth of fine-tuning, with the scope and budget fixed in advance: three encoders at three learning rates and one seed, selection on validation data only, then two further seeds "
        "for the selected encoder. The hypothesis was that the larger model class, which contains the probe, lowers the training risk and that the validation AUC rises by 0.02 to 0.05 and "
        "the development AUC changes between slightly negative and +0.03. <ref tab:i1runs> lists the nine runs of the first stage.")
    B.table("i1runs", "Fine-tuning of three encoders at three learning rates, one seed (validation and development slice AUC).",
            ["Encoder", "Learning rate 1e-5", "Learning rate 3e-5", "Learning rate 1e-4"],
            [["DINOv2 small", "0.9869 / 0.9031", "0.7056 / 0.6412", "0.5871 / 0.5679"],
             ["SigLIP base", "0.9893 / 0.9210", "0.9814 / 0.8947", "0.5186 / 0.5432"],
             ["DINOv2 base", "0.9918 / 0.9036", "0.5218 / 0.5379", "0.5583 / 0.5667"]],
            [3.6, 4.2, 4.2, 4.2], ["left", "center", "center", "center"], size=9, trace="E-005 stage 1 (exploratory)")
    B.p("The lowest rate was best for every encoder. At the two higher rates five of the six runs lost most of the signal (validation AUC between 0.52 and 0.71) and only SigLIP base at "
        "3e-5 stayed high, so the learning-rate sensitivity is large, 0.40 to 0.47 in AUC between the best and the worst rate of an encoder; no non-finite loss or instability event was recorded, "
        "so these are training failures and not numerical ones. The selection on validation data (the cheaper candidate wins when the paired interval includes zero) gave the leader DINOv2 base "
        "(0.9918 [0.9847, 0.9974]), a tie set of DINOv2 base and SigLIP base (difference SigLIP minus leader −0.0025 [−0.0068, +0.0009]) and the winner SigLIP base at 1e-5 because its "
        "fine-tuning compute is smaller (203 against 238 seconds). One run (DINOv2 base at 1e-4) stopped when a data-loader worker died and was repeated once with the same configuration and seed; "
        "the error record is kept.")
    B.table("i1final", "Fine-tuned SigLIP base (three seeds) against the linear probe of the same encoder (exploratory).",
            ["Split", "Fine-tuned", "Linear probe", "Difference [95 percent interval]"],
            [["Training subset (900 clips)", "0.9994 [0.9986, 1.0000]", "0.9245 [0.9093, 0.9388]", "+0.0749 [+0.0607, +0.0902]"],
             ["Validation", "0.9899 [0.9809, 0.9967]", "0.8767 [0.8582, 0.8966]", "+0.1132 [+0.0948, +0.1299]"],
             ["Development slice", "0.9096 [0.8776, 0.9388]", "0.7396 [0.7031, 0.7832]", "+0.1700 [+0.1228, +0.2114]"]],
            [4.2, 4.0, 4.0, 4.2], ["left", "center", "center", "center"], size=9, trace="E-005 winner, seeds 0 to 2 (exploratory)")
    B.p("The seeds give a validation AUC of 0.9893, 0.9896 and 0.9908 and a development AUC of 0.9210, 0.9166 and 0.8912. The three criteria are met (lower bound of the validation gain "
        "above zero, training-subset AUC at least 0.90, lower bound of the development gain above zero) and the verdict is <i>supported</i> on this slice. The gains are larger than predicted: "
        "+0.1132 against +0.02 to +0.05 on validation and +0.1700 against at most +0.03 on the slice. The battery after the iteration shows no bias any more (training minus validation +0.0095) "
        "and a smaller shift (validation minus slice +0.0803 against +0.1052). On validation the AUC per manipulation is 1.000 (Deepfakes), 0.991 (Face2Face), 0.999 (FaceSwap) and 0.969 (NeuralTextures). "
        "Aggregation still does not help: the frame level scores 0.852 on the slice, the clip mean 0.921, and the median and the mean of the three highest 0.921 and 0.926. After temperature scaling the "
        "calibration error is 0.0174 on validation and 0.0331 on the slice, and the balanced accuracy at the validation threshold is 0.974 and 0.824. Of the 20 worst errors on the slice, 7 belong to one identity group. "
        "The slice has 15 identity groups and one kind of fake, so these numbers say nothing yet about other sources; they are the reason for the sealed test.")

    # ---- iterations 2 to 4
    B.h3("Iterations 2 to 4: speech preprocessing, augmentation and combination", trace="E-007, E-008, E-009; D-054 to D-056")
    B.p("The speech task kept its shift trigger after the first iteration: validation AUC 0.9942 and slice AUC 0.6100 (a gap of 0.38), with recording-condition cues that reverse in sign between "
        "ASVspoof 2019 and In-the-Wild. The first lever removed two of these cues by construction. Each clip is trimmed of leading and trailing silence (samples at most 0.005 of full scale) and divided by its "
        "peak before the window of 4 seconds is cut. The hypothesis was that a score that cannot use level and leading silence moves towards its content-based value out of domain, with a prediction of "
        "+0.05 to +0.15 on the slice and at most −0.01 on validation. A manipulation check was written into the criteria: after the change the single recording features must have intervals "
        "that include 0.5 on ASVspoof development, otherwise the hypothesis is not tested. It passed. The slice AUC rose from 0.6100 to 0.6723 [0.6126, 0.7584], a gain of +0.0623 [+0.0114, +0.1048], and the "
        "validation AUC fell by 0.0054 [−0.0115, −0.0006]; the verdict is supported. The energy of the clip, as a single feature, still has an AUC of 0.135 [0.054, 0.382] on In-the-Wild, so a "
        "level-related cue remains.")
    B.p("The second lever trained the head on the clean clips plus three augmented copies of each training clip, each copy with a random choice of additive noise (5 to 30 dB), a low-pass filter "
        "(cut-off 3,400 to 7,000 Hz), a synthetic room reverberation and an MP3 round trip (24 to 128 kbit/s); the ranges were fixed before the run and not tuned on the slice. The hypothesis was that "
        "a head fitted to the augmented distribution cannot rely on cues that the transformations destroy. A precondition was written: the augmentation must be a perturbation for the previous head, "
        "which it was (its AUC on an augmented validation copy was 0.6569 against 0.9889 on the clean one, while the new head scores 0.9436 against 0.9869). The slice AUC rose to 0.7455 [0.7052, 0.8121], "
        "+0.0732 [+0.0122, +0.1225] over the previous head and +0.1355 [+0.0515, +0.2088] over the working version, and the validation AUC changed by −0.0020 [−0.0043, +0.0006]; the verdict is supported.")
    B.p("The third lever averaged the logits of three tied speech encoders (HuBERT base, WavLM base plus and the Whisper small encoder), each with the pipeline of the second lever. The hypothesis "
        "was that the variance of the average falls with the correlation of the members. The mean pairwise correlation of the clip scores is 0.534 on the slice and 0.624 on validation. The average reaches "
        "0.8130 [0.7523, 0.8808] on the slice, +0.0675 [+0.0300, +0.0929] over the HuBERT head alone, and 0.9964 [0.9943, 0.9982] on validation (+0.0095); the verdict is supported, and the gain is "
        "slightly larger than the predicted +0.02 to +0.06. The price is three encoder passes at inference: on the slice the end-to-end extraction runs at 237 clips per second for HuBERT, 210 for "
        "WavLM and 64 for Whisper. <ref tab:speechres> gives the ladder of the speech task and the members.")
    B.table("speechres", "Speech task: the working version, the three levers and the members of the average (exploratory; slice of 12 speakers).",
            ["Head", "Validation AUC", "Development slice AUC", "Change on the slice"],
            [["Working version, HuBERT base", "0.9942 [0.9910, 0.9967]", "0.6100 [0.5434, 0.7256]", "baseline"],
             ["Trim and normalise (step 2)", "0.9889 [0.9831, 0.9933]", "0.6723 [0.6126, 0.7584]", "+0.0623 [+0.0114, +0.1048]"],
             ["Plus augmentation (step 3)", "0.9869 [0.9818, 0.9911]", "0.7455 [0.7052, 0.8121]", "+0.0732 [+0.0122, +0.1225]"],
             ["Average of three encoders (step 4)", "0.9964 [0.9943, 0.9982]", "0.8130 [0.7523, 0.8808]", "+0.0675 [+0.0300, +0.0929]"],
             ["Member: HuBERT base", "0.9869 [0.9818, 0.9911]", "0.7455 [0.7052, 0.8121]", "same as step 3"],
             ["Member: WavLM base plus", "0.9699 [0.9595, 0.9786]", "0.7823 [0.7172, 0.8436]", "not applicable"],
             ["Member: Whisper small encoder", "0.9951 [0.9924, 0.9973]", "0.7526 [0.6851, 0.8406]", "not applicable"]],
            [5.2, 3.8, 3.8, 3.6], ["left", "center", "center", "center"], size=9, trace="E-004, E-007, E-008, E-009 (exploratory)")
    B.p("Two limits stand with these three results. The slice has 12 speakers, so its intervals are wide and a true gain below about +0.09 cannot be shown by a single step; the steps are "
        "supported because their gains exceed this width. And the calibration error on the slice is still high after temperature scaling (0.1670 for the average), so the audio score remains a ranking score, and the page keeps its warning.")

    # ---- iterations 5 and 6
    B.h3("Iterations 5 and 6: LAV-DF, the pre-check and in-domain heads", trace="E-010, E-011, E-012; D-057 to D-059")
    B.p("LAV-DF is the only data set on the machine in which video and audio are manipulated independently and every video has an audio stream. A fixed subset was drawn from its official splits: "
        "500 videos for each of four cells (real, video fake only, audio fake only, both fake) in each of the training, validation and test splits, with the source video as the group, "
        "so each split has 2,000 videos. Only the training part (used to fit the fusion layer) and the validation part were scored; the test part is sealed. The label for the fused question is fake if either "
        "modality is fake, and the label of each modality comes from the metadata.")
    B.p("The first step was the pre-check written in advance: the frozen visual head (trained on FaceForensics++) and the frozen speech average are scored without any fitting, and a head whose lower "
        "bound against its own label is at or below 0.55 is declared near chance. The visual head failed, and failed below chance: its AUC against the video label is 0.3838 [0.3599, 0.4079] and against the "
        "fused label 0.2786 [0.2534, 0.3042]. It also scores the videos whose frames are untouched (audio fakes) as more real than the real originals (AUC 0.2876 for that cell), so the reversed sign is not a "
        "response to a visual manipulation. The code path was checked against the earlier validation scores on FaceForensics++ (largest difference 0.00012). The speech average gives 0.6476 [0.6244, 0.6701] against the audio "
        "label. The fusion was therefore not tested. A logistic regression fitted on the training part reached 0.7536 on validation only by giving the visual head a negative weight; this exploits a "
        "data-set-specific inversion and is recorded but not used.")
    B.p("The next step trained both heads in the domain: the visual encoder was fine-tuned with the recipe of iteration 1, unchanged, on 10,000 LAV-DF training videos (disjoint in source video from the "
        "fusion-fit and validation parts), and linear heads for the three speech encoders were fitted on the audio of the same videos; the temperature of each modality was fitted on the fusion-fit part. "
        "<ref tab:lavdf> shows the result and the cell-wise values.")
    B.table("lavdf", "Heads trained on LAV-DF, on its validation part (AUC with 95 percent group bootstrap interval; exploratory).",
            ["Score", "Fused label", "Own modality label", "Video-only fakes against real", "Audio-only fakes against real"],
            [["Visual head", "0.9913 [0.9871, 0.9949]", "0.7491 [0.7280, 0.7697]", "0.9905 [0.9857, 0.9945]", "0.9912 [0.9864, 0.9951]"],
             ["Speech average", "0.7480 [0.7260, 0.7689]", "0.8604 [0.8435, 0.8769]", "0.5175 [0.4836, 0.5532]", "0.8711 [0.8474, 0.8920]"],
             ["Sum of log-odds", "0.9795 [0.9726, 0.9857]", "not applicable", "not applicable", "not applicable"],
             ["Logistic regression", "0.9935 [0.9905, 0.9962]", "not applicable", "not applicable", "not applicable"]],
            [3.0, 3.5, 3.5, 3.2, 3.2], ["left", "center", "center", "center", "center"], size=8, trace="E-011 (exploratory)")
    B.figure("lavdfcells", "fig09_lavdf_cells.png", "AUC of the visual and the audio head on the LAV-DF validation subset: against the real originals for each kind of fake, and within the produced videos only.", 15.0,
             "Bars of AUC for the visual head and the audio head for three kinds of fake against real originals, and within produced videos, with intervals",
             tag="exploratory", trace="E-011 and its post-hoc control (exploratory)")
    B.p("Both heads pass the pre-check (0.7491 and 0.8604 against their own labels, lower bounds above 0.55), the logistic-regression fusion exceeds the better single head by +0.0023 [+0.0009, +0.0040] on the "
        "fused label, and the mechanical verdict of the step is <i>supported</i>. The table and <ref fig:lavdfcells> show what this number means. The visual head scores the audio-only fakes, whose frames "
        "are the original frames, as high as the video fakes (0.9912 against 0.9905 when each is compared with the real originals). A post-hoc control, labelled as such, keeps only the produced videos, "
        "that is, those that went through the data set's own pipeline: the visual head's AUC against the video label is then 0.5070 [0.4771, 0.5361], which is chance, while the speech average reaches 0.8574 [0.8394, 0.8752] "
        "against the audio label. The visual head has therefore learned to separate the original videos from the produced ones, and the fused AUC of 0.9935 is an in-data-set result that this cue drives. "
        "The part of the system that detects a manipulation on LAV-DF is the audio head.")
    B.p("One follow-up tested whether a visual signal exists at all. The visual head was trained on the produced videos only, with the audio fakes as the negatives, so that the pipeline cue is constant. "
        "Its AUC within the produced videos is 0.5169 [0.4945, 0.5392], the difference to the previous head is +0.0100 [−0.0159, +0.0372], and the verdict is <i>not supported</i>: this recipe, on this subset and "
        "with 16 face crops per video, does not recover a visual manipulation signal. The report therefore makes no claim of audio-visual deepfake detection from these results. "
        "The audio-visual mode of the system shows the visual and the audio score of a video side by side, with a warning, and no fused verdict.")

    # ---- iteration 7
    B.h3("Iteration 7: the pulse signal check", trace="E-006; D-052")
    B.p("The signal check S-1 ran on the processor on 432 real clips (up to 150 per data set: FaceForensics++ original sequences, real Celeb-DF v2 videos and real LAV-DF videos), each represented by 10 seconds of "
        "contiguous frames at the native frame rate. The spectral peak in 0.7 to 3 Hz and its signal-to-noise ratio were computed, and a time-shuffled control with at least ten shuffles per clip was pooled; "
        "its 95th percentile, 1.118 dB, is the reference. The rule written in advance required 60 percent of the real clips above the reference. The share is 130 of 432, that is 0.301 [0.260, 0.346]; "
        "per data set it is 0.280 for FaceForensics++, 0.288 for Celeb-DF v2 and 0.333 for LAV-DF, and the median signal-to-noise ratio of the real clips is −0.222 dB. The check fails, "
        "and the iteration reports exactly this: <i>the pulse signal is not recoverable from these data</i>. No separation or fusion experiment was run. The check has no ground-truth pulse, so it says that this extractor "
        "does not recover a pulse from these compressed videos, not that no pulse exists; another extractor or uncompressed video could differ.")

    # ---- baseline
    B.h3("The re-run baseline and the margin of the comparison", trace="E-013; D-060, D-061")
    B.p("The comparison with a published detector is made against a family that is re-run on the same data, split, budget and metric. The family is Xception with ImageNet weights, fine-tuned on the "
        "same FaceForensics++ frames with the recipe and budget of iteration 1; the learning rate was chosen on validation data from the same three values and three seeds were run. The selected rate, 1e-4, is "
        "the highest of the grid (validation AUC 0.9646, 0.9809 and 0.9883 for 1e-5, 3e-5 and 1e-4), so a better rate outside the grid cannot be excluded, which would favour the system in the comparison. "
        "<ref tab:baseline> gives the result.")
    B.table("baseline", "Xception baseline against the fine-tuned SigLIP base, three seeds each (exploratory).",
            ["Split", "Xception baseline", "SigLIP base fine-tuned", "Difference [95 percent interval]"],
            [["Validation", "0.9872 [0.9784, 0.9942]", "0.9899 [0.9809, 0.9967]", "+0.0027 [−0.0009, +0.0060]"],
             ["Development slice", "0.8750 [0.8435, 0.9089]", "0.9096 [0.8776, 0.9388]", "+0.0346 [+0.0136, +0.0550]"]],
            [3.6, 4.2, 4.2, 4.4], ["left", "center", "center", "center"], size=9, trace="E-013 (exploratory)")
    B.p("The seed standard deviation of the baseline is 0.0010 on validation and 0.0072 on the slice. On the slice the fine-tuned system is higher by +0.0346 [+0.0136, +0.0550], and the hypothesis that "
        "it is higher (lower bound above zero) is supported as an exploratory result. The baseline, however, also gains +0.1817 [+0.1396, +0.2166] on the slice over the working version, so most of the gain of iteration 1 comes from fine-tuning and "
        "not from the particular encoder. The margin of the non-inferiority comparison was fixed before any sealed evaluation by the rule that the margin is twice the larger of the two seed standard deviations, "
        "with a floor of 0.01, computed from the unrounded standard deviations: $\\delta$ = 0.0143.")


def final_system(B):
    B.p("The definition of the final system and the comparisons were also fixed before any sealed evaluation. The final system is the configuration after the last supported lever of each task: "
        "the fine-tuned SigLIP base for face video, the frozen OpenAI CLIP head for generated images, and the average of the three speech encoders with trimming, normalisation and augmentation for speech. "
        "The audio-visual fusion on LAV-DF is not part of it. The first confirmatory comparison is the final system against the working version, on the sealed Celeb-DF family test for face video and on the sealed "
        "In-the-Wild test for speech; the image head is the same in both, so there is nothing to compare. It is called an improvement if the lower bound of the paired interval is above zero. "
        "The second is the final face-video system against the Xception baseline on the sealed Celeb-DF family test: superior if the lower bound is above zero, non-inferior if it is above −0.0143, and, if the interval "
        "contains zero but not −0.0143, worded as not distinguished and non-inferior by the rule. All other sealed numbers are exploratory. The freeze is on Thursday 29 October 2026; "
        "none of these sealed evaluations has been run, and their cells in <ref tab:status> are marked pending.")


def software(B):
    B.p("The inference library was extended for the frozen system. A second checkpoint format records, per task, how the score is made (a linear probe on frozen features, a fine-tuned encoder, or the mean "
        "logit of an ensemble of heads), keeps the hash of every file, and checks the preprocessing identifier against the configuration recorded in the checkpoint; the audio preprocessing now includes the trimming "
        "and normalisation, so it has a new identifier, and the checkpoint of the working version still loads and is scored with its own rule. The format and its refusals are covered by 304 unit tests, which pass. "
        "A candidate checkpoint of the frozen system was built, and a parity check on the graphics processor passed: 11 training or validation items (6 FaceForensics++ videos, 3 ASVspoof files and 2 ProGAN images) "
        "were scored from the stored scores and by the real adapter from the raw files, and the largest absolute difference of the calibrated score was 0.0019 against a tolerance of 0.02. "
        "The local web page still runs the checkpoint of the working version until the freeze.")


def limits(B):
    B.h3("Limits of these results")
    B.bullets([
        "Every verdict above rests on the validation split or on a development slice and is exploratory. The slices are small: 15 identity groups for face video, two generator units for generated images and 12 speakers for speech, so the intervals are wide and a gain below about +0.09 cannot be shown on the speech slice.",
        "The development slice of the face task has one kind of fake, so the gain from fine-tuning is not shown to hold on other manipulations before the sealed test.",
        "The baseline learning rate is at the edge of the grid, one run was repeated after a data-loader error, and three seeds give only a rough standard deviation for the margin.",
        "The LAV-DF subset has 2,000 videos per split, the identities of LAV-DF are not in its metadata, and the early-stopping epoch of its audio heads was chosen on its validation part, so the audio values there are slightly optimistic.",
        "The pulse check has no ground-truth pulse, and the image task has two generator units, so its intervals are not population intervals.",
    ])
