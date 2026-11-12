"""Chapter 4, sealed-test results of the freeze run. Every number is copied from the result block of the freeze run (E-014): the three
confirmatory tests T1 to T3 are confirmatory, everything else here is exploratory. Identifiers live only in the Word comments (trace)."""


def sealed_results(B):
    B.p("The sealed run was made on 25 October 2026, after the code, the preprocessing and the checkpoints had been frozen. Each frozen iteration was evaluated once on each sealed test set under the final flag, "
        "and the ledger holds 74 entries. Every value is for three seeds (0 to 2), at the level of the clip or the image, with the 95 percent group bootstrap interval of 2,000 resamples. "
        "Only the three tests declared before the run are confirmatory; every other sealed number in this section is exploratory and says so.")
    B.h3("The three confirmatory tests", trace="E-014 result block, tests T1 to T3; D-067, D-068")
    B.p("<ref tab:sealed3> gives the three tests. In the first two, the final system is compared with the working version on the same sealed set, in a paired group bootstrap: the lower bound of the 95 percent interval decides, and the interval at "
        "the 98.33 percent level is printed beside it as a robustness line. In the third, the final face-video system is compared with the re-run Xception baseline at the selected rate of 3e-4, with the margin 0.0100.")
    B.table("sealed3", "The three confirmatory tests on the sealed test sets (AUC and difference with 95 percent group bootstrap intervals; the last two columns give the robustness interval at the 98.33 percent level and the verdict).",
            ["Test", "Sealed set", "Final system", "Compared with", "Difference, 95 percent", "98.33 percent interval", "Verdict"],
            [["Face video, final system against working version", "Celeb-DF family, 7,650 videos, 344 groups", "0.8218<br/>[0.8020, 0.8393]", "0.6238<br/>[0.6015, 0.6486]", "+0.1980<br/>[+0.1688, +0.2242]", "[+0.1627, +0.2303]", "improved; robust"],
             ["Speech, final system against working version", "In-the-Wild, 24,788 clips, 42 speakers", "0.9148<br/>[0.8896, 0.9383]", "0.7516<br/>[0.6802, 0.8035]", "+0.1633<br/>[+0.1177, +0.2266]", "[+0.1085, +0.2481]", "improved; robust"],
             ["Face video, final system against Xception baseline (margin 0.0100)", "Celeb-DF family, 7,650 videos, 344 groups", "0.8218<br/>[0.8020, 0.8393]", "0.8042<br/>[0.7841, 0.8223]", "+0.0176<br/>[+0.0025, +0.0327]", "[−0.0011, +0.0355]", "superior at 95 percent; not robust at 98.33 percent; non-inferior at both"]],
            [3.2, 2.6, 2.1, 2.1, 2.2, 2.0, 2.2], ["left", "left", "center", "center", "center", "center", "left"], size=8, keep=False,
            trace="E-014 result block, tests T1 to T3 (confirmatory, sealed)")
    B.p("For face video the final system reaches an AUC of 0.8218 [0.8020, 0.8393] against 0.6238 [0.6015, 0.6486] for the working version, a difference of +0.1980 [+0.1688, +0.2242]. The lower bound is above zero, "
        "so the verdict under the declared rule is <i>improved</i>, and the interval at the 98.33 percent level, [+0.1627, +0.2303], is also above zero. For speech the final system reaches 0.9148 [0.8896, 0.9383] "
        "against 0.7516 [0.6802, 0.8035], a difference of +0.1633 [+0.1177, +0.2266] and, at the 98.33 percent level, [+0.1085, +0.2481]: <i>improved</i>, and robust to the correction. The speech intervals are wide, "
        "because the 42 speakers of the sealed set are the resampling unit.")
    B.p("For the comparison with the baseline, the difference is +0.0176 [+0.0025, +0.0327]. The lower bound is above zero, so by the declared 95 percent rule the final face-video system is <i>superior</i> to the re-run Xception "
        "baseline on this set. The robustness interval, [−0.0011, +0.0355], contains zero, so the superiority is not robust to the Bonferroni level. The lower bound is above −0.0100 at both levels, so the system is non-inferior "
        "at both. The wording that the data support is therefore: superior at the 95 percent level, not robust to the 98.33 percent level, non-inferior at both. The result concerns one baseline family at 224 pixels "
        "and one sealed set, and it supports no claim beyond this.")
    B.p("The predictions written before the run held for the video task, including the range for the difference to the baseline. Two predictions for speech failed, both upward: the final speech system was predicted "
        "at 0.60 to 0.85 and reached 0.9148, and the working version was predicted at 0.50 to 0.70 and reached 0.7516. The sealed video value is below the development value of the same system (0.8218 against 0.9096), "
        "and the sealed speech value is above it (0.9148 against 0.8130). The sets differ in identities and speakers, so these are different measurements and not a fall or a rise of one quantity, "
        "but for speech the direction is the opposite of what the small development slice suggested.")
    B.h3("The staircase on the sealed sets", trace="E-014 result block, staircase rows (exploratory); D-068")
    B.p("Each frozen iteration was also scored once on every sealed set. Apart from the Celeb-DF family row of <ref tab:stairvideo> and the working-version and iteration 4 values of the In-the-Wild row of <ref tab:stairspeech>, which carry the tests above, "
        "every number in this section is exploratory. <ref tab:stairvideo> gives the video and image tasks. The face-video model is the same in iterations 1 to 4, because the later levers act on speech, so one column stands for the four, "
        "and the image head is unchanged in all iterations, so its three rows are the same everywhere. The Celeb-DF++ videos have no real videos of their own, so Celeb-DF++ alone has one class and no AUC; "
        "its row uses the Celeb-DF v2 real videos, and the family set mixes the Celeb-DF v2 test videos and the Celeb-DF++ fakes against those reals.")
    B.table("stairvideo", "Sealed staircase for face video and generated images (AUC, three seeds, 95 percent group bootstrap interval; exploratory except the Celeb-DF family row, which carries tests T1 and T3).",
            ["Sealed set", "Items (groups)", "Working version", "Iterations 1 to 4", "Xception, 3e-4"],
            [["Celeb-DF family", "7,650 (344)", "0.6238<br/>[0.6015, 0.6486]", "0.8218<br/>[0.8020, 0.8393]", "0.8042<br/>[0.7841, 0.8223]"],
             ["Celeb-DF v2 test alone", "4,801 (344)", "0.6322<br/>[0.6018, 0.6658]", "0.8999<br/>[0.8759, 0.9210]", "0.8527<br/>[0.8237, 0.8777]"],
             ["Celeb-DF++ fakes with the v2 reals", "3,593 (344)", "0.6119<br/>[0.5821, 0.6427]", "0.7105<br/>[0.6877, 0.7331]", "0.7351<br/>[0.7083, 0.7599]"],
             ["DeepFakeDetection", "3,431 (28)", "0.5927<br/>[0.5629, 0.6258]", "0.8797<br/>[0.8566, 0.9041]", "0.8847<br/>[0.8655, 0.9052]"],
             ["FaceForensics++ test (in-dataset)", "700 (70)", "0.7946<br/>[0.7667, 0.8243]", "0.9899<br/>[0.9826, 0.9959]", "0.9895<br/>[0.9823, 0.9952]"],
             ["ProGAN test (in-dataset images)", "8,000 (20)", "1.0000<br/>[0.9999, 1.0000]", "same", "not applicable"],
             ["CNN synthesis subset (images)", "48,805 (16)", "0.9316<br/>[0.8933, 0.9694]", "same", "not applicable"],
             ["Diffusion test set (images)", "10,000 (10)", "0.9354<br/>[0.8967, 0.9665]", "same", "not applicable"]],
            [4.8, 2.6, 3.0, 3.0, 3.0], ["left", "center", "center", "center", "center"], size=8, keep=False,
            trace="E-014 result block, staircase rows for video and images (exploratory except the Celeb-DF family row, T1 and T3)")
    B.p("On the Celeb-DF v2 test videos alone the fine-tuned model reaches 0.8999 [0.8759, 0.9210], and on DeepFakeDetection 0.8797 [0.8566, 0.9041] against 0.5927 [0.5629, 0.6258] for the working version; "
        "on the in-dataset test of the training data set it reaches 0.9899 [0.9826, 0.9959]. The Xception baseline is not lower than the final system on every set: on DeepFakeDetection and on the Celeb-DF++ fakes with the v2 reals its values, 0.8847 and 0.7351, "
        "are above those of the final system, 0.8797 and 0.7105, with overlapping intervals. No paired test was declared for these sets, so they are descriptive and not a finding in either direction. "
        "The pooled family value of 0.8218 is lower than the value on the Celeb-DF v2 test videos alone because the Celeb-DF++ fakes with the v2 reals are at 0.7105; "
        "the cause of that lower value, for example the newer generators of Celeb-DF++, was not tested here.")
    B.p("<ref tab:stairspeech> gives the speech task. The working version and iteration 1 are the same on audio. On In-the-Wild the AUC rises in each step, "
        "0.7516, 0.8125, 0.8672 and 0.9148, in the order of the lever that trims and normalises the clip, the augmentation, and the average of three encoders. In the training corpus the order is not monotone: the AUC on the in-dataset test is 0.9684 for the working version, "
        "falls to 0.9632 and 0.9541 for the first two levers and is 0.9835 for the final system, as the validation AUC did in the development runs.")
    B.table("stairspeech", "Sealed staircase for speech (AUC, three seeds, 95 percent group bootstrap interval; exploratory except the In-the-Wild values of the working version and of iteration 4, which carry test T2).",
            ["Sealed set", "Clips (groups)", "Working version", "Iteration 2", "Iteration 3", "Iteration 4"],
            [["In-the-Wild", "24,788 (42)", "0.7516<br/>[0.6802, 0.8035]", "0.8125<br/>[0.7361, 0.8560]", "0.8672<br/>[0.8320, 0.8989]", "0.9148<br/>[0.8896, 0.9383]"],
             ["ASVspoof 2019 LA test (in-dataset)", "71,237 (67)", "0.9684<br/>[0.9564, 0.9777]", "0.9632<br/>[0.9535, 0.9717]", "0.9541<br/>[0.9452, 0.9626]", "0.9835<br/>[0.9798, 0.9872]"]],
            [4.2, 2.4, 2.45, 2.45, 2.45, 2.45], ["left", "center", "center", "center", "center", "center"], size=8, keep=False,
            trace="E-014 result block, staircase rows for speech (exploratory except the In-the-Wild values of I-0 and I-4, T2)")
    B.h3("LAV-DF on the sealed subset", trace="E-014 result block, I-5 to I-6bp rows (exploratory); D-068")
    B.p("The sealed subset of LAV-DF has 2,000 videos from 1,736 source groups, and its subset of produced videos has 1,500 videos from 1,335 groups. <ref tab:stairlavdf> gives the heads of the audio-visual iterations. "
        "The pattern of the development runs repeats. The frozen visual head is below chance (0.3226 [0.2953, 0.3490]). The head trained on LAV-DF reaches 0.9921 [0.9880, 0.9956] on the whole subset and the fusion 0.9945 [0.9917, 0.9969]. "
        "Within the produced videos, where the cue that separates original videos from produced ones is constant, the visual head is at 0.5032 [0.4746, 0.5320] and the head trained on the produced videos only at 0.5203 [0.4973, 0.5429], "
        "while the audio head is at 0.8480 [0.8281, 0.8670]. The sealed numbers therefore support no claim of audio-visual deepfake detection, and the system still shows the two scores side by side without a fused verdict.")
    B.table("stairlavdf", "Sealed LAV-DF subset (AUC, three seeds, 95 percent group bootstrap interval; exploratory).",
            ["Head", "Videos scored", "Label", "AUC [95 percent interval]"],
            [["Frozen visual head", "2,000 (all)", "fake if either modality is fake", "0.3226 [0.2953, 0.3490]"],
             ["Frozen speech average", "2,000 (all)", "fake if either modality is fake", "0.6462 [0.6188, 0.6723]"],
             ["Visual head trained on LAV-DF", "2,000 (all)", "fake if either modality is fake", "0.9921 [0.9880, 0.9956]"],
             ["Speech heads trained on LAV-DF", "2,000 (all)", "fake if either modality is fake", "0.7344 [0.7133, 0.7549]"],
             ["Fusion of the two", "2,000 (all)", "fake if either modality is fake", "0.9945 [0.9917, 0.9969]"],
             ["Visual head, produced-videos-only recipe", "2,000 (all)", "fake if either modality is fake", "0.5216 [0.5007, 0.5428]"],
             ["Visual head trained on LAV-DF", "1,500 (produced)", "video label", "0.5032 [0.4746, 0.5320]"],
             ["Speech heads trained on LAV-DF", "1,500 (produced)", "audio label", "0.8480 [0.8281, 0.8670]"],
             ["Visual head, produced-videos-only recipe", "1,500 (produced)", "video label", "0.5203 [0.4973, 0.5429]"]],
            [5.4, 3.0, 4.6, 3.4], ["left", "center", "left", "center"], size=8, keep=False,
            trace="E-014 result block, I-5, I-6, I-6b, I-6p, I-6bp rows (exploratory)")
