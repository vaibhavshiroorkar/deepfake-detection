"""Chapter 4, later sections: the analysis platform, software testing and verification, and the experiments after the freeze.

Sources: src/dfd/platform (read on 2026-10-26: schema.py, module.py, coordinator.py, pipeline.py, modules/*.py, provenance/modules.py,
dashboard/serve.py); docs/EXPERIMENTS.md E-015, E-018 to E-025; docs/DECISIONS.md D-069 to D-078; docs/STATUS.md (test records of
2026-10-26); docs/RESEARCH_INTEGRITY.md; docs/SETUP.md. Identifiers appear only in Word comments (trace), never in visible text.
Every number is a logged one (docs/NUMBERS.json); test counts are engineering records, not detector results.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG10 = os.path.join(ROOT, "paper", "figures", "fig10_platform.png")
FIG11 = os.path.join(ROOT, "paper", "figures", "fig11_dashboard.png")


# ----------------------------------------------------------------------------------------------------------------------
def platform(B):
    B.h2("The Analysis Platform", trace="D-069, D-072, D-073, D-074; src/dfd/platform (schema.py, module.py, coordinator.py, pipeline.py)")
    B.p("After the freeze the frozen detectors were placed inside a modular analysis platform. The inference library of the working version returns one score per modality, which is "
        "the right form for the three tasks that were measured, but it has no place for evidence of other kinds, such as the agreement between the lips and the speech, or a signed "
        "statement of where a file came from, and it invites a reader to treat a high score as a verdict on the whole file. The platform keeps the frozen detectors unchanged and "
        "adds a second layer around them: each capability is a module, each module reports what it found within its own tested scope, and no number is ever combined into one overall "
        "probability that the media is fake. Emotion and expression analysis was removed from the scope at this point; lip synchronisation stayed. The platform is a separate package, so "
        "the frozen inference library and its web page were not changed. <ref fig:platform> shows the structure.")
    B.figure("platform", FIG10, "Structure of the analysis platform. The coordinator runs every registered module; each module appears once in the report, in one of three sections, "
             "with its status and, if it completed, its finding, tested scope and limitations. There is no global score.", 15.5,
             "Diagram: an uploaded file is validated and probed, a coordinator runs modules grouped into authenticity, cross-modal and provenance sections, "
             "and a report of schema version 2.1 feeds a local dashboard", tag="schematic", trace="src/dfd/platform; D-069")

    B.h3("Module interface and coordinator")
    B.p("A module declares a module identifier, a version, its section and a time limit, and it implements two functions. The first, applicability, receives a probe of the file (media type, "
        "duration, whether a video and an audio stream exist, size, hash) and answers applicable, not applicable (the input lacks what the module needs, for example a still image for a "
        "speech module) or unsupported (the capability is not available here, for example because a weights file is not installed). A field of the probe that is not known is left empty "
        "and never read as absent. The second function, run, does the work and returns a module result; it may still return not applicable when the answer needs the decoded media, "
        "for example when no face is found.")
    B.p("The coordinator runs the registered modules one at a time, because the graphics processor is shared, and gives each its own time limit. Nothing a module does can break the report: "
        "an exception becomes a failed entry with the exception type, a module that does not answer in time becomes timed out, and a result that does not satisfy the report contract is "
        "replaced by a failed entry that names the broken rule. Python cannot stop a thread, so a timed-out module is abandoned: the coordinator sets a cancellation flag that the module "
        "may poll, discards any late result or exception, and gives every module a view of the shared cache that ignores writes once the module is cancelled, so a late thread cannot change "
        "the report or what other modules read. While an abandoned module that uses the graphics processor is still running, a second such module is not started and is reported as skipped "
        "with the reason; a total time limit for the whole analysis turns modules that were not started into skipped entries as well. Every registered module appears in the report exactly once.")

    B.h3("Report schema version 2 (extended in 2.1)")
    B.p("The report lists one entry per module with its status, its finding, an optional score with a sentence that says what the number is, whether it is calibrated and the threshold "
        "used, the model block (name, version, weight hash, licence and the evaluation it rests on), the tested scope, the limitations, warnings, errors and observations, and the "
        "processing time. <ref tab:statuses> gives the six statuses and <ref tab:findings> the findings that each section allows.")
    B.p("Three rules keep the report from saying more than the evidence. There is no global score: the contract checker rejects a report that carries a fused score, an overall "
        "probability or a verdict field, and no section has a finding that means genuine; the strongest negative finding is no indication within the tested scope. Every report ends with "
        "one fixed statement: “This report lists what each module found within its tested scope. A module that found nothing, did not run, was unsupported or was inconclusive does not show "
        "that the media is authentic. No single probability that the media is fake is given.” And the state of the report is failed when no module completed, partial when any module "
        "failed, timed out, was unsupported or was skipped by the coordinator, and ok otherwise, so a report in which nothing was analysed is never ok. When authenticity modules disagree, "
        "the report adds a note that the modules cover different kinds of manipulation, so the disagreement is not necessarily a contradiction. A development stand-in result carries no "
        "model block and no calibrated score, as in the inference library.")

    B.table("statuses", "Statuses of a module in the report.",
            ["Status", "Meaning", "Finding and score"],
            [["completed", "the module ran and returned a valid result", "a finding of its section; a score only with its meaning"],
             ["not_applicable", "the input lacks what the module needs (no face, no audio stream, a still image for a video module)", "none; a reason is required"],
             ["unsupported", "the capability is not available here (weights or trained head not installed, incompatible head)", "none; a reason is required"],
             ["failed", "an exception, a failed applicability check or a result that breaks the contract", "none; an error entry is required"],
             ["timed_out", "no result within the time limit of the module", "none; an error entry is required"],
             ["skipped", "not started: total time limit reached, or a timed-out module still holds the graphics processor", "none; a reason is required"]],
            [3.0, 8.4, 5.0], ["left", "left", "left"], size=10, trace="src/dfd/platform/schema.py STATUSES; coordinator.py")
    B.table("findings", "Findings allowed in each section of the report. None of them means that the media is genuine.",
            ["Section", "Findings", "Modules"],
            [["authenticity", "indicates_manipulation; no_indication_within_tested_scope; inconclusive", "face_video, image_detector, speech, generated_video"],
             ["cross_modal", "consistent; potential_mismatch; inconclusive", "lip_sync"],
             ["provenance", "credentials_valid; credentials_invalid; no_credentials_found; inconclusive", "media_metadata, content_credentials"]],
            [2.6, 8.0, 5.8], ["left", "left", "left"], size=10, trace="src/dfd/platform/schema.py FINDINGS")
    B.h3("Modules that wrap the frozen detectors")
    B.p("Three modules wrap the frozen system without changing it. One adapter is built per process from the frozen checkpoint, the file is analysed once and the result is cached under "
        "the hash of the file, so the three modules never score a file twice; a missing checkpoint is an error and never falls back to a stand-in. The face_video module reports the "
        "calibrated score of the fine-tuned SigLIP base head on the largest face in up to 16 frames, with the threshold of the checkpoint. The image_detector module reports the path "
        "taken: a face crop goes through the same video-trained head, and its score is marked as not calibrated for still images, because that head was never evaluated on still "
        "face-swapped photos; a whole image without a face goes through the frozen CLIP head of the generated-image task. The speech module reports the mean logit of the three speech "
        "heads on the first window of about 4 seconds, marked as not calibrated, because the calibration of the audio scores stayed exploratory. Each module states its tested scope, "
        "quotes the sealed results of its detector in its evaluation field and lists its limitations, among them the sealed AUC below 0.5 of the face head on LAV-DF. A parity check "
        "compared the three wrapped modules with the stored scores of the freeze parity check on 11 non-sealed files: the largest difference was 0.0019 against a tolerance of 0.02. "
        "One limitation was found in the demonstration run and documented, not changed: the speech module has no speech check of its own, so a synthetic 440 Hz tone receives a speech "
        "finding (no indication within the tested scope, score 0.0) although such input lies outside what the speech detector was trained on; the lip-sync module reports the same audio "
        "as not speech-like. A speech gate was not added, because it would change the decision of a frozen detector with a heuristic that was not evaluated for it.")
    B.p("The routing of still images left a gap: an image in which a face is found goes only to the face-swap head, so a fully generated portrait was never shown to the "
        "generated-image model. An additional module, generated_image, closes the gap for display without changing the frozen path. When, and only when, the image analysis took "
        "the face-crop path, it passes the whole image to the frozen generated-image head through the adapter's own whole-image function, the one used for images without a face, so "
        "the number is the score the frozen model gives that image. The module is experimental: it never makes a finding, applies no decision line, and its score is never the key "
        "figure of its card, because the generated-image head missed most generated faces when it was measured on images that contain a face (see the later subsection on modern generators). On the demonstration still image the image analysis result was "
        "unchanged (0.999983 on the face crop) and the new module reported a low score without a finding.")

    B.h3("Lip-sync module")
    B.p("The lip_sync module measures the temporal agreement between the speech and the lips of one tracked face. The network and the offset search are ports of the public SyncNet "
        "reference code (MIT licence); the weights file comes from the same source, which states no licence for it, so its use is limited to this academic prototype. The steps are: the "
        "audio at 16 kHz, at most the first 30 s; at least 2 s of audio; enough audible audio; a cheap speech-likelihood heuristic from spectral flatness and loudness variation, named a "
        "heuristic and not a speech detector; a YuNet face track at 25 frames per second; and the offset in frames, within 15 frames either way, that minimises the embedding distance, "
        "together with its confidence. The decision rule is: inconclusive when the clip is too short, the audio is silent or not speech-like, two or more faces share the frames (the "
        "speaker is not established), the confidence is below 3.0, or the offset lies at the edge of the search range; otherwise consistent when the offset lies in the band of −6 to +2 "
        "frames, and potential_mismatch outside it; not applicable without an audio stream or without a face; unsupported when the weights file is not installed.")
    B.p("The band and the confidence minimum were set by experiments described in the section on the experiments after the freeze. The module is weak evidence by design: its "
        "vocabulary has no finding that says manipulated, a mismatch can come from dubbing, bad muxing or a variable frame rate, and a fake that was generated to be synchronised passes. "
        "Its score is the confidence rescaled for display and is not a probability.")

    B.h3("Provenance: file metadata and Content Credentials")
    B.p("Two modules form the provenance section; neither is a detector and neither has a model. The media_metadata module lists container, stream and tag properties read with "
        "ffprobe and a few descriptive flags (for example a missing creation time, or the name of the tool that last wrote the file), and its finding is always inconclusive, because "
        "metadata can be stripped, rewritten or copied. The content_credentials module checks C2PA Content Credentials. When the optional c2pa-python package is installed (version "
        "0.38.0 was used) the manifest is validated by the library, which checks the claim signature and the assertion and data hashes; remote manifest fetching and certificate "
        "revocation checks are off. <ref tab:c2pa> gives the mapping from what the library reports to the finding.")
    B.table("c2pa", "Findings of the Content Credentials module.",
            ["Situation", "Finding"],
            [["c2pa-python not installed", "byte scan for C2PA marker bytes only: inconclusive if markers occur, no_credentials_found otherwise; nothing is verified"],
             ["Manifest valid, signer not on a configured trust list (the default: no list is bundled)", "inconclusive, with the observation that integrity was checked and the signer is not established"],
             ["Manifest valid and the signer chains to a trust anchor configured by the operator", "credentials_valid, relative to that configuration"],
             ["File or manifest changed after signing", "credentials_invalid"],
             ["No manifest", "no_credentials_found"],
             ["Unsupported file type or unreadable manifest", "inconclusive, with a warning naming the reason"]],
            [7.4, 9.0], ["left", "left"], size=10, trace="src/dfd/platform/provenance/modules.py; docs/SETUP.md section 4; D-069")
    B.p("A valid credential attests to claims that the signer made about the file at signing time; it does not prove that the content is authentic, and a self-made certificate can sign "
        "anything, which is why an untrusted signer is inconclusive. Absence of credentials proves nothing, because many editing and sharing paths remove them; credentials held in a "
        "separate file or a cloud store, and invisible watermarks, are not checked. A declared generative source in a manifest is reported as a statement of the signer, not as a detection.")

    B.h3("Generated-video module")
    B.p("The generated_video module looks at the whole picture, independent of faces, for video produced by a text-to-video or image-to-video generator. The clip is re-encoded by one "
        "fixed recipe (the first 2.0 s, 7 frames per second, a 224 by 224 centre crop, fixed keyframe and B-frame structure, no metadata and no audio), 8 frames go through the frozen "
        "DINOv2 small encoder <c dinov2>, and a logistic regression on the mean class-token embedding scores the clip. This head was selected by a rule fixed before its experiment; it is "
        "an appearance baseline and not the ReStraV method <c restrav>, whose temporal-geometry features were tried as another candidate and were at chance in the same experiment.")
    B.p("The head did not generalise to a held-out benchmark (the section on the experiments after the freeze gives the numbers), so the module is experimental and its finding is always "
        "inconclusive. A gate in the code enforces this: a definitive finding needs the boolean validation flag in the head file, the SHA-256 of the exact head file in a table of approved "
        "heads that only a reviewed code change can extend, a finite score and threshold, and evaluation and limitation texts that do not contradict validation. The table is empty, so "
        "every head, including the installed one, gives inconclusive with the warning that the capability is experimental and unvalidated; the score is shown for inspection only and no "
        "threshold is shown. The gate hashes the same bytes that it loads. A missing head, a head with another feature size or a head trained under another preprocessing protocol makes "
        "the module unsupported, and non-finite features or scores make it fail.")

    B.h3("Dashboard", trace="D-076; src/dfd/platform/dashboard")
    B.p("A local dashboard, served by Flask on 127.0.0.1 only, presents the report to a reader who need not know how the models work. In live mode it opens on an upload panel "
        "(drag and drop or a file picker, the accepted types and the 512 MB limit read from the inspector of the frozen library, and a preview played in the browser); nothing is "
        "analysed and no report of any kind is shown until the user presses Analyse. The result opens with a description of the file, then a sentence that only counts the checks by "
        "what they found, what is known about faces and speech, and the statement that no overall score is given. The checks follow in groups by meaning: merits attention, nothing "
        "found within the tested scope, inconclusive, file information, experimental, did not apply, not available and did not complete. Each check shows a plain headline, what it "
        "checks, at most one rounded number and what the result does not establish; the explanation and the technical details (identifiers, raw scores and thresholds, model, hashes "
        "and evaluation lines) open on request. The experimental generated-video check is shown in its own group with an EXPERIMENTAL badge and its score only as an experimental "
        "signal. Invented scenarios stay available on request under a hatched MOCK banner. The colours follow a fixed palette in which no green stands for nothing found; every text "
        "colour pair meets the WCAG AA contrast ratio in light and dark mode, which a test computes. <ref fig:dashboard> shows the top of a live report. In live mode the server accepts uploads into a single worker, refuses a "
        "new one while 4 analyses are queued or running, and deletes every uploaded file when the analysis returns. The frozen web page of the working version stays as it was.")
    B.figure("dashboard", FIG11, "Top of a live report on the dashboard (demonstration input 3, a real LAV-DF training clip, 27 October 2026), taken before the per-module "
             "verdict labels described below were added: file description, counting "
             "summary without an overall score, and the first group of checks. A screenshot of the software, not a result.", 14.0,
             "Screenshot of the dashboard: a live-analysis strip, the file description, a summary that counts checks by what they found and says no overall "
             "score is given, and cards for the face manipulation and synthetic speech checks", tag="screenshot", trace="docs/test_evidence/2026-10-27_redesign")

    B.h3("Explanations from the stored result", trace="D-076; src/dfd/platform/explain.py; tests/platform_explain")
    B.p("Report schema version 2.1 adds three optional parts: an explanation for each module, a summary of the report, and evidence that a module has already computed. The "
        "explanation is written on the server by fixed templates, one per module and outcome, from the fields of the module's own result; it says what was checked, what was observed, "
        "what it means, what it does not establish and the main limitations, and every sentence lists the fields it was built from. A validator rejects a sentence whose numbers are "
        "not in its cited fields, a cited field that does not exist, a sentence of model output on a module that did not run, wording such as authentic or proves, an experimental "
        "score shown as a headline number and an unmarked stand-in result; the summary must equal its recomputation from the module results. The evidence is display only: the "
        "lip-sync module returns the 31 distances of its offset search, which it computed before and discarded, and the generated-video module returns the 8 frames its encoder "
        "received. Faces and speech, which the probe cannot know before the modules run, are reported from module results with their source, or as not established; speech is never "
        "inferred from the presence of an audio stream. When the lip-sync check finds the audio unlike speech, the speech explanation says so, while the speech result itself is "
        "unchanged. These explanations describe the stored result faithfully; they are not an account of how a network reached its score, and no language model writes them.")

    B.h3("Per-module verdicts and uncertainty", trace="D-087; src/dfd/platform/verdict.py, verdict_evidence.py; tests/platform_verdict")
    B.p("Each completed module result carries an additive assessment: a verdict label, a tier (detector, limited, experimental, provenance or context), the basis, the raw score with "
        "its kind (model output or ranking score), a calibration statement, an uncertainty level with a sentence of text, and evidence entries tied to stored result files, each of which "
        "a unit test checks against its file. There is no report-level verdict and no fused score: the schema still rejects keys such as an overall score or a probability of fake, and "
        "the report keeps only counts of checks per group. A label is offered only where the stored evidence justifies it, as <ref tab:verdicts> shows. The reason for the face-video "
        "label is a measurement: on the sealed half of FakeAVCeleb the frozen video threshold had specificity 0.996 but sensitivity 0.480, so a flag is rarely a false alarm there but "
        "about half of the fake videos are missed, and the absence of a flag is therefore reported with high uncertainty. That specificity belongs to one sealed set: computed afterwards "
        "from the stored sealed scores, with no model run, the same threshold flagged about 0.11 of the real videos of the Celeb-DF family and 0.14 of those of DFD. A flag is therefore "
        "labelled Possible manipulation, with high uncertainty, as a possible sign and not a determination, and the check is not described as reliable on the strength of one set. For the same reason a low score from the whole-image model is "
        "not reassuring: on modern generated images it missed 0.7395 at its threshold (macro AUC 0.7039).")
    B.table("verdicts", "Verdict labels of each module. The label depends on whether the stored threshold comparison flags the file.",
            ["Module", "Flag", "No flag", "Uncertainty without a flag"],
            [["Face video", "Possible manipulation", "No strong evidence of manipulation detected", "high"],
             ["Image, face crop", "Possible manipulation (weak evidence)", "Inconclusive", "stated in the card"],
             ["Image, whole image", "Possible AI-generation (weak evidence)", "Inconclusive", "stated in the card"],
             ["Speech", "Possible manipulation (weak evidence)", "No strong evidence", "moderate"],
             ["Lip sync", "Possible audio-visual mismatch (weak evidence)", "No audio-visual mismatch found", "stated in the card"],
             ["Experimental modules", "Experimental signal observed, no reliable verdict available", "same label at every score", "not applicable"]],
            [3.4, 4.8, 4.8, 3.4], ["left", "left", "left", "left"], size=9, trace="D-087; src/dfd/platform/verdict.py; tests/platform_verdict")
    B.p("The scores of the face-video, image and speech detectors are temperature-scaled on each detector's validation split, so they behave like probabilities only for data like "
        "that split; the later sealed and modern-generator measurements show that they do not elsewhere. The platform therefore says score, not calibrated probability, and shows the raw "
        "value only under technical details, labelled model output or ranking score and never as a percentage. The experimental heads give uncalibrated ranking scores with no threshold "
        "applied. They cannot create a finding for two reasons: no experimental head passed its own acceptance test, and the approval table of the generated-video module is empty, so the "
        "code refuses a finding from any head file whose hash is not in it. The label Likely AI-generated exists in the vocabulary, but no rule can reach it, and a test pins this. "
        "The layer fails closed: it recomputes the comparison of score and threshold itself, and a missing, non-finite or out-of-range score, a missing threshold, or a stored finding "
        "that disagrees with the comparison gives Inconclusive with the reason, never a guess. A failed, timed-out or skipped module gives Analysis failed, and an unsupported input "
        "gives Unsupported input. On the dashboard a card shows the label, the headline, supporting signals and evidence, model and modality, limitations and uncertainty, in that order; "
        "experimental cards use a separate dashed style with no detector colour.")

    B.h3("Evaluation Lab", trace="D-088; src/dfd/platform/evallab; tests/platform_evallab")
    B.p("The Evaluation Lab is a separate page of the local dashboard, next to the page for analysing a file, with an interface under /api/evallab. It re-analyses recorded score files "
        "(run, iteration, seed, modality, dataset, split, sample, group, label and score). It runs no model, trains nothing, writes no checkpoint, never reads or writes the approval "
        "table or any threshold, and writes nothing to the evaluation ledger, so a good result in the lab is not a product approval. The workflow has five steps: choose a score file "
        "and a block (a dataset and split shown with its role: training, validation, development slice or final held-out); review the settings (model iteration, runs and seeds, "
        "threshold source, purpose, sample counts and class balance); run, with progress and failures shown; read the results (AUC, equal error rate, accuracy, balanced accuracy, "
        "precision, recall, specificity and the two error rates, with group-bootstrap intervals, a confusion matrix, an ROC curve, an optional breakdown, warnings and skipped rows); and "
        "compare with a baseline run and export JSON or Markdown.")
    B.bullets([
        "<b>Provenance and frozen configuration.</b> The SHA-256 of the score file, its row count and the git state go into every run record, and the configuration and its SHA-256 are "
        "frozen into the record before any metric is computed, so identical configurations give identical metrics.",
        "<b>Test-set isolation.</b> A final held-out split runs only as a re-analysis of an evaluation already in the ledger, only with the purpose final re-analysis, and only if the "
        "SHA-256 of the score file equals the hash recorded in that ledger entry, so an edited copy cannot pass. The lab never makes a first look at sealed data.",
        "<b>Thresholds.</b> A threshold is fitted only on the validation rows of the same runs (maximum balanced accuracy, the project's rule) or given with its provenance; it is never "
        "fitted on the evaluated rows.",
        "<b>Leakage checks.</b> Before computing, the lab looks for a group shared between the evaluated block and the training or validation rows (refused for a final split, warned "
        "otherwise), duplicate sample identifiers, and duplicate file hashes across splits when the file records them.",
        "<b>Comparisons.</b> Only on identical samples, by a paired group bootstrap; otherwise the lab reports that the comparison is unavailable and why.",
    ])
    B.p("The known limit is stated in every run record: when a score file does not record its training data, the lab cannot prove that the training data excluded the evaluated items. "
        "On 28 October 2026 an engineering check exercised the safeguards. Re-analysis of the frozen final system on the sealed Celeb-DF family reproduced the logged AUC of 0.8218; a "
        "first look at a sealed split labelled as development was refused; and the ledger file was byte-identical afterwards. This is a record of software behaviour, not a new "
        "detection result.")


# ----------------------------------------------------------------------------------------------------------------------
def testing(B):
    B.h2("Software Testing and Verification", trace="docs/TESTING_RECORD.md sections 1 to 7 and section 10 and the logs in docs/test_evidence/2026-10-27_redesign/ and 2026-10-26/; docs/SETUP.md section 9")
    B.p("Testing of the software is kept apart from the experimental evaluation. A test establishes that the code behaves as specified on constructed inputs or that the application "
        "runs end to end; it says nothing about how well a detector separates real from fake media, which only an experiment on held-out data can show. The current record was made on "
        "28 October 2026, after the per-module verdict layer and the Evaluation Lab were added, in the project environment (Python 3.13.15, synced from the lock file with the C2PA extra) on the development machine, with every count taken from a saved log. "
        "<ref tab:testing> summarises it; results of earlier runs that were not repeated on the final state of the code are marked as earlier records.")
    B.bullets([
        "<b>Unit and contract tests.</b> Data loaders and split guards, the frozen inference library and web page, the report contract (including the rejection of a global score), the "
        "coordinator (exceptions, timeouts, invalid results, late results of abandoned modules, the total time limit), the three wrappers with test doubles for the adapter, the lip-sync "
        "decision rule and missing weights, the generated-video gate (an unapproved head stays inconclusive at every score, metadata cannot authorise a finding, non-finite values fail), "
        "the Content Credentials module with a test certificate authority created for the test and without the package, the metadata module, the explanation layer (numbers from "
        "cited fields, wording by status, tampered explanations that must be caught), the verdict layer (labels by rule, fail-closed cases, evidence entries checked against stored files) and the Evaluation Lab (frozen configuration, test-set isolation, leakage checks), the contrast of the palette, and the dashboard in a real browser (upload-first opening, grouping, "
        "the explanation on screen equal to the stored one, keyboard reach). The "
        "suite (710 passed, 2 skipped) ran with the graphics processor hidden; 13 data-dependent tests that read the real dataset inventories were deselected and are not verified in this record.",
        "<b>Opt-in tests on the graphics processor.</b> A parity test of the three wrapped detectors against the stored scores of the freeze parity check on 11 non-sealed files "
        "(largest difference 0.0019, tolerance 0.02), and a live test of the face path.",
        "<b>Application run.</b> The dashboard was started in live mode and given the 8 inputs of the demonstration (six media files of every kind and two error cases) by driving "
        "the page in the Edge browser; every report that came back was checked against the schema and every number shown against the stored score.",
        "<b>Documentation and dependency checks.</b> The documentation check of the three deliverables and the check that the lock file matches the declared dependencies.",
    ])
    B.table("testing", "Software tests and their recorded outcomes on 28 October 2026 (engineering verification, not detector accuracy).",
            ["Level", "What was run", "Recorded outcome"],
            [["Safe suite", "all unit and contract tests, graphics processor hidden; the 13 data-dependent tests deselected", "710 passed, 2 skipped (the two opt-in tests)"],
             ["Opt-in, graphics processor (record of 27 October 2026)", "parity of the wrapped detectors on 11 non-sealed files, and the live face path", "5 passed; largest parity difference 0.0019 against 0.02"],
             ["Live dashboard, 8 demonstration inputs", "the page driven in a browser; each report checked against the schema and each number shown against the stored score", "all 8 behaved as specified, 0 page errors (on 27 October 2026: 6 media files with all seven modules reported, every report valid; 2 error cases failed with UNSUPPORTED_MEDIA_TYPE and UNREADABLE_FILE)"],
             ["Documentation check", "the check of the paper, report and slides, in the project environment", "71 pass, 0 fail (before the verdict and Evaluation Lab text was added)"],
             ["Lock file", "lock file against the declared dependencies", "consistent, exit 0"],
             ["Earlier records, not re-run on this state", "live checks of all modules; Content Credentials cases; generated-video reproducibility; full suite with the graphics processor visible; clean installation from the lock file",
              "60 checks, 0 failed; 5 of 5 cases; 16 of 16 frames identical; 508 passed, 2 skipped; 495 passed, 2 skipped"],
             ["Earlier records, 26 and 27 October 2026", "safe suite before the redesign, and after it", "497 passed, 2 skipped; 551 passed, 2 skipped"]],
            [3.6, 6.4, 6.4], ["left", "left", "left"], size=9, trace="TESTING_RECORD sections 2 to 4 and section 10 (710/2/13, 497/2/13, 551/2, 5, parity 0.001919, 8 inputs, 71/0, uv lock); sections 5 and 6 for the history (60/0, 5 of 5, 16 of 16, 508/2, 495/2)")
    B.p("One defect was found and fixed during the application run. In live mode, after a failed or refused analysis (an unsupported file type or an unreadable file), the stand-in "
        "scenario shown when the page loads, tagged MOCK, stayed on screen next to the failure message. The page now hides any report before a new analysis starts; a regression test "
        "covers it, and in the browser the two error cases now show the error and no report.")
    B.p("The limits of this evidence are stated with it. The tests ran on the development machine, which also holds the large files that version control does not track; a fresh "
        "clone without those files was not run end to end, and the data-dependent tests were not run. The application run and the browser tests examine structure and integration "
        "(every module appears, statuses and reasons are right, the page renders warnings and errors), not accuracy.")


# ----------------------------------------------------------------------------------------------------------------------
def after_freeze(B):
    B.h2("Experiments after the Freeze", trace="E-015, E-018 to E-030; D-070 to D-086")
    B.p("Four groups of experiments followed the sealed run: the frozen system on a further sealed set, FakeAVCeleb, with two follow-ups on its development half; audits of technical "
        "shortcuts in two generated-video corpora; three candidate generated-video detectors; and the validation of the lip-sync module. Each was registered with its hypothesis, "
        "success line and failure line before its run (the limits of that record are stated at the end of this section). None of them changed the frozen system, its thresholds or "
        "its calibration.")

    B.h3("The frozen system on the sealed half of FakeAVCeleb", trace="E-015; D-070")
    B.p("FakeAVCeleb <c fakeav> arrived after the freeze. Its identities were split into two halves inside each race and gender cell by a salted hash fixed before any split was computed, "
        "and a clip was kept only if its source and target identities fall on the same side; the development half and the sealed half each have 250 identities and share none. The "
        "sealed half (5,013 clips) was scored once by the frozen system under the final flag, with the predictions written before the labels were read. Each modality was judged by its "
        "own label, and two controls asked whether a module responds to the other modality. <ref tab:e015> gives the results.")
    B.table("e015", "The frozen system on the sealed half of FakeAVCeleb (one run; 95 percent group bootstrap interval over 250 source identities, 2,000 resamples).",
            ["Question", "Comparison", "AUC [95 percent interval]", "Pre-stated rule"],
            [["Visual transfer", "real video (500 clips) against fake video (4,513 clips)", "0.9552 [0.9442, 0.9657]", "supported (lower bound above 0.70)"],
             ["Audio transfer", "real audio (2,725 clips) against fake audio (2,288 clips)", "0.9263 [0.9149, 0.9366]", "supported"],
             ["Independence, visual", "visual score, same real video with real against fake audio", "0.5613 [0.5428, 0.5816]", "supported (inside 0.40 to 0.60)"],
             ["Independence, audio", "audio score, same real audio with real against fake video", "0.5011 [0.4801, 0.5243]", "supported"]],
            [3.0, 5.6, 3.8, 4.0], ["left", "left", "center", "left"], size=9, trace="E-015 RESULT")
    B.p("All four criteria were met, but three findings limit what this means. First, the frozen thresholds did not transfer: at them the visual sensitivity is 0.480 (0.231 for the "
        "Wav2Lip clips) at a specificity of 0.996, and the audio specificity is 0.637 at a sensitivity of 0.962, so the frozen decision rule would miss about half of the fake videos and "
        "flag about a third of the real audio. Second, the real audio is interview speech while the fake audio is clean synthesis, a channel difference that the experiment cannot "
        "separate from the manipulation. Third, the visual score separates two clips with identical real video slightly (0.5613), which pointed to a re-encoding effect. The prediction "
        "written before the run for the visual AUC (0.55 to 0.80) was wrong. Two follow-ups on the development half examined the third point; the sealed half was not opened again.")
    B.p("The first follow-up found that the gap between the two identical real videos is present on the development half, 0.5517 [0.5325, 0.5729], and disappears after one common "
        "re-encode, 0.5012 [0.4806, 0.5213], while nearly every pair differs in encoder tag, bit rate and frame count. It also found a large technical shortcut in the data set: the video "
        "codec alone separates real from fake video (a metadata model reaches 0.9985), because the real videos use the h264 codec and almost all fake videos the mpeg4 codec (the exceptions are face-swap clips), and "
        "cheap pixel statistics without any face reach 0.9585. On the same 1,000 development clips the frozen visual score falls from 0.9732 [0.9642, 0.9816] to 0.8772 [0.8562, 0.8981] "
        "after a common re-encode (paired change −0.0960 [−0.1127, −0.0803]), and on the Wav2Lip clips from 0.9548 to 0.7944, with the fake scores falling and the real ones unchanged.")
    B.p("The second follow-up gave the real clips an MPEG-4 generation at a bit rate that brackets that of the fakes before the common re-encode. The separation did not fall: 0.8856 "
        "[0.8643, 0.9056] against 0.8763 [0.8548, 0.8963] without the matching, a change of +0.0092 [+0.0006, +0.0180]; by the pre-stated rule the outcome is that manipulation evidence "
        "remains. The prediction (0.70 to 0.85) was wrong again. The reverse control, an extra MPEG-4 generation for the fakes, lowered the separation by −0.0430 [−0.0528, −0.0339], so the "
        "score of the fakes depends partly on their compression. This control cannot settle the question: in the development half of FakeAVCeleb the codec and the method coincide completely (every h264 "
        "fake video is a face swap, every FSGAN or Wav2Lip fake is mpeg4), so a difference between them cannot be assigned to codec rather than method or content. Whether the sealed AUC of "
        "0.9552 is inflated by a data set artefact, and by how much, is not estimated and not claimed.")

    B.h3("Shortcut audits of the generated-video corpora", trace="E-018, E-019 (v1, v2, v3); D-071")
    B.p("Before any content-based generated-video experiment, the metadata of DeepAction <c deepaction> (2,605 clips, 100 real stock clips against clips from six generators) was audited "
        "with models that see only container and stream headers. They separated real from generated clips perfectly (AUC 1.0000, also for every generator left out), through the encoder "
        "tag, the colour tags, the creation time, the resolution and the frame rate. Every clip of DeepAction and of DF26 <c df26> was therefore re-encoded by one recipe. The first "
        "recipe copied two colour tags from the source, which left a perfect shortcut; the second set them and failed the pre-stated acceptance rule by a small margin (an upper bound "
        "of 0.6048 against the line of 0.60, carried by the keyframe and B-frame structure); the third fixed that structure, made every header field constant and passed. The pass is "
        "true by construction and holds for the headers only: it is the third recipe tried on the same data, and seven cheap pixel statistics still separate the classes on DeepAction "
        "with an AUC of 0.773 [0.729, 0.817] by logistic regression and 0.787 [0.743, 0.827] by gradient boosting. The re-encoded clips are therefore not free of shortcuts, and every "
        "content-based result on them is reported next to that pixel-statistics baseline.")

    B.h3("Generated-video candidates and their held-out test", trace="E-020; D-071, D-072, D-074")
    B.p("Three candidates were trained on the re-encoded DeepAction clips with features of the frozen DINOv2 small encoder: a temporal-geometry classifier in the style of ReStraV "
        "<c restrav>, a logistic regression on the mean class-token embedding (an appearance baseline), and the pixel-statistics control, which is never deployed. The development metric "
        "was the mean over the six generators of the AUC with each generator left out of training, and the selection rule, fixed before the run, required a lower bound above 0.70 and a "
        "paired advantage over the control. The heads were frozen and hashed before the held-out benchmark DF26 (2,691 talking-head clips, 271 real and 2,420 generated by ten generators) "
        "was read; DF26 is licensed for evaluation only and was used for nothing else. <ref tab:e020> gives the results.")
    B.table("e020", "Generated-video candidates: development (DeepAction, generator left out) and held-out test (DF26, once). AUC [95 percent cluster bootstrap interval].",
            ["Candidate", "DeepAction, development", "DF26, all clips", "Outcome"],
            [["Temporal geometry, ReStraV-style classifier", "0.5073 [0.4608, 0.5542]", "0.5077 [0.4723, 0.5424]", "at chance on both"],
             ["Mean class-token embedding, logistic regression", "0.8281 [0.7908, 0.8662]", "0.4957 [0.4755, 0.5168]", "selected; did not transfer"],
             ["Seven pixel statistics (control)", "0.7613 [0.7182, 0.8033]", "0.5604 [0.5361, 0.5852]", "control, not deployed"]],
            [5.0, 3.8, 3.8, 3.8], ["left", "center", "center", "left"], size=9, trace="E-020 development RESULT and DF26 RESULT")
    B.p("Only the appearance baseline met the selection rule (its advantage over the control was 0.0668 [0.0147, 0.1209]). On the pre-registered primary question, the closed-generator "
        "subset of DF26, it reached 0.4880 [0.4534, 0.5204], so the hypothesis that it transfers is not supported; at its development threshold the sensitivity is 0.3165 and the "
        "specificity 0.6568. The likely reading, a hypothesis that was not tested, is that on DeepAction the head recognised stock footage against generated action content rather than "
        "generation itself. This is why the installed head is experimental and its finding is always inconclusive. The result does not show that the method cannot work; it shows that "
        "this head, trained on this corpus, does not generalise to talking-head video from other generators.")

    B.h3("Do real videos from other sources look generated to that head?", trace="E-026; D-078")
    B.p("The reading above was tested on 27 October 2026 in a pre-registered diagnostic that fits nothing. First, the development result of the appearance baseline was recomputed from "
        "the stored features with the original code and matched the record (0.8281). Then the installed head scored unmodified real videos from two other sources through the same "
        "preprocessing as the live module: all 140 original videos of the FaceForensics++ validation split and 300 real clips drawn at random from the official LAV-DF training set "
        "(neither was used to train the head; no sealed split and no DF26 clip was read). At the head's own threshold, 338 of the 440 real clips, 0.7682 [0.7265, 0.8052], were "
        "scored as generated-like, against 19 of 100, 0.19 [0.1251, 0.2778], for DeepAction's own real clips scored out of fold; FaceForensics++ originals 72 of 140, LAV-DF clips "
        "266 of 300. The pre-registered hypothesis of a source confound is therefore supported: on DeepAction the head separates that library's stock footage from everything else "
        "far more than it detects generation. Which property drives it (source, content, framing, resolution or compression) is not identified, because the other sources differ "
        "from the stock clips in all of them at once. The module now states this result in its limitations; its finding stays inconclusive.")

    B.h3("Training on the open generators of DF26", trace="E-027; D-079, D-085")
    B.p("The licence of DF26 allows detector training on its six open generators (HunyuanVideo 1.5, LTX 2.3 and Wan 2.2, each image-to-video and text-to-video) under a "
        "leave-one-generator-out protocol, and keeps its four commercial generators for evaluation only. A pre-registered experiment used this route on 27 October 2026. Real clips "
        "were grouped by source video and split into five folds so that no source appears on both sides; each of the three open families was held out in turn, because the two "
        "variants of one family share their backbone. Eight candidates were declared in advance (four frozen image encoders, each with the mean embedding of 8 frames or with "
        "temporal statistics over 14 frames), with the pixel statistics and the installed head as controls and three fold seeds. <ref tab:e027> gives the main results.")
    B.table("e027", "DF26 open-generator training and the one counted look at the commercial generators. AUC [95 percent cluster bootstrap interval over source groups].",
            ["Quantity", "Result"],
            [["Chosen head (CLIP ViT-L/14, mean of 8 frames, logistic regression), open families held out", "0.7453 [0.7174, 0.7710]"],
             ["Seven pixel statistics, same protocol (control)", "0.4921 [0.4682, 0.5119]"],
             ["Installed head of the earlier experiment, not refitted (control)", "0.4995 [0.4819, 0.5168]"],
             ["Temporal features minus pooled appearance (14 frames)", "-0.0404 [-0.0650, -0.0165]"],
             ["Commercial generators, 794 clips against 200 scenario-matched reals (one look)", "0.7548 [0.7189, 0.7885]"],
             ["Weakest and strongest commercial generator", "Kling 3.0 0.6483; Wan 2.6 0.8758"],
             ["Real FF++ and LAV-DF clips flagged at the balanced threshold", "0.8023 [0.7625, 0.8368]"]],
            [10.0, 6.4], ["left", "center"], size=9, trace="E-027 RESULT; data/_results/modern/e027")
    B.p("In-domain training changed the held-out picture: the commercial generators, on which the earlier head was at chance (0.4880), were separated with AUC 0.7548, and the "
        "pre-registered success line was met. Two results limit the reading. Temporal features did not add to the averaged appearance, so the head compares how frames look, not how "
        "they move. And the source confound of the earlier head remains: at its balanced threshold the deployable head flags 80 percent of unmodified real clips from FaceForensics++ "
        "and LAV-DF, against 28 percent of DF26's own real clips, and inside DF26 it flags YouTube-sourced real clips far more often than the TalkingCelebs clips. The real side of the "
        "commercial look is the same 271 real clips seen in development, and this was the second look at that subset. The head is therefore not installed; a head that is "
        "to make findings needs real training clips from many sources and must pass the same source-confound control.")

    B.h3("The generated-image model on modern generators", trace="E-028; D-080, D-081, D-083")
    B.p("The generated-image model of the product is a linear probe on frozen CLIP ViT-L/14 features, trained on ProGAN images. Before the experiment below the project had no "
        "measurement of it on contemporary generators or on portraits, so on 27 October 2026 the frozen model was scored, without any change, on a test-only subset of "
        "Community Forensics-Eval <c commforensics>: 11,550 images (5,759 generated by 21 generators, 5,791 real), chosen by a written rule that prefers the "
        "cheapest shards and that was fixed before any image was seen. The generators are commercial services (DALL-E 2 and 3, Firefly, FLUX, Ideogram, Imagen 3, "
        "Midjourney), latent and pixel diffusion models and two GANs; eight of them share one pooled set of 121 RAISE photographs as real images. The experiment was "
        "registered with its hypothesis and with the lines for adequate and inadequate before the first image was decoded; the ordering record is a hash file written by the "
        "script (no commit was authorised), which is weaker evidence than a commit. <ref tab:e028> gives the results.")
    B.table("e028", "The frozen generated-image model on 11,550 images of 21 modern generators (AUC and rates with 95 percent bootstrap intervals over images, 2,000 resamples).",
            ["Quantity", "Result"],
            [["Macro AUC over the 21 generators", "0.7039 [0.6855, 0.7229]"],
             ["Pooled AUC / generated images missed at the frozen threshold / real images flagged", "0.7973 / 0.7395 [0.7300, 0.7500] / 0.0418"],
             ["AUC, two GANs", "0.9580 [0.9475, 0.9674]"],
             ["AUC, pixel diffusion (Hourglass)", "0.8410 [0.8221, 0.8595]"],
             ["AUC, commercial generators (share missed 0.8839)", "0.7351 [0.7239, 0.7467]"],
             ["AUC, latent diffusion against RAISE reals", "0.5808 [0.5337, 0.6322]"],
             ["AUC of a format-only score (1 for PNG), pooled", "0.9222"],
             ["Model AUC on PNG real against PNG generated images", "0.6123 [0.5908, 0.6353]"],
             ["Change of pooled AUC after JPEG re-encoding of 380 generated PNG images", "-0.059 [-0.067, -0.051]"],
             ["Images with a face (3,119): whole-image model / face-swap path of the product", "0.7859 [0.7717, 0.7995] / 0.6020 [0.5836, 0.6202]"],
             ["Paired difference, whole-image model minus face-swap path", "+0.1839 [+0.1596, +0.2074]"]],
            [10.0, 6.4], ["left", "center"], size=9, trace="E-028 RESULT; data/_results/modern/e028")
    B.p("By the line written in advance the model is inadequate as a check for modern generators: the upper bound of its macro AUC, 0.7229, is below 0.75, and it misses about three "
        "quarters of the generated images at its threshold, 71 to 95 percent for every commercial generator, because the threshold was chosen on ProGAN validation images. It "
        "ranks GAN images well, is at chance on one latent diffusion model (0.4951) and below chance on another (0.4145, both against the RAISE photographs). Part of what it "
        "separates is the file format: 92.8 percent of the generated images are PNG and 88.1 percent of the real ones are JPEG, a score that only reads the format reaches 0.9222, "
        "the model's AUC on PNG against PNG falls to 0.6123, and recompressing generated PNG files as JPEG lowers its AUC. The observed macro AUC of 0.7039 was just above the "
        "range written beforehand (0.50 to 0.70). For still images with a face, which the product sends to the face-swap model only, the whole-image model separates the "
        "generated faces better (paired difference +0.1839, interval above zero), but it still misses 80.6 percent of them, so neither is usable on modern generated faces. "
        "This is the evidence behind the experimental whole-image check: images with a face also receive a whole-image score, shown as experimental, without a finding. Limits: a subset chosen by "
        "cost, one pooled real set for eight generators, reals and generated images that differ in format and size, and no image that was edited or shared through a social "
        "network.")

    B.h3("Retraining the generated-image head on recent generators", trace="E-029; D-080, D-081")
    B.p("To ask whether the weakness of the generated-image model is its training data, the same kind of head (frozen CLIP ViT-L/14 features and one linear layer) was retrained on 27 October "
        "2026 on images of many recent generators, using the training part of Community Forensics-Small: 23,807 training images (14,000 generated by 125 generators, 9,807 real) and "
        "5,333 validation images from 31 other generators and other real images, so that no generator and no real image is on both sides. Three seeds were trained for each of two "
        "encoders; the temperature, the threshold and the choice of encoder used the validation images only, and the six heads were frozen and hashed before the evaluation images were scored. "
        "The evaluation was the same 11,550 images as above, scored once. The registration was written before any feature was extracted; an amendment, also written before any "
        "feature existed, stopped the per-source cap from discarding real images. <ref tab:e029> gives the results.")
    B.table("e029", "The head retrained on 125 recent generators, scored once on the 11,550 evaluation images (seed mean over three seeds; 95 percent bootstrap intervals over images).",
            ["Quantity", "Result"],
            [["Validation AUC on held-out generators (chosen encoder)", "0.9823 [0.9662, 0.9969]"],
             ["Macro AUC over the 21 evaluation generators", "0.8839 [0.8732, 0.8942] (seeds 0.8863, 0.8832, 0.8823)"],
             ["Paired gain over the installed head (macro AUC)", "+0.1800 [+0.1643, +0.1959]; each seed above zero"],
             ["Paired gain, generators without a related model in training (18)", "+0.1675 [+0.1540, +0.1803]"],
             ["Paired gain by type: latent diffusion / commercial / pixel diffusion", "+0.32 / +0.12 / -0.01 [-0.03, +0.01]"],
             ["AUC on PNG real against PNG generated images only (installed head 0.6123)", "0.8365"],
             ["Change of pooled AUC after JPEG re-encoding of 380 generated PNG images", "-0.0729 [-0.0827, -0.0643]"],
             ["Real images flagged at the head's validation threshold (frozen candidate)", "0.5213 [0.5091, 0.5339]"],
             ["Real images flagged: LAION photographs / COCO photographs", "0.5885 / 0.0328"],
             ["Generated images missed at that threshold", "0.0545 [0.0495, 0.0596]"]],
            [10.0, 6.4], ["left", "center"], size=9, trace="E-029 RESULT; data/_results/modern/e029")
    B.p("The hypothesis of improvement is supported: on 21 generators that are not in its training data, the retrained head ranks generated against real images clearly better than the "
        "installed one, in all three seeds, most on latent-diffusion generators, and the gain is still present when only generators without a related model in training are counted "
        "and when real and generated images are restricted to one file format. It is not a usable detector as it stands. The threshold was chosen on validation real images from two "
        "sources (COCO and FFHQ), and on real photographs from other sources it flags about half of them (more than half of the LAION photographs, which are 82 percent of the real "
        "evaluation images). The preregistered line for a candidate product check required a false-positive rate of at most 0.10, and the JPEG control lowered the pixel-diffusion AUC "
        "by more than 0.10, so the heads stay research artefacts and nothing is installed. The encoder that validation ranked lower reached a higher macro AUC on the evaluation images "
        "(0.9142); the rule fixed in advance was applied as written and was not changed after the result. Three written predictions failed (the smallest gain was not on the commercial "
        "generators, JPEG-compressed fakes were not missed, and the two encoders were not equal on validation). Limits: a subset of the training set, validation real images from two "
        "sources, and no evaluation of edited images or of images shared through social networks.")

    B.h3("Comparison with published results", trace="docs/PERFORMANCE_BENCHMARK.md; E-030; D-084")
    B.p("A published number is set against ours only when task, data, protocol and metric match; otherwise it is reported as context and no difference is computed. The full table "
        "with 32 verified papers is kept in the repository. Most rows are partially comparable, because our test subsets differ from the official ones or because the metric differs "
        "(image papers report average precision, speech papers report equal error rate). To match the speech metric, the equal error rate was computed afterwards from the stored "
        "sealed scores of the frozen speech system, with no model run and nothing selected. On the official ASVspoof 2019 LA evaluation set, with the official training "
        "data, it is 6.94 percent [5.95, 7.82] against 1.13 percent for AASIST <c aasist>, 5.81 points worse: a system of three frozen general speech encoders built to transfer "
        "pays for that in-corpus. On the 42 sealed In-the-Wild speakers it is 15.8 percent [12.7, 18.8]; published values on the full set, 33.9 percent for the best model of "
        "<c inthewild> and 7.2 percent for a frozen encoder of 2 billion parameters <c pascu>, use more speakers, so no gap is computed. On DF26 without DF26 training, our earlier "
        "head (0.4957) lies within the range the DF26 authors report for detectors trained on FaceForensics++ (pooled AUROC 44.0 to 69.7 percent, without intervals) <c df26>.")
    B.p("Each row of the full table carries one of four comparability classes. Class A is directly comparable: same data, test items, metric, aggregation and declared training "
        "protocol, and only these rows receive a difference. Class B is the same dataset under a different protocol, class C is cross-dataset or out-of-distribution, and class D is "
        "contextual only. The class A rows are the speech rows above, with a third: RawNet2 under the protocol of Pascu et al. reaches 5.9 percent, so our point estimate is slightly higher "
        "but the sign of the difference is not established; the zero-shot generated-video row on DF26 (our head 0.4957, at chance, against 44.0 to 69.7 for nine published detectors "
        "trained on FaceForensics++); and our own re-run Xception baseline against the final video system on the same sealed tests (final system 0.8218 on the Celeb-DF family). The "
        "In-the-Wild speech row is class B. No class A row exists for the face-video task against any published detector, because published results use other subsets, frame counts and "
        "training compression and give no intervals; the DFDC dataset was never used.")
    B.p("The experimental heads remain unapproved for a stated reason: each failed its own acceptance line, on real clips from other sources (a source confound) or on false positives. "
        "The installed generated-image model has a macro AUC of 0.7039 on 21 modern generators and misses 0.7395 of generated images, which is inadequate. The retrained image heads gain "
        "0.1800 in ranking (macro AUC 0.8839) but flag 0.5213 of real evaluation images at the validation threshold, so they are not installed. The generated-video head trained on the "
        "DF26 open generators reaches 0.7548 on unseen closed generators but flags 0.8023 of unmodified real clips from two other sources. The approval table stays empty, and promotion "
        "would need a new preregistered test with a false-positive line on sources not used for calibration.")

    B.h3("Validation of the lip-sync module", trace="E-022, E-024, E-025; D-073")
    B.p("The lip-sync module was checked in three steps on development clips only. The first step ran constructed cases on 40 LAV-DF training clips (shifted audio, non-speech audio, "
        "unsuitable inputs) and exposed two problems, spurious offsets for shifts beyond the search range and confident results on non-speech audio; the confidence minimum of 3.0 and "
        "the speech heuristic were chosen on those same clips, so that step is exploratory. The second step froze the module and measured it on fresh clips with lines fixed in advance: "
        "shifts of 120 to 480 ms were recovered within 2 frames or flagged in 97.8 percent [94.4, 99.1] of the files, every shift of 1000 ms and every synthetic non-speech track was "
        "inconclusive, and every unsuitable input was not applicable or inconclusive. It failed on real speech: with the placeholder tolerance of 2 frames, 15.0 percent [8.1, 26.1] of 60 "
        "LAV-DF clips and 13.3 percent [6.9, 24.2] of 60 FakeAVCeleb clips were flagged, against a line of 10 percent. Wav2Lip clips were flagged in 85 to 93 percent, almost all at an "
        "offset of exactly −3 frames, against the written prediction.")
    B.p("The third step fitted a band to the natural spread of real clips and validated it on disjoint clips. On 230 calibration clips (150 LAV-DF and 80 FakeAVCeleb) the rule fixed in "
        "advance gave the band of −6 to +2 frames, which holds 98.2 percent of the confident offsets. On 230 other real clips the band flagged 3, that is 1.3 percent [0.4, 3.8], against "
        "27 of 230 (11.7 percent) under the old rule, which passed the line. The cost is measured as well: the Wav2Lip fakes, which sit at −3 or −2 frames, were flagged in 0 of 160 cases, "
        "and real audio shifts of 120 ms were flagged in 14 of 28 clips in one direction and 2 of 28 in the other. The earlier separation of the Wav2Lip clips was therefore carried by a "
        "one-frame margin that also covers about one real clip in ten. The module keeps the band, reports the offset and confidence as observations, and remains weak evidence, not a "
        "detector of lip-sync fakes.")

    B.h3("Limits of the later evidence and of its record", trace="docs/RESEARCH_INTEGRITY.md sections 1, 3, 4, 6; docs/SETUP.md section 7; E-025 post-audit clarification")
    B.bullets([
        "Order of registration. The registration texts of the generated-video and lip-sync experiments and of the two FakeAVCeleb follow-ups first appear in version control in one "
        "commit, together with the scripts that ran them. Each text states that it was written before its run, and file times are consistent with that, but version control cannot show "
        "it. This is a gap in the record, not evidence that the order was different; the record is not rewritten to close it, and every later experiment is to commit its registration "
        "on its own before the run. The diagnostic of 27 October 2026 could not follow that rule, because no commit was authorised before it ran; its script wrote the hash of "
        "its registration text and the time to a file before the first clip was read, which is weaker evidence than a commit.",
        "The re-encoding recipe that passed the header audit is the third one tried on the same data, and the pass holds for the headers only; pixel statistics still separate the "
        "classes at about 0.77 to 0.79, so the re-encoded clips are not shortcut-free.",
        "Inside FakeAVCeleb the codec and the manipulation method coincide completely, so no analysis of its development half can say whether the residual separation is manipulation "
        "evidence or another artefact; the sealed AUC is read as ranking quality on a data set with a codec shortcut.",
        "The LAV-DF half of the lip-sync calibration and validation pools was split by file, because its metadata has no speaker field; the same speaker can occur in both, so the 1.3 "
        "percent is not evidence of speaker-independent behaviour, and both sources derive from VoxCeleb2.",
        "Licences: the SyncNet weights have no stated licence, and the terms of the FakeAVCeleb request form are not recorded in the project; neither blocks internal academic use, both "
        "block any redistribution of weights or derived data. DF26 is used for evaluation only, as its licence requires.",
        "The frozen checkpoints and most result files are not tracked by version control; for them, only the hashes recorded in their own metadata and in the ledger exist, not an "
        "independent hash list made before they were used.",
        "The speech module has no speech check of its own: a pure tone receives a speech finding (score 0.0), so its finding on non-speech audio means nothing.",
        "The test counts above are engineering records of the code on the development machine; they are not measurements of detection accuracy.",
    ])
