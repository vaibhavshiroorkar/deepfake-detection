# ITERATIONS

Status: DECIDED (user), 2026-10-24 (D-040, D-041), with the user's six changes applied, written 2026-10-24 before any model score exists. Nothing in this file is a result, and no outcome is pre-written: the "prediction" lines are hypotheses with a sign and a rough size, and each one is judged against its failure line after the run. Levers of I-1 to I-6 are candidates; the lever actually used in a slot is chosen after the diagnostic battery (EXPERIMENTS.md section "Diagnostic battery") of the previous iteration and logged as its own decision before the run. There are seven experimental iterations after the MVP, I-1 to I-7 (I-7 is rPPG), plus I-8, integration and freeze, which runs no experiments. I-7 and I-8 are fixed.

Derivations for every slot are in MATH_NOTES.md sections 10 to 17. Notation: $\ell$ is a logit, $p=\sigma(\ell)$, $y\in\{0,1\}$ with 1 = fake.

## How a slot is run (the same template every time)

Observation (numbers from the battery) -> Diagnosis (bias, variance, shift, calibration, aggregation, missing signal) -> Mathematical hypothesis -> Change -> Prediction (sign and size) -> Result -> Verdict (supported or not). Comparisons follow CLAUDE.md rule 4: same data, split, budget and metric, 3 seeds, paired group bootstrap, cheaper candidate wins when intervals overlap. Tuning uses train and validation (and the dev_x slice for ranking only, D-035); the test splits are touched once, at the freeze (I-8).

## Evidence status: exploratory and confirmatory (D-041, PROTOCOL.md addendum)

Every verdict below that rests on dev_x or on validation is exploratory: it reports sign, size and interval and chooses the next step. At the freeze each frozen iteration is evaluated once on the sealed test sets (a staircase of one test score per iteration). The confirmatory claims are two pre-declared paired comparisons on sealed data: the final system versus I-0, and the final system versus the re-run baseline of D-025. Nothing else is called confirmed.

## Schedule (one schedule with PLAN.md; the freeze is Thu 2026-10-29)

| Iteration | Target day | Needs |
|---|---|---|
| I-0 MVP | Mon 26 Oct (provisional 18:00, STATUS.md) | E-001 to E-003 features, heads, web page on a real checkpoint |
| I-1 (video task first), I-2, I-3 | Tue 27 Oct | I-0 diagnostics, measured step times (D-042), GPU time |
| I-4, I-5, I-6 | Wed 28 Oct | LAV-DF extracted (for I-6), diagnostics of the previous slot |
| I-7 rPPG | Wed 28 Oct decision on dev_x; S-1 runs earlier on CPU, in parallel with I-1 to I-3 | contiguous-frame extraction, see I-7 |
| I-8 integration and freeze | Thu 29 Oct (no experiments; the one-time test evaluations and the two confirmatory comparisons) | all earlier work frozen |

D-036 item 3 (fine-tune the top three encoders per task at three learning rates) is replaced by D-042, which sets the I-1 budget from measured step times: the video task first, extra seeds only for its winner, a per-run time cap, and the encoders that do not fit are named there.

## I-0 MVP (not an improvement; the baseline everything is compared with)

Frozen encoders chosen by the sweep (D-036), a linear head, temperature scaling (D-024), evaluated on the in-dataset validation split and dev_x with 3 seeds and intervals. Definition of "working MVP": D-040. The diagnostic battery is run on it and fixes the order of what follows.

## I-1 to I-6: diagnosis-chosen slots

For each slot: the primary candidate lever, the diagnostic that triggers it, the alternative if the diagnostic says otherwise, the equation, the predicted effect, the failure line. Primary candidates are only a starting guess for planning time; they are replaced without penalty when the battery points elsewhere.

### I-1 (primary candidate: fine-tuning depth)

| Field | Content |
|---|---|
| Lever | Linear probe -> train the last $k$ blocks, or low-rank adapters, or full fine-tuning. Budget and scope: D-042 (video task first: 3 encoders x 3 learning rates x 1 seed under a per-run time cap, then extra seeds for the winner only; T-I and T-A stay frozen in I-1 unless time remains) |
| Triggered when | Bias: training AUC and validation AUC are both low or close (gap below 0.02 and AUC under about 0.90), so frozen features lack the signal (E-001 criterion f2 territory) |
| Instead if | Variance or shift (validation AUC high, dev_x much lower): do not add capacity; go to the augmentation and regularisation slot (I-4 lever) first. If calibration alone is bad: calibration slot |
| Equation | $W' = W + \tfrac{\alpha}{r} B A$, $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$; trainable parameters $r(d+k)$ instead of $dk$ (MATH_NOTES 11) |
| Predicted effect | In-dataset validation AUC up (about +0.02 to +0.05). dev_x: sign uncertain, between slightly negative and +0.03, because deeper training can fit source-specific cues (R-044) |
| Failure | The paired dev_x interval of (fine-tuned minus linear probe) includes zero or is below zero while validation AUC rises: capacity went into source-specific cues. Also failure if peak VRAM or time per epoch is outside the measured 12 GB budget |

### I-2 (primary candidate: frame-to-video aggregation)

| Field | Content |
|---|---|
| Lever | Mean of frame scores (D-024 default) -> top-$k$ mean, max, or learned attention pooling over frame features |
| Triggered when | Aggregation: frame-level AUC is clearly above video-level AUC, or fake videos show a wide spread of frame scores (a few frames carry the artefact), or the mask shows many no-face frames |
| Instead if | Frame and video AUC agree: aggregation is not the bottleneck; take the loss or class-weighting slot (I-3 lever) |
| Equation | $\hat s_{\text{mean}}=\tfrac1T\sum_t s_t$ vs $\hat s_{\text{top-}k}=\tfrac1k\sum_{t\in \text{top }k} s_t$; $\operatorname{Var}(\hat s_{\text{mean}})=\tfrac{\sigma^2}{T}\left(1+(T-1)\rho\right)$ (MATH_NOTES 12) |
| Predicted effect | Video AUC change from 0 to +0.02 when few frames are informative; negative when frame noise dominates and max inflates real scores |
| Failure | The paired interval of (new minus mean) includes zero, or the real-video score histogram shifts right (more false alarms) |

### I-3 (primary candidate: loss and class weighting)

| Field | Content |
|---|---|
| Lever | BCE -> class-weighted BCE, focal loss, label smoothing, or a pairwise AUC surrogate |
| Triggered when | Imbalance or saturation: balanced accuracy at the validation threshold is far below what the AUC implies, score histograms of one class pile up at 0 or 1, or ECE is high and temperature scaling does not repair it. FF++ is 80 percent fake |
| Instead if | Scores are well spread and calibrated: loss is not the bottleneck; take regularisation (I-4 lever) |
| Equation | focal: $\partial L/\partial \ell=-(1-p)^{\gamma}\,(1-p-\gamma p\ln p)$ for $y=1$; pairwise: $L=\tfrac{1}{n_1n_0}\sum_{i:y_i=1}\sum_{j:y_j=0}\ln\left(1+e^{-(s_i-s_j)}\right)$ (MATH_NOTES 13) |
| Predicted effect | AUC change within about $\pm0.01$ (the ranking is mostly fixed by the features); balanced accuracy and ECE change more (up to a few points) |
| Failure | AUC interval includes zero and balanced accuracy and ECE improve by less than their intervals: the extra loss is not worth its tuning cost |

### I-4 (primary candidate: augmentation matched to test perturbations, and regularisation)

| Field | Content |
|---|---|
| Lever | Train with compression, blur and resize perturbations; mixup; weight decay and dropout |
| Triggered when | Shift: validation AUC high, dev_x or the compression/face-size splits much lower (gap above about 0.10), worse at stronger compression |
| Instead if | No shift gap: skip augmentation; combine encoders (I-5 lever) |
| Equation | $\min_\theta\ \mathbb{E}_{(x,y)}\mathbb{E}_{t\sim\mathcal T}\,L(f_\theta(t(x)),y)$; mixup $\tilde x=\lambda x_i+(1-\lambda)x_j$, $\tilde y=\lambda y_i+(1-\lambda)y_j$; ridge shrinkage (MATH_NOTES 14) |
| Predicted effect | dev_x AUC up (about +0.01 to +0.04) when the shift is a compression or resolution shift; in-dataset AUC flat or down by up to 0.01 |
| Failure | dev_x paired interval includes zero, or gains appear only on the perturbation types we trained on (no transfer to the others) |

### I-5 (primary candidate: combining several encoders)

| Field | Content |
|---|---|
| Lever | Concatenate features of two or three encoders, or average their scores |
| Triggered when | Ties and complementary errors: the sweep left several candidates tied (D-036), and their worst-error sets differ (low error correlation) |
| Instead if | Errors are shared (same videos wrong): combining will not help; stay with the cheapest tied encoder |
| Equation | $\operatorname{Var}\left(\tfrac1m\sum_j s_j\right)=\sigma^2\left(\bar\rho+\tfrac{1-\bar\rho}{m}\right)$; concatenation adds $\sum_j d_j$ head parameters (MATH_NOTES 15) |
| Predicted effect | AUC +0.01 to +0.03 if $\bar\rho<0.8$; no gain near $\bar\rho\to1$; score averaging at least as good as concatenation when training data are small |
| Failure | Interval includes zero: the extra encoders' cost (time, VRAM, licences) is not justified |

### I-6 (primary candidate: late fusion across modalities and calibration of each)

| Field | Content |
|---|---|
| Lever | Per-modality temperature scaling, then fuse audio and visual scores: sum of log-odds vs logistic regression on the two log-odds |
| Triggered when | Both unimodal heads are above chance on LAV-DF and calibrated (ECE small), and their errors are not strongly correlated. Needs LAV-DF extracted and its labels per modality |
| Instead if | One modality is near chance (the audio-visual risk and fallback claim of 764519a): report single-modality results and use the fallback claim; or errors are strongly correlated: learn weights with logistic regression, not a plain sum |
| Equation | under conditional independence, $\ell_{\text{fused}}=\ell_a+\ell_v-\ln\tfrac{\pi}{1-\pi}$; logistic regression $\sigma(w_0+w_a\ell_a+w_v\ell_v)$ (MATH_NOTES 16) |
| Predicted effect | Fused AUC at or above the better unimodal AUC; gain of 0 to +0.05 when the modalities carry independent evidence; the learned weights fall below 1 when they are dependent |
| Failure | The fused lower interval bound is below the better single modality's, or the gain is within the interval |
| Data that trains the heads | The unimodal heads are the ones frozen in I-0 to I-5: visual head trained on FF++ train, audio head on ASVspoof 2019 LA train, each with temperature scaling fitted on its own validation set. Only the fusion layer (the sum of log-odds has no parameter; the logistic regression has 3) is fitted, on LAV-DF train; its choice is made on LAV-DF validation; the identities of the three LAV-DF splits are disjoint. Because the heads come from other domains, per-modality temperature is also refitted on LAV-DF validation and reported both ways. Decided autonomously - needs user review for the choice of FF++ and ASVspoof heads (the alternative is heads retrained on LAV-DF) |
| Fused label | Fake if either modality is fake (video manipulated or audio manipulated), real only if both are real. The per-modality labels (visual-fake, audio-fake) from the LAV-DF metadata are reported too, one AUC per modality label. The metadata field names are to be verified after extraction (NR now) |

## I-7 rPPG under O-010 (fixed slot)

Success and failure thresholds are written here, before any run, and frozen by the approval of this file; they are not edited after the first run. The pulse method is a projection-based one (MATH_NOTES 17); the literature reference for it is NR until it is read and marked READ in RESEARCH.md.

| Field | Content |
|---|---|
| Parallel signal-sanity check S-1 (starts early, CPU only, during I-1 to I-3) | Contiguous-frame extraction (about 10 s, native frame rate; the 16-frame sampling of D-034 cannot carry a pulse). On real clips of the datasets we hold, compute the spectral peak in 0.7 to 3 Hz and its SNR. Negative control: the same clips with time-shuffled frames; the control is pooled across clips, with several independent shuffles per clip (at least 10), and the 95th percentile is taken over the pooled shuffled SNR values, written here before any run |
| S-1 passes when | At least 60 percent of real clips have a peak SNR above the 95th percentile of the shuffled control, and, if a dataset with pulse ground truth is obtained (UBFC-rPPG, currently DEFER in D-021), the mean absolute error of heart rate is at most 8 bpm. If neither holds, the pulse is not recoverable from our data |
| If S-1 fails | I-7 reports exactly that: "rPPG signal not recoverable from these data (numbers)", with the negative control, and no separation or fusion experiment is run. This is an honest result of the slot |
| (b) Separation, only if S-1 passes | rPPG-feature AUC on an identity-disjoint split, for at least two generator families; success if the lower bound of the interval is above 0.55; failure if the interval includes 0.5 |
| (c) Added value | Fused model with rPPG minus fused model without it, paired group bootstrap, 3 seeds, computed on dev_x. The adopt or do-not-adopt decision is made on dev_x (never on test) and adopts rPPG only if the whole interval is above zero (O-010). Test sets are touched only at the freeze. Failure: interval includes zero or is below |
| (d) Negative controls | Time-shuffled and identity-mismatched signals must give AUC whose interval includes 0.5; if not, the "separation" is an artefact |
| Prediction | Sign uncertain by design. Most of our videos are compressed, which attenuates the pulse; the prior expectation is that S-1 passes only partially. This is a hypothesis, not a result |

## I-8 integration and freeze (fixed slot, no new experiments)

Freeze the code, the preprocessing and the checkpoints; tag `iter-N` for every frozen iteration; touch each final test split once under `--final`; leakage check on the final splits; fill the paper, report and slides only from logged E-xxx entries; end-to-end demo with `inference_source` set to real; push. Delete the LAV-DF tar by hand (D-039). Anything not frozen by Thu 2026-10-29 is reported as not done.

## Model comparison table (defined once, filled from the logs)

One row per candidate encoder, one table per task (T-V, T-I, T-A). Columns are fixed in EXPERIMENTS.md section "Model comparison table"; the table is generated from the feature metadata, the weights log (data/_logs/encoder_weights.json) and the score files, never typed by hand. Weights tried and dropped (for example DINOv3, excluded by D-038) stay in the table with the reason.
