"""Chapter 3: mathematical background (definitions and short derivations only; nothing beyond the project notes)."""


def build(B):
    B.h1("Mathematical Background")
    B.box("This chapter presents the definitions and the short derivations that the later chapters rely on: error rates, ranking and calibration metrics, the group bootstrap "
          "and the paired tests, and the argument behind each lever of the iteration ladder, with its assumption and the conditions under which it does not help.")
    # ------------------------------------------------------------------
    B.h2("Notation and Setting", trace="MATH_NOTES section 0; DECISIONS D-024, D-012")
    B.p("Each example <i>i</i> has a label $y_i\\in\\{0,1\\}$, where 1 means fake and is the positive class and 0 means real. A model gives a score $s_i\\in\\mathbb{R}$; when the score is "
        "the probability of the fake class it is written $\\hat p_i\\in[0,1]$. The numbers of fake and real examples are $n_1$ and $n_0$, their sum is $n$, and the fake prevalence is")
    B.equation("prev", r"\pi=\frac{n_1}{n},\qquad n=n_0+n_1")
    B.p("A decision rule with threshold $t$ predicts fake when $s_i\\ge t$. At a threshold, $\\mathrm{TP}$, $\\mathrm{FP}$, $\\mathrm{TN}$ and $\\mathrm{FN}$ are the counts of true positives, false positives, true negatives and "
        "false negatives. Population statements use $S_1$ for the score of a randomly drawn fake and $S_0$ for the score of a randomly drawn real example. The unit that is resampled "
        "in the uncertainty estimates is the group, which is the source video together with the person identity (Chapter 4).")
    # ------------------------------------------------------------------
    B.h2("Error Rates, the ROC Curve and the Area under It", trace="MATH_NOTES section 1 and 2")
    B.p("The true positive rate and the false positive rate at threshold $t$ are conditioned on the true class, so they do not depend on the prevalence $\\pi$:")
    B.equation("rates", r"\mathrm{TPR}(t)=\frac{TP}{TP+FN}=P(S\ge t\mid Y=1),\qquad \mathrm{FPR}(t)=\frac{FP}{FP+TN}=P(S\ge t\mid Y=0)")
    B.p("The false negative rate is $1-\\mathrm{TPR}$ and the true negative rate (specificity) is $1-\\mathrm{FPR}$. The ROC curve is the set of points $(\\mathrm{FPR}(t),\\mathrm{TPR}(t))$ traced as "
        "$t$ moves from $+\\infty$, which gives the point $(0,0)$, to $-\\infty$, which gives $(1,1)$. The area under it is")
    B.equation("auc_int", r"\mathrm{AUC}=\int_0^1 \mathrm{TPR}\,d(\mathrm{FPR})")
    B.p("The ranking interpretation follows from a change of variable. Let $F_0(t)=P(S_0\\le t)$ for continuous score distributions. Then $\\mathrm{FPR}(t)=1-F_0(t)$ and "
        "$\\mathrm{TPR}(t)=P(S_1\\ge t)$, and substituting $u=1-F_0(t)$ gives $\\mathrm{AUC}=\\int P(S_1\\ge t)\\,dF_0(t)=P(S_1>S_0)$. When ties are allowed the general form is")
    B.equation("auc_rank", r"\mathrm{AUC}=P(S_1>S_0)+\tfrac{1}{2}\,P(S_1=S_0)")
    B.p("so the AUC is the probability that a random fake receives a higher score than a random real example. The empirical estimate is the Mann-Whitney form, where $U$ is the "
        "Mann-Whitney statistic:")
    B.equation("auc_emp", r"\widehat{\mathrm{AUC}}=\frac{1}{n_1 n_0}\sum_{i:\,y_i=1}\ \sum_{j:\,y_j=0}\left[\mathbf{1}(s_i>s_j)+\tfrac{1}{2}\mathbf{1}(s_i=s_j)\right]=\frac{U}{n_1 n_0}")
    B.p("The AUC lies in $[0,1]$; $0.5$ means ranking no better than chance and $1$ means that every fake outranks every real example. It is unchanged by any strictly increasing "
        "transformation of the scores, so it does not see calibration or the threshold, and its population value does not depend on the prevalence. These properties are why it is the "
        "headline metric of the project: the test sets are unbalanced (the pooled Celeb-DF family test set is 90.3 percent fake, 9.7 percent real), and a metric that depended on the class ratio would change with the "
        "test set and not with the detector.")
    # ------------------------------------------------------------------
    B.h2("Equal Error Rate", trace="MATH_NOTES section 3")
    B.p("The equal error rate is the common value of the two error rates at the threshold $t^{*}$ where they are equal:")
    B.equation("eer", r"\mathrm{FPR}(t^{*})=\mathrm{FNR}(t^{*}),\qquad \mathrm{EER}=\mathrm{FPR}(t^{*})")
    B.p("On a finite sample the two rates are step functions and may not cross exactly. A common estimate takes the threshold that minimises the gap and averages the two rates there:")
    B.equation("eer_hat", r"\hat t=\arg\min_{t}\left|\mathrm{FPR}(t)-\mathrm{FNR}(t)\right|,\qquad \widehat{\mathrm{EER}}=\tfrac{1}{2}\left(\mathrm{FPR}(\hat t)+\mathrm{FNR}(\hat t)\right)")
    B.p("The threshold $t^{*}$ is a property of the data on which it is computed; a threshold taken from one data set and applied to another no longer gives equal errors. Like the AUC, the EER "
        "uses only the ranking of the scores. It is reported for standalone speech, where it is the usual metric of the anti-spoofing literature and allows comparison with published tables "
        "<c asvspoof, aasist>.")
    # ------------------------------------------------------------------
    B.h2("Precision, Accuracy and Balanced Accuracy", trace="MATH_NOTES section 4 and 5")
    B.p("Precision depends on the prevalence. By Bayes\u2019 rule")
    B.equation("prec", r"\mathrm{Precision}(t)=\frac{TP}{TP+FP}=\frac{\pi\,\mathrm{TPR}(t)}{\pi\,\mathrm{TPR}(t)+(1-\pi)\,\mathrm{FPR}(t)}")
    B.p("so the same detector has a different precision on a test set with a different real-to-fake ratio, although its true and false positive rates stay the same. Accuracy and balanced "
        "accuracy are")
    B.equation("bacc", r"\mathrm{Acc}(t)=\pi\,\mathrm{TPR}(t)+(1-\pi)\,\mathrm{TNR}(t),\qquad \mathrm{BA}(t)=\tfrac{1}{2}\left(\mathrm{TPR}(t)+\mathrm{TNR}(t)\right)")
    B.p("The two agree when $\\pi=1/2$. Accuracy rewards predicting the majority class when the classes are unbalanced, whereas balanced accuracy weights the two classes equally whatever "
        "$\\pi$ is. Both depend on the threshold. The threshold of the project is the one that maximises balanced accuracy on the validation split of the training data set, and accuracy at a "
        "fixed 0.5 is reported only as a reference.")
    # ------------------------------------------------------------------
    B.h2("Probabilistic Metrics: Log Loss, Brier Score and Calibration Error", trace="MATH_NOTES section 6; PROTOCOL section 4; DECISIONS D-024")
    B.p("Three metrics see the probabilities and not only their ranking. The logarithmic loss is the negative mean log likelihood of the labels under the predicted probabilities; "
        "the probabilities are clipped to $[10^{-7},1-10^{-7}]$ before it is computed:")
    B.equation("logloss", r"\mathrm{LL}=-\frac{1}{n}\sum_{i=1}^{n}\left[y_i\ln\hat p_i+(1-y_i)\ln(1-\hat p_i)\right]")
    B.p("The Brier score is the mean squared error of the probabilities, a proper scoring rule whose expected value is minimised only by reporting the true conditional probability. "
        "Always predicting 0.5 gives 0.25 and always predicting the prevalence gives $\\pi(1-\\pi)$:")
    B.equation("brier", r"\mathrm{BS}=\frac{1}{n}\sum_{i=1}^{n}(\hat p_i-y_i)^2\in[0,1]")
    B.p("If the forecasts take $K$ distinct values $p_k$, with $n_k$ cases having that value, $\\bar o_k$ the observed fake fraction among them and $\\bar o=\\pi$ the overall fake fraction, "
        "the Murphy decomposition separates reliability, resolution and uncertainty:")
    B.equation("murphy", r"\mathrm{BS}=\underbrace{\frac{1}{n}\sum_{k}n_k(p_k-\bar o_k)^2}_{\text{reliability}}-\underbrace{\frac{1}{n}\sum_{k}n_k(\bar o_k-\bar o)^2}_{\text{resolution}}+\underbrace{\bar o(1-\bar o)}_{\text{uncertainty}}")
    B.p("For continuous forecasts the sums are taken over bins and the identity then holds only approximately. The reliability term plays the role of a squared-error version of the "
        "calibration gap.")
    B.p("A classifier is perfectly calibrated when $P(Y=1\\mid\\hat p=p)=p$ for all $p\\in[0,1]$. The expected calibration error (ECE) measures the departure by binning. Fix $M$ equal-width "
        "bins $I_m=(\\tfrac{m-1}{M},\\tfrac{m}{M}]$ and let $B_m=\\{i:\\hat p_i\\in I_m\\}$. The project uses the top-label version of Guo et al. <c calib>: the confidence of example $i$ is "
        "$\\hat c_i=\\max(\\hat p_i,1-\\hat p_i)$, its predicted label is $\\hat y_i=\\mathbf{1}(\\hat p_i\\ge 0.5)$, the examples are binned by confidence and")
    B.equation("ece", r"\mathrm{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}\left|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\right|")
    B.p("where the accuracy and the mean confidence of a bin are")
    B.equation("ece_bins", r"\mathrm{acc}(B_m)=\frac{1}{|B_m|}\sum_{i\in B_m}\mathbf{1}(\hat y_i=y_i),\qquad \mathrm{conf}(B_m)=\frac{1}{|B_m|}\sum_{i\in B_m}\hat c_i")
    B.p("with $M=15$. A second version bins the fake probability itself and compares the mean label with the mean probability in each bin; it gives a different number for the same "
        "predictions, so the version is always stated. ECE depends on $M$ and on the binning rule, is zero for a perfectly calibrated model and is estimated with bias on small bins.")
    # ------------------------------------------------------------------
    B.h2("Platt Scaling and Temperature Scaling", trace="MATH_NOTES section 7; DECISIONS D-024")
    B.p("Given a raw score, for example a logit $s_i$, Platt scaling maps it to a calibrated probability by a logistic function with two scalar parameters fitted on a held-out calibration set "
        "by minimising the negative log likelihood <c calib>:")
    B.equation("platt", r"q_i=\sigma(a\,s_i+b),\qquad \sigma(u)=\frac{1}{1+e^{-u}}")
    B.equation("platt_fit", r"(\hat a,\hat b)=\arg\min_{a,b}\;-\sum_i\left[y_i\ln q_i+(1-y_i)\ln(1-q_i)\right]")
    B.p("This is a logistic regression with one feature, so the objective is convex in $(a,b)$. If $\\hat a>0$ the map is increasing, so the ranking, the AUC and the EER do not change; "
        "only the probabilities change. Temperature scaling is the special case $b=0$ and $a=1/T$ with the single parameter $T>0$ fitted on the validation set by the same objective:")
    B.equation("temp", r"q_i=\sigma\!\left(\frac{s_i}{T}\right),\qquad \hat T=\arg\min_{T>0}\;-\sum_i\left[y_i\ln q_i+(1-y_i)\ln(1-q_i)\right]")
    B.p("It does not change the predicted class. The calibration set must be separate from the training data and from the test data. The project fits $T$ on the validation split of the "
        "training data set and reports ECE and the Brier score before and after scaling. Temperature scaling cannot repair a shift between data sets, so a large ECE on a cross-dataset "
        "test set after scaling is a finding about the shift and not an error of the fit.")
    # ------------------------------------------------------------------
    B.h2("Group Bootstrap Intervals", trace="MATH_NOTES section 8; PROTOCOL section 6")
    B.p("The data consist of $N$ independent units, here the groups of the protocol, and each unit may contain many examples such as frames or clips. A metric $\\theta$ (AUC, EER, balanced "
        "accuracy, ECE, Brier score) is computed from all examples and gives $\\hat\\theta$. Examples inside one group are dependent, so the group and not the example is resampled. "
        "For $b=1,\\dots,B$ the procedure draws $N$ groups with replacement, keeping all examples of each drawn group, and computes $\\hat\\theta^{*b}$ on the resample $D^{*b}$. "
        "The $(1-\\alpha)$ percentile interval is")
    B.equation("boot", r"\left[q_{\alpha/2},\;q_{1-\alpha/2}\right],\qquad q_u=u\text{-quantile of }\left\{\hat\theta^{*b}\right\}_{b=1}^{B}")
    B.p("The project uses $B=2000$ resamples, $\\alpha=0.05$ and a fixed base random seed. A resample that contains only one class is drawn again, because the AUC and the EER do not exist "
        "for one class. For a small number of groups the percentile interval can cover less than its nominal level; this is why an exact paired sign-flip permutation test over groups is run as "
        "a sensitivity check beside the bootstrap on the small development slices.")
    B.p("When a candidate is trained with several seeds, the metric is computed for every seed on each resample and averaged over the $S$ seeds, and the interval is that of the "
        "seed-averaged value:")
    B.equation("seedavg", r"\bar\theta^{*b}=\frac{1}{S}\sum_{s=1}^{S}\theta_s\!\left(D^{*b}\right)")
    B.p("The standard deviation over seeds of the point estimate is reported next to it. The two sources of uncertainty are different: resampling groups measures the uncertainty due to which "
        "groups are in the test set, whereas the seeds measure the variation due to training. Neither covers the other.")
    # ------------------------------------------------------------------
    B.h2("Paired Comparison of Two Models", trace="MATH_NOTES section 8.1; DECISIONS comparison rules 4 and 5")
    B.p("Let models $A$ and $B$ be scored on the same test groups. The quantity of interest is $\\Delta=\\theta_A-\\theta_B$. For each resample the same drawn groups are used for both "
        "models, and")
    B.equation("paired", r"\Delta^{*b}=\theta_A\!\left(D^{*b}\right)-\theta_B\!\left(D^{*b}\right),\qquad p_{\mathrm{boot}}=2\min\!\left(\frac{\#\{b:\Delta^{*b}\le 0\}}{B},\frac{\#\{b:\Delta^{*b}\ge 0\}}{B}\right)")
    B.p("The percentile interval of $\\{\\Delta^{*b}\\}$ is the interval of the difference. If it contains zero the data do not separate the two models at that level. Pairing matters because")
    B.equation("varpair", r"\mathrm{Var}(\hat\Delta)=\mathrm{Var}(\hat\theta_A)+\mathrm{Var}(\hat\theta_B)-2\,\mathrm{Cov}(\hat\theta_A,\hat\theta_B)")
    B.p("and two models scored on the same groups have positively correlated errors, so the covariance term makes the paired interval narrower than a comparison of two separate "
        "intervals by eye.")
    # ------------------------------------------------------------------
    B.h2("Non-inferiority and the Tie Set", trace="DECISIONS D-025 and D-036; PROTOCOL addendum")
    B.p("Published detectors report numbers under protocols of their own, so \u201ccomparable with a published detector\u201d is defined as non-inferior to a re-run of that detector on the "
        "same data, split, budget and metric. Let $\\Delta^{*b}$ be the paired differences (ours minus the baseline) resampled over groups, and let $\\delta>0$ be a margin that is "
        "stated before the final comparison and chosen from the measured seed-to-seed spread of the baseline. Then")
    B.equation("noninf", r"\text{non-inferior:}\;\; q_{0.025}\!\left(\Delta^{*}\right)>-\delta,\qquad \text{superior:}\;\; q_{0.025}\!\left(\Delta^{*}\right)>0")
    B.p("An interval that only contains zero is reported as not distinguished, never as comparable. The same paired interval defines the tie set of the encoder comparison. If $\\ell$ is the "
        "candidate with the highest mean AUC on the development slice, the tie set is")
    B.equation("tieset", r"\mathcal{T}=\left\{c:\;0\in\mathrm{CI}_{95}\!\left(\theta_{\ell}-\theta_{c}\right)\right\}")
    B.p("and the cheapest member of $\\mathcal{T}$ wins, where cheaper means the lower end-to-end extraction time per 1,000 inputs and, if equal, fewer parameters. The rule follows from "
        "the comparison principle of the project: if the intervals of the difference include zero, the evidence does not separate the candidates, and the simpler or cheaper one is preferred.")
    # ------------------------------------------------------------------
    B.h2("Which Quantity Sees What", trace="MATH_NOTES section 9")
    B.p("<ref tab:sees> summarises what each metric depends on. It is the reason for the choice of a ranking metric as the headline and of calibration metrics as secondary results.")
    B.table("sees", "What each quantity depends on.",
            ["Quantity", "Uses only ranking", "Depends on a threshold", "Depends on prevalence", "Sees calibration"],
            [["ROC curve, AUC", "yes", "no", "no (population value)", "no"],
             ["EER", "yes", "defined at its own threshold", "no", "no"],
             ["TPR, FPR, TNR, FNR", "no", "yes", "no", "no"],
             ["Precision", "no", "yes", "yes", "no"],
             ["Accuracy", "no", "yes", "yes", "no"],
             ["Balanced accuracy", "no", "yes", "no", "no"],
             ["ECE", "no", "no", "indirectly", "yes"],
             ["Brier score", "no", "no", "yes", "yes, and ranking quality"],
             ["Platt or temperature scaling", "not a metric", "fitted on a calibration set", "via the calibration set", "changes calibration only"]],
            [4.3, 2.4, 3.4, 3.2, 3.1], ["left", "center", "center", "center", "center"], size=10)
    # ------------------------------------------------------------------
    B.h2("Linear Probe, Binary Cross-Entropy and Class Weights", trace="MATH_NOTES section 10; iteration I-0")
    B.p("The working version places a linear head on frozen, standardised features $z_i\\in\\mathbb{R}^d$. With $p_i=\\sigma(w^{\\top}z_i+b)$ the binary cross-entropy is")
    B.equation("probe_loss", r"L(w,b)=\frac{1}{n}\sum_{i=1}^{n}\left[-y_i\ln p_i-(1-y_i)\ln(1-p_i)\right]")
    B.p("Its gradient and Hessian are")
    B.equation("probe_grad", r"\nabla_w L=\frac{1}{n}\sum_i(p_i-y_i)\,z_i,\qquad \nabla_w^2 L=\frac{1}{n}\sum_i p_i(1-p_i)\,z_i z_i^{\top}\succeq 0")
    B.p("so the problem is convex and its optimum does not depend on the initialisation; the seeds change only the order of the batches and the point at which early stopping fires. With "
        "class weights $w_1=\\tfrac{n}{2n_1}$ and $w_0=\\tfrac{n}{2n_0}$ the gradient becomes")
    B.equation("classw", r"\nabla_w L_w=\frac{1}{n}\sum_i w_{y_i}\,(p_i-y_i)\,z_i")
    B.p("which is the gradient of the risk under a balanced prior $\\pi=\\tfrac{1}{2}$. This matters because the training sets are unbalanced: FaceForensics++ has four fake videos for every real "
        "video and ASVspoof 2019 has about nine spoofed utterances per bona fide utterance. The assumption is that the feature distribution of the training set is the one the head is "
        "applied to. The probe does not help when the features do not separate the classes, in which case the optimum is near chance whatever the weights, or when the shift between training "
        "and test data dominates.")
    # ------------------------------------------------------------------
    B.h2("Fine-tuning Depth: Low-rank Adapters", trace="MATH_NOTES section 11; iteration I-1; LoRA paper")
    B.p("A frozen weight matrix $W\\in\\mathbb{R}^{d\\times k}$ is replaced by a sum with a low-rank update <c lora>:")
    B.equation("lora", r"W'=W+\frac{\alpha}{r}\,B\,A,\qquad B\in\mathbb{R}^{d\times r}\ (B=0\text{ at start}),\quad A\in\mathbb{R}^{r\times k},\quad r\ll\min(d,k)")
    B.p("With $G=\\partial L/\\partial W'$ the gradients are")
    B.equation("lora_grad", r"\frac{\partial L}{\partial B}=\frac{\alpha}{r}\,G\,A^{\top},\qquad \frac{\partial L}{\partial A}=\frac{\alpha}{r}\,B^{\top}G,\qquad \#\text{trainable}=r(d+k)\ \text{instead of}\ dk")
    B.p("Because $B=0$ at the start, $W'=W$ at step zero, so the frozen model is the starting point, and only $B$ receives a gradient in the first step. The update has rank at most $r$. "
        "The assumption is that the useful change of the representation is well approximated by a low-rank matrix. The lever does not help when the missing signal is not in the span of "
        "low-rank updates, when the training data are too few for the extra parameters, or when the gap is a distribution shift: more capacity then fits source-specific cues such as identity "
        "or compression, and the score on the development slice can fall while the validation score rises <c identity>.")
    # ------------------------------------------------------------------
    B.h2("Frame-to-video Aggregation", trace="MATH_NOTES section 12; iteration I-2; DECISIONS D-024")
    B.p("Let the frame scores $s_1,\\dots,s_T$ of one video have a common variance $\\sigma^2$ and a pairwise correlation $\\rho$. The variance of their mean is")
    B.equation("aggvar", r"\mathrm{Var}\!\left(\frac{1}{T}\sum_{t=1}^{T}s_t\right)=\frac{\sigma^2}{T}\left(1+(T-1)\rho\right)\ \longrightarrow\ \rho\,\sigma^2\quad (T\to\infty)")
    B.p("so averaging removes the independent part of the noise but not the shared part. If only $m$ of the $T$ frames carry the manipulation and each is shifted by $\\mu$, the mean is "
        "shifted by only $\\tfrac{m}{T}\\mu$, so the signal is diluted, whereas the mean of the top $k=m$ scores keeps the shift $\\mu$ but adds selection bias. For $T$ independent "
        "$N(0,\\sigma^2)$ scores")
    B.equation("aggmax", r"\mathbb{E}\!\left[\max_t s_t\right]\approx\sigma\sqrt{2\ln T}")
    B.p("which raises the score of real videos too and so raises false alarms. The clip score of the working version is the plain mean of the frame probabilities. The assumptions are equal "
        "variance and exchangeable frames. The alternatives do not help when $\\rho\\approx 1$, because the frames are then nearly identical and any aggregate is the same number, or when "
        "every frame is informative, because the mean is then near-optimal. That the rule matters in practice is shown by an audio-visual benchmark, where the video AUC of one model was "
        "61.44 with the mean of the frame scores and 68.81 with the mean of the five highest <c avdf1m>.")
    # ------------------------------------------------------------------
    B.h2("Loss Functions", trace="MATH_NOTES section 13; iteration I-3")
    B.p("For a positive example with $p=\\sigma(\\ell)$ the focal loss is $L=-(1-p)^{\\gamma}\\ln p$, and its derivative with respect to the logit is")
    B.equation("focal", r"\frac{dL}{d\ell}=-(1-p)^{\gamma}\left(1-p-\gamma\,p\ln p\right)")
    B.p("For $\\gamma=0$ this is $p-1$, the gradient of the cross-entropy; for $\\gamma>0$ the factor $(1-p)^{\\gamma}$ shrinks the gradient of examples that are already well classified. "
        "Label smoothing replaces $y$ by $\\tilde y=(1-\\varepsilon)y+\\varepsilon/2$; the gradient is $p-\\tilde y$, and the minimiser of the expected loss has a finite logit instead of "
        "diverging on separable data:")
    B.equation("smooth", r"\ell^{*}=\ln\frac{1-\varepsilon/2}{\varepsilon/2}")
    B.p("A pairwise surrogate of the AUC uses the logistic loss on score differences:")
    B.equation("pairwise", r"L=\frac{1}{n_1 n_0}\sum_{i:\,y_i=1}\ \sum_{j:\,y_j=0}\ln\!\left(1+e^{-(s_i-s_j)}\right)")
    B.p("Since $\\ln(1+e^{-m})\\ge\\ln 2\\cdot\\mathbf{1}[m\\le 0]$, the quantity $L/\\ln 2$ is a convex upper bound on $1-\\mathrm{AUC}$. The assumption is that the loss is the only change "
        "and the features are fixed. Because the AUC depends only on the ranking, all three change the AUC little for a linear head on fixed features; they matter for calibration, balanced "
        "accuracy and the behaviour on rare hard examples.")
    # ------------------------------------------------------------------
    B.h2("Regularisation and Augmentation", trace="MATH_NOTES section 14; iteration I-4")
    B.p("Ridge regularisation (weight decay) on a linear model with the singular value decomposition $X=U\\Sigma V^{\\top}$ gives")
    B.equation("ridge", r"\hat w_\lambda=\left(X^{\top}X+n\lambda I\right)^{-1}X^{\top}y,\qquad \text{component }j\text{ is scaled by }\frac{s_j^2}{s_j^2+n\lambda}")
    B.p("relative to least squares, and its variance changes from $\\sigma^2/s_j^2$ to $\\sigma^2 s_j^2/(s_j^2+n\\lambda)^2$: the directions with small singular values, which are the "
        "noisy ones, are damped most, at the price of a bias. This is the maximum a posteriori estimate under a Gaussian prior on $w$ with variance $1/(n\\lambda)$ per coordinate. "
        "Augmentation replaces empirical risk minimisation by")
    B.equation("augment", r"\min_{\theta}\ \mathbb{E}_{(x,y)}\,\mathbb{E}_{t\sim\mathcal{T}}\ L\!\left(f_\theta(t(x)),y\right)")
    B.p("that is, empirical risk minimisation under the pushed-forward distribution; invariance is learned only to the transformations in $\\mathcal{T}$. Mixup trains on convex "
        "combinations of examples,")
    B.equation("mixup", r"\tilde x=\lambda x_i+(1-\lambda)x_j,\qquad \tilde y=\lambda y_i+(1-\lambda)y_j,\qquad \lambda\sim\mathrm{Beta}(a,a)")
    B.p("which penalises scores that change sharply between examples. The assumption is that the test perturbations (compression, blur, resize) are covered by $\\mathcal{T}$. The lever "
        "does not help when the shift is of a kind not in $\\mathcal{T}$, such as a new generator family, when the model is under-fitting so that the problem is bias and not variance, or when "
        "the regularisation strength is tuned on the same data it is judged on.")
    # ------------------------------------------------------------------
    B.h2("Combining Encoders", trace="MATH_NOTES section 15; iteration I-5")
    B.p("For $m$ scores with equal variance $\\sigma^2$ and average pairwise correlation $\\bar\\rho$,")
    B.equation("ens", r"\mathrm{Var}\!\left(\frac{1}{m}\sum_{j=1}^{m}s_j\right)=\frac{1}{m^2}\left(m\sigma^2+m(m-1)\bar\rho\sigma^2\right)=\sigma^2\left(\bar\rho+\frac{1-\bar\rho}{m}\right)")
    B.p("The bias of the average is the average of the biases, so only the variance part falls. Concatenating the features raises the head size from $d_j$ to $\\sum_j d_j$, which for a small "
        "training set can increase variance, whereas averaging the scores adds no parameters. The scores must be on a common scale, that is, logits after the same calibration, before "
        "averaging. The lever does not help when $\\bar\\rho\\to 1$ because the encoders then make the same errors, when one member is much worse because the average inherits its bias, "
        "or when the cost or the licence restrictions of the extra encoders outweigh a gain smaller than the interval.")
    # ------------------------------------------------------------------
    B.h2("Late Fusion of Audio and Visual Scores", trace="MATH_NOTES section 16; iteration I-6")
    B.p("Let $x_a$ and $x_v$ be the audio and the visual input. If they are conditionally independent given the class,")
    B.equation("fusion_odds", r"\ln\frac{P(y=1\mid x_a,x_v)}{P(y=0\mid x_a,x_v)}=\ln\frac{\pi}{1-\pi}+\ln\frac{p(x_a\mid 1)}{p(x_a\mid 0)}+\ln\frac{p(x_v\mid 1)}{p(x_v\mid 0)}")
    B.p("A calibrated unimodal model trained at prior $\\pi$ gives $\\ell_m=\\ln\\frac{\\pi}{1-\\pi}+\\mathrm{LLR}_m$, where LLR is the log likelihood ratio of that modality, so the fused logit is")
    B.equation("fused", r"\ell_{\text{fused}}=\ell_a+\ell_v-\ln\frac{\pi}{1-\pi}")
    B.p("By the Neyman-Pearson lemma the likelihood-ratio test is the most powerful at every false positive rate, so its ROC curve dominates all others and its AUC is maximal among rules based "
        "on $(x_a,x_v)$. Temperature scaling fits one scalar $T$ per modality; it is monotone, so it leaves the AUC of each modality unchanged and only makes the sum meaningful. A logistic "
        "regression with three parameters is the alternative:")
    B.equation("fusion_lr", r"p(y=1\mid x_a,x_v)=\sigma\!\left(w_0+w_a\ell_a+w_v\ell_v\right)")
    B.p("With dependent modalities the plain sum counts shared evidence twice, and the fitted weights $w_a,w_v$ come out below 1. Fusion does not help when one modality is near chance, "
        "because it then adds noise and the fused AUC can fall below the better single modality; when the modalities are strongly dependent and the weights are not fitted; when the "
        "modality labels differ, because a video can be fake in the audio only, the picture only or both, so that the label of the fused question has to be defined first; or when the calibration "
        "is poor. A zero-shot audio-visual comparison supports this caution: joint late fusion was no better than independent detectors combined by an OR rule in a published study <c zhou>.")
    # ------------------------------------------------------------------
    B.h2("The Pulse Signal and Its Signal-to-noise Ratio", trace="MATH_NOTES section 17; iteration I-7; POS paper")
    B.p("Let $c(t)=[R(t),G(t),B(t)]^{\\top}$ be the mean skin colour of the face region per frame. A projection-based pulse method <c pos> normalises $C_n=c/\\bar c$ over a sliding "
        "window, forms two projections and combines them so that the common specular and motion component cancels:")
    B.equation("pos", r"S_1=G_n-B_n,\qquad S_2=-2R_n+G_n+B_n,\qquad h=S_1+\frac{\sigma(S_1)}{\sigma(S_2)}\,S_2")
    B.p("where $\\sigma$ is the standard deviation over the window. The pulse frequency is the peak of the power spectrum of $h$ in $[0.7,3]$ Hz, so the heart rate is $60f$ beats per minute. "
        "The signal-to-noise ratio is the power within a small band $\\delta$ around the peak, and around its second harmonic, divided by the remaining power in the band:")
    B.equation("snr", r"\mathrm{SNR}=\frac{\sum_{f\in\mathcal{B}_\delta}P(f)}{\sum_{f\in[0.7,3]\setminus\mathcal{B}_\delta}P(f)}")
    B.p("The negative control shuffles the frame order of each clip, which destroys the temporal structure and flattens the spectrum; the distribution of the SNR of the shuffled clips is the "
        "null against which the real clips are compared. The method assumes visible skin, stable illumination, small motion and enough frames at the native frame rate. It does not work when "
        "video compression removes the sub-percent colour fluctuation, when faces are too small or move too much, or when the frames are resampled so that the temporal sampling is lost, "
        "as in the 16-frame sampling of the working version, which is why the pulse experiment extracts contiguous frames separately.")
    # ------------------------------------------------------------------
    B.h2("Summary and Outline of the Next Chapter")
    B.p("This chapter defined the metrics that rank, threshold and calibrate a detector, the group bootstrap and the paired tests that attach an interval to every comparison, and the "
        "short argument for each lever of the iteration ladder together with the condition under which it does not help. Chapter 4 applies them: it describes the environment, the data "
        "and their licences, the preprocessing, the design of the encoder comparison, the working version, the iterations, the web application and the test protocol.")
