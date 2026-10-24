# MATH_NOTES

Status of this file: BACKGROUND, CANDIDATE METRICS, NOT YET ADOPTED.

Last updated: 2026-10-24 (Session 1)

This note collects definitions and formulas a mathematically minded examiner may ask about. It decides nothing. Which metrics the project uses, which threshold policy it follows and whether it calibrates are open decision O-005 in DECISIONS.md. Listing a quantity here does not mean it will be used.

The text is plain text with LaTeX-style mathematics ($...$), ready to be moved into the paper, report or slides. Sources: the forms of ECE and Platt scaling follow Guo et al. (RESEARCH.md, R-016, read in part). The other quantities are standard definitions; their original sources have not been read in this project, so no citation is given for them.

## 0. Setting and notation

- Each example $i$ has a label $y_i \in \{0, 1\}$ with $1$ meaning fake (the positive class) and $0$ meaning real.
- The model gives a score $s_i \in \mathbb{R}$. When the score is a probability of the fake class it is written $\hat p_i \in [0, 1]$.
- $n_1$ is the number of fakes, $n_0$ the number of real examples, $n = n_0 + n_1$, and $\pi = n_1 / n$ is the fake prevalence.
- A decision rule with threshold $t$ predicts fake when $s_i \ge t$.
- At threshold $t$, $TP, FP, TN, FN$ are the counts of true positives, false positives, true negatives and false negatives.
- Population quantities use $S_1$ for the score of a random fake and $S_0$ for the score of a random real example.

## 1. Error rates at a threshold

$\mathrm{TPR}(t) = \dfrac{TP}{TP + FN} = P(S \ge t \mid Y = 1)$ (true positive rate, recall, sensitivity)

$\mathrm{FPR}(t) = \dfrac{FP}{FP + TN} = P(S \ge t \mid Y = 0)$ (false positive rate)

$\mathrm{FNR}(t) = 1 - \mathrm{TPR}(t)$, $\quad \mathrm{TNR}(t) = 1 - \mathrm{FPR}(t)$ (specificity).

These four do not depend on the prevalence $\pi$, because each is conditioned on the true class.

## 2. ROC curve and AUC

The ROC curve is the set of points $\{(\mathrm{FPR}(t), \mathrm{TPR}(t)) : t \in \mathbb{R}\}$, traced as $t$ moves from $+\infty$ (point $(0,0)$) to $-\infty$ (point $(1,1)$).

The area under it is

$\mathrm{AUC} = \displaystyle\int_0^1 \mathrm{TPR}\, d(\mathrm{FPR}).$

Ranking interpretation. If $S_1$ and $S_0$ have continuous distributions, write $F_0(t) = P(S_0 \le t)$. Then $\mathrm{FPR}(t) = 1 - F_0(t)$ and $\mathrm{TPR}(t) = P(S_1 \ge t)$, and the change of variable $u = 1 - F_0(t)$ gives

$\mathrm{AUC} = \displaystyle\int P(S_1 \ge t)\, dF_0(t) = P(S_1 > S_0).$

With ties allowed the general form is

$\mathrm{AUC} = P(S_1 > S_0) + \tfrac{1}{2} P(S_1 = S_0).$

Empirical estimate (the Mann-Whitney form):

$\widehat{\mathrm{AUC}} = \dfrac{1}{n_1 n_0} \displaystyle\sum_{i : y_i = 1} \sum_{j : y_j = 0} \left[ \mathbf{1}(s_i > s_j) + \tfrac{1}{2} \mathbf{1}(s_i = s_j) \right] = \dfrac{U}{n_1 n_0},$

where $U$ is the Mann-Whitney statistic.

Properties: it takes values in $[0, 1]$, $0.5$ means ranking no better than chance and $1$ means every fake outranks every real example. It is unchanged by any strictly increasing transformation of the scores, so it does not see calibration or the threshold. Its population value does not depend on $\pi$.

## 3. Equal error rate (EER)

The equal error rate is the common value of the two error rates at the threshold $t^\ast$ where they are equal:

$\mathrm{FPR}(t^\ast) = \mathrm{FNR}(t^\ast), \qquad \mathrm{EER} = \mathrm{FPR}(t^\ast).$

On a finite sample the two rates are step functions and may not cross exactly. A common estimate takes $\hat t = \arg\min_t |\mathrm{FPR}(t) - \mathrm{FNR}(t)|$ and reports $\widehat{\mathrm{EER}} = \tfrac{1}{2}\left( \mathrm{FPR}(\hat t) + \mathrm{FNR}(\hat t) \right)$. The threshold $t^\ast$ is a property of the data it is computed on; a threshold taken from one data set and applied to another no longer gives equal errors. Like AUC, the EER uses only rank information of the scores.

## 4. Precision and recall

$\mathrm{Precision}(t) = \dfrac{TP}{TP + FP} = P(Y = 1 \mid S \ge t), \qquad \mathrm{Recall}(t) = \mathrm{TPR}(t) = \dfrac{TP}{TP + FN}.$

Precision depends on the prevalence. By Bayes' rule

$\mathrm{Precision}(t) = \dfrac{\pi\, \mathrm{TPR}(t)}{\pi\, \mathrm{TPR}(t) + (1 - \pi)\, \mathrm{FPR}(t)},$

so the same detector has a different precision on a test set with a different real-to-fake ratio, while its TPR and FPR stay the same.

## 5. Accuracy and balanced accuracy

$\mathrm{Accuracy}(t) = \dfrac{TP + TN}{n} = \pi\, \mathrm{TPR}(t) + (1 - \pi)\, \mathrm{TNR}(t).$

$\mathrm{BalancedAccuracy}(t) = \dfrac{1}{2} \left( \mathrm{TPR}(t) + \mathrm{TNR}(t) \right).$

The two agree when $\pi = 1/2$. Accuracy rewards predicting the majority class when the classes are unbalanced; balanced accuracy weights the two classes equally whatever $\pi$ is. Both depend on the threshold $t$.

## 6. Calibration

A probabilistic classifier is perfectly calibrated when

$P(Y = 1 \mid \hat p = p) = p \quad \text{for all } p \in [0, 1].$

### 6.1 Expected calibration error (ECE)

Fix a number of bins $M$ and split $[0, 1]$ into equal-width intervals $I_m = \left( \tfrac{m-1}{M}, \tfrac{m}{M} \right]$. Let $B_m = \{ i : \hat p_i \in I_m \}$.

Class-1 (fake probability) version:

$\mathrm{ECE} = \displaystyle\sum_{m=1}^{M} \dfrac{|B_m|}{n} \left| \bar y(B_m) - \bar p(B_m) \right|, \qquad \bar y(B_m) = \dfrac{1}{|B_m|} \sum_{i \in B_m} y_i, \quad \bar p(B_m) = \dfrac{1}{|B_m|} \sum_{i \in B_m} \hat p_i.$

Top-label (confidence) version as defined by Guo et al. (R-016): the confidence of example $i$ is $\hat c_i = \max(\hat p_i, 1 - \hat p_i)$ and its predicted label is $\hat y_i = \mathbf{1}(\hat p_i \ge 0.5)$. Bin by confidence and compute

$\mathrm{ECE} = \displaystyle\sum_{m=1}^{M} \dfrac{|B_m|}{n} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|, \qquad \mathrm{acc}(B_m) = \dfrac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat y_i = y_i), \quad \mathrm{conf}(B_m) = \dfrac{1}{|B_m|} \sum_{i \in B_m} \hat c_i.$

The two versions are different numbers for the same predictions, so the version must be stated. R-016 uses the top-label version with $M = 15$. Both depend on $M$ and on the binning rule (equal width here; equal mass is another choice). ECE is a weighted average of absolute bin gaps; it is $0$ for a perfectly calibrated model and is estimated with bias on small bins.

### 6.2 Brier score

$\mathrm{BS} = \dfrac{1}{n} \displaystyle\sum_{i=1}^{n} (\hat p_i - y_i)^2 \in [0, 1].$

It is a proper scoring rule: its expected value is minimised only by reporting the true conditional probability. Reference values: always predicting $0.5$ gives $0.25$; always predicting the prevalence $\pi$ gives $\pi (1 - \pi)$.

Murphy decomposition. If forecasts take $K$ distinct values $p_k$, with $n_k$ cases having that value, $\bar o_k$ the observed fake fraction among them and $\bar o = \pi$ the overall fake fraction, then

$\mathrm{BS} = \underbrace{\dfrac{1}{n} \sum_{k} n_k (p_k - \bar o_k)^2}_{\text{reliability}} - \underbrace{\dfrac{1}{n} \sum_{k} n_k (\bar o_k - \bar o)^2}_{\text{resolution}} + \underbrace{\bar o (1 - \bar o)}_{\text{uncertainty}}.$

For continuous forecasts the sums are taken over bins and the identity then holds only approximately. The reliability term plays the role of a squared-error version of the calibration gap in ECE.

## 7. Platt scaling

Given a raw score (for example a logit) $s_i$, Platt scaling maps it to a calibrated probability by a logistic function with two scalar parameters (R-016, section on calibration methods):

$q_i = \sigma(a\, s_i + b), \qquad \sigma(u) = \dfrac{1}{1 + e^{-u}}.$

The parameters are fitted on a held-out calibration set by minimising the negative log likelihood

$(\hat a, \hat b) = \arg\min_{a, b} \; -\displaystyle\sum_{i} \left[ y_i \ln q_i + (1 - y_i) \ln (1 - q_i) \right].$

This is a logistic regression with one feature, so the objective is convex in $(a, b)$. If $\hat a > 0$ the map is increasing, so the ranking, the AUC and the EER do not change; only the probabilities change. The calibration set must be separate from the training data and from the test data.

Temperature scaling (R-016) is the special case $b = 0$ with $a = 1/T$, that is $q_i = \sigma(s_i / T)$ for a logit $s_i$, with the single parameter $T > 0$ fitted on the validation set by the same negative log likelihood. It does not change the predicted class.

## 8. Bootstrap confidence interval

Setting. The data consist of $N$ independent units (for example identities or source videos); each unit may contain many examples (frames, clips). A metric $\theta$ (AUC, EER, balanced accuracy, ECE, Brier score) is computed from all examples, giving $\hat\theta$. Examples inside one unit are dependent, so the unit and not the example is resampled.

Percentile bootstrap. For $b = 1, \dots, B$:
1. Draw $N$ units with replacement from the $N$ units, keeping all examples of each drawn unit; call the result $D^{\ast b}$.
2. Compute $\hat\theta^{\ast b} = \theta(D^{\ast b})$.

The $(1 - \alpha)$ percentile interval is $\left[ q_{\alpha/2},\; q_{1 - \alpha/2} \right]$, where $q_u$ is the $u$-quantile of $\{ \hat\theta^{\ast b} \}_{b=1}^{B}$. The standard deviation of $\{ \hat\theta^{\ast b} \}$ estimates the standard error of $\hat\theta$.

Practical conditions: a resample must contain both classes for AUC or EER to exist; if units are pure real or pure fake, resampling within each class (stratified) keeps this true and is a choice to be stated. The number of resamples $B$ and the level $1 - \alpha$ are set in Stage 2 (DECISIONS.md, comparison rule 3). For small $N$ the percentile interval can cover less than its nominal level; refinements such as the bias-corrected and accelerated interval exist and are not defined here.

### 8.1 Paired bootstrap for comparing two models

Let models $A$ and $B$ be scored on the same test units, giving metrics $\theta_A(D)$ and $\theta_B(D)$. The quantity of interest is the difference

$\Delta = \theta_A - \theta_B.$

For each resample $D^{\ast b}$ (the same drawn units for both models):

$\Delta^{\ast b} = \theta_A(D^{\ast b}) - \theta_B(D^{\ast b}).$

The percentile interval of $\{ \Delta^{\ast b} \}$ is the interval for the difference. If it contains $0$ the data do not separate the two models at that level, and under comparison rule 5 the simpler or cheaper model is preferred. A bootstrap two-sided p-value can be read as $2 \min\left( \tfrac{1}{B} \#\{ \Delta^{\ast b} \le 0 \},\; \tfrac{1}{B} \#\{ \Delta^{\ast b} \ge 0 \} \right)$.

Pairing matters because

$\mathrm{Var}(\hat\Delta) = \mathrm{Var}(\hat\theta_A) + \mathrm{Var}(\hat\theta_B) - 2\, \mathrm{Cov}(\hat\theta_A, \hat\theta_B),$

and two models scored on the same units have positively correlated errors, so the covariance term makes the paired interval narrower than comparing two separate intervals by eye.

### 8.2 What the bootstrap does not cover

Resampling test units measures the uncertainty due to which units are in the test set. It does not measure the variation due to the random seed of training; that is reported separately as a mean and standard deviation over at least 3 seeds (comparison rules 2 and 3). How to combine the two in one table (for example by computing the metric for every seed on each resample and averaging over seeds) is a Stage 2 choice and is not made here.

## 9. Which quantity sees what

| Quantity | Uses only ranking of scores | Depends on a threshold | Depends on prevalence $\pi$ | Sees calibration |
|---|---|---|---|---|
| ROC curve, AUC | yes | no | no (population value) | no |
| EER | yes | defined at its own threshold | no | no |
| TPR, FPR, TNR, FNR | no | yes | no | no |
| Precision | no | yes | yes | no |
| Accuracy | no | yes | yes | no |
| Balanced accuracy | no | yes | no | no |
| ECE | no | no | indirectly | yes |
| Brier score | no | no | yes | yes (and ranking quality) |
| Platt or temperature scaling | not a metric | fitted on a calibration set | via the calibration set | changes calibration only |


## 10. Linear probe, BCE and class weights (iteration I-0 and the loss slot)

Setting. Standardised features $z_i\in\mathbb{R}^d$, head $p_i=\sigma(w^\top z_i+b)$, loss $L(w,b)=\tfrac1n\sum_i\left[-y_i\ln p_i-(1-y_i)\ln(1-p_i)\right]$. Then $\nabla_w L=\tfrac1n\sum_i (p_i-y_i)\,z_i$ and the Hessian $\tfrac1n\sum_i p_i(1-p_i)\,z_iz_i^\top\succeq 0$, so the problem is convex and the optimum does not depend on the initialisation (seeds change only the batch order and the early-stopping point). With class weights $w_1=\tfrac{n}{2n_1}$, $w_0=\tfrac{n}{2n_0}$ the gradient is $\tfrac1n\sum_i w_{y_i}(p_i-y_i)z_i$, the gradient of the risk under a balanced prior $\pi=\tfrac12$. Assumption: the feature distribution of the training set is the one the head is applied to. Does not help: when the features do not separate the classes (the optimum is then near chance whatever the weights) or when the shift between training and test data dominates.

## 11. Fine-tuning depth: low-rank adapters (iteration I-1)

A frozen weight matrix $W\in\mathbb{R}^{d\times k}$ is replaced by $W'=W+\tfrac{\alpha}{r}BA$ with $B\in\mathbb{R}^{d\times r}$ (initialised to zero), $A\in\mathbb{R}^{r\times k}$ and $r\ll\min(d,k)$. With $G=\partial L/\partial W'$: $\partial L/\partial B=\tfrac{\alpha}{r}GA^\top$ and $\partial L/\partial A=\tfrac{\alpha}{r}B^\top G$. Because $B=0$ at the start, $W'=W$ at step 0 (the frozen model is the starting point) and only $B$ receives a gradient in the first step. The update has rank at most $r$ and the trainable count is $r(d+k)$ instead of $dk$. Assumption: the useful change of the representation is well approximated by a low-rank matrix. Does not help: when the missing signal is not in the span of low-rank updates, when training data are too few for the extra parameters, or when the gap is a distribution shift: more capacity then fits source-specific cues (identity, compression) and the dev_x score can fall while the validation score rises.

## 12. Frame-to-video aggregation (iteration I-2)

Let frame scores $s_1,\dots,s_T$ of one video have common variance $\sigma^2$ and pairwise correlation $\rho$. Then $\operatorname{Var}\left(\tfrac1T\sum_t s_t\right)=\tfrac{\sigma^2}{T}\left(1+(T-1)\rho\right)\to\rho\sigma^2$ as $T\to\infty$, so averaging removes the independent part of the noise but not the shared part. If only $m$ of $T$ frames carry the manipulation, each shifted by $\mu$, the mean is shifted by only $\tfrac{m}{T}\mu$ (the signal is diluted), whereas the mean of the top $k=m$ scores keeps the shift $\mu$ but adds selection bias. For $T$ independent $N(0,\sigma^2)$ scores, $\mathbb{E}[\max_t s_t]\approx\sigma\sqrt{2\ln T}$, which raises the score of real videos too and so raises false alarms. Assumptions: equal variance and exchangeable frames. Does not help: when $\rho\approx1$ (frames nearly identical, so any aggregate is the same number) or when every frame is informative (the mean is then near-optimal).

## 13. Loss functions (loss slot)

Focal loss for $y=1$, $p=\sigma(\ell)$: $L=-(1-p)^{\gamma}\ln p$. Then $\tfrac{dL}{dp}=\gamma(1-p)^{\gamma-1}\ln p-\tfrac{(1-p)^{\gamma}}{p}$ and $\tfrac{dp}{d\ell}=p(1-p)$, so $\tfrac{dL}{d\ell}=-(1-p)^{\gamma}\left(1-p-\gamma p\ln p\right)$. For $\gamma=0$ this is $p-1$, the BCE gradient; for $\gamma>0$ the factor $(1-p)^{\gamma}$ shrinks the gradient of examples that are already well classified ($p$ near 1). Label smoothing replaces $y$ by $\tilde y=(1-\varepsilon)y+\varepsilon/2$; the gradient is $p-\tilde y$, and the minimiser of the expected loss has finite logit $\ell^{\ast}=\ln\tfrac{1-\varepsilon/2}{\varepsilon/2}$ instead of diverging on separable data. Pairwise AUC surrogate: $L=\tfrac1{n_1n_0}\sum_{i:y_i=1}\sum_{j:y_j=0}\ln\left(1+e^{-(s_i-s_j)}\right)$. Since $\ln(1+e^{-m})\ge\ln2\cdot\mathbf 1[m\le0]$, $L/\ln2$ is a convex upper bound on $1-\mathrm{AUC}$. Assumptions: the loss is the only change; features are fixed. Does not help: AUC depends only on the ranking, so for a linear head on fixed features all of these change AUC little; they matter for calibration, balanced accuracy and the behaviour on rare hard examples.

## 14. Regularisation and augmentation (iteration I-4)

Ridge (weight decay) on a linear model with $X=U\Sigma V^\top$: $\hat w_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y$, so component $j$ is multiplied by $\tfrac{s_j^2}{s_j^2+n\lambda}$ relative to least squares, and its variance changes from $\tfrac{\sigma^2}{s_j^2}$ to $\tfrac{\sigma^2 s_j^2}{(s_j^2+n\lambda)^2}$: the directions with small singular values (the noisy ones) are damped most, at the price of a bias. This is the MAP estimate under a Gaussian prior on $w$ with variance $1/(n\lambda)$ per coordinate. Augmentation replaces empirical risk minimisation by $\min_\theta\mathbb{E}_{(x,y)}\mathbb{E}_{t\sim\mathcal T}L(f_\theta(t(x)),y)$, i.e. ERM under the pushed-forward distribution; invariance is learned only to the transformations in $\mathcal T$. Mixup trains on $\tilde x=\lambda x_i+(1-\lambda)x_j$, $\tilde y=\lambda y_i+(1-\lambda)y_j$ with $\lambda\sim\mathrm{Beta}(a,a)$, which penalises scores that change sharply between examples. Assumptions: the test perturbations (compression, blur, resize) are covered by $\mathcal T$. Does not help: when the shift is of a kind not in $\mathcal T$ (a new generator family), when the model is under-fitting (bias, not variance), or when the regularisation strength is tuned on the same data it is judged on.

## 15. Combining encoders (iteration I-5)

For $m$ scores with equal variance $\sigma^2$ and average pairwise correlation $\bar\rho$: $\operatorname{Var}\left(\tfrac1m\sum_j s_j\right)=\tfrac{1}{m^2}\left(m\sigma^2+m(m-1)\bar\rho\sigma^2\right)=\sigma^2\left(\bar\rho+\tfrac{1-\bar\rho}{m}\right)$. The bias of the average is the average of the biases, so only the variance part falls. Concatenating the features raises the head size from $d_j$ to $\sum_j d_j$, which for a small training set can increase variance, whereas averaging the scores adds no parameters. Scores must be on a common scale (logits after the same calibration) before averaging. Does not help: when $\bar\rho\to1$ (the encoders make the same errors), when one member is much worse (the average inherits its bias), or when the extra encoders' cost or licence restrictions outweigh a gain smaller than the interval.

## 16. Late fusion and calibration (iteration I-6)

Let $x_a,x_v$ be the audio and visual inputs. If they are conditionally independent given the class, $\ln\tfrac{P(y=1\mid x_a,x_v)}{P(y=0\mid x_a,x_v)}=\ln\tfrac{\pi}{1-\pi}+\ln\tfrac{p(x_a\mid1)}{p(x_a\mid0)}+\ln\tfrac{p(x_v\mid1)}{p(x_v\mid0)}$. A calibrated unimodal model trained at prior $\pi$ gives $\ell_m=\ln\tfrac{\pi}{1-\pi}+\mathrm{LLR}_m$, so $\ell_{\text{fused}}=\ell_a+\ell_v-\ln\tfrac{\pi}{1-\pi}$. By the Neyman-Pearson lemma the likelihood-ratio test is the most powerful at every false-positive rate, so its ROC curve dominates all others and its AUC is maximal among rules based on $(x_a,x_v)$. Temperature scaling fits one scalar $T$ per modality by minimising the validation negative log-likelihood of $\sigma(\ell/T)$; it is monotone, so it leaves each modality's AUC unchanged and only makes the sum meaningful. Logistic regression fits $\sigma(w_0+w_a\ell_a+w_v\ell_v)$; with dependent modalities the plain sum counts shared evidence twice, and the fitted $w_a,w_v$ come out below 1. Does not help: when one modality is near chance (it adds noise: the fused AUC can fall below the better single one), when the modalities are strongly dependent and the weights are not fitted, when the modality labels differ (a video can be fake in audio only, visual only or both, so the label for the fused question must be defined first, O-009), or when calibration is poor.

## 17. rPPG signal check (iteration I-7)

Let $c(t)=[R(t),G(t),B(t)]^\top$ be the mean skin colour of the face region per frame. A projection-based pulse method (reference NR until read, RESEARCH.md) normalises $C_n=c/\bar c$ over a sliding window, forms $S_1=G_n-B_n$ and $S_2=-2R_n+G_n+B_n$, and outputs $h=S_1+\tfrac{\sigma(S_1)}{\sigma(S_2)}S_2$, where $\sigma$ is the standard deviation over the window; the weight cancels the common (specular and motion) component. The pulse frequency is the peak of the power spectrum of $h$ in $[0.7,3]$ Hz (heart rate $=60f$ bpm). The signal-to-noise ratio is the power within a small band $\delta$ around the peak (and its second harmonic) divided by the remaining power in the band. Negative control: shuffling the frame order destroys the temporal structure, so the spectrum is flat and the SNR distribution of the shuffled clips is the null against which the real clips are compared (percentile of the null). Assumptions: visible skin, stable illumination, small motion, enough frames at the native rate. Does not work when: video compression removes the sub-percent colour fluctuation, faces are too small or too moving, or frames are resampled so the temporal sampling is lost (as in the 16-frame sampling of D-034, hence the separate contiguous extraction).

Addition 2026-10-25 (S-1, E-006; source R-057 equations 32 to 34 and Algorithm 1): the window length of the projection method is $l=\lceil 1.6\,\mathrm{s}\times \text{fps}\rceil$ frames (the source uses $l=32$ at 20 fps, 1.6 s), the window slides one frame, the output $h$ of every window is made zero-mean and overlap-added, and no band-pass filter is applied in the algorithm itself. The SNR band half-width $\delta$ is not given in the text read; E-006 uses $\delta=2/T=0.2$ Hz for $T=10$ s (the half-width of the main lobe of a Hann window), an implementer's choice.
