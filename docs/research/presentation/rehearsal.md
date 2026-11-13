# Research presentation rehearsal

Each member receives five slides and four minutes. Total speaking time is twelve minutes, followed by questions.
This is a speaking allocation, not a claim that one member alone performed the work on those slides.
All three members share equal project credit and prepare the full report.

| Member | Slides | Time | Coverage |
|---|---|---|---|
| Vaibhav Shiroorkar | 1 to 5 | 4 minutes | Question, two research lines, early history and protocol |
| Shivansh Gupta | 6 to 10 | 4 minutes | Architecture, sealed results, threshold risk, corrections and exploratory work |
| Adithya Jithesh | 11 to 15 | 4 minutes | Interpretation, November plans, demonstration, conclusion and equal credit |

## Questions to prepare

| Question | Answer |
|---|---|
| What improved? | Under the team sealed protocol, video AUROC rose from 0.6238 to 0.8218 and speech from 0.7516 to 0.9148. |
| Did fusion generalize better? | The root historical comparison needs nested cross-fitting before that conclusion is supportable. |
| Did you beat Xception? | The corrected superiority interval includes zero. State non-inferiority under the declared margin. |
| Is lip-sync proven? | No. Attention statistics alone do not establish the mechanism. Corrected windows and causal perturbations are needed. |
| Are the new evaluations independent? | They use unread samples from seen distributions and are exploratory. |
| What did November add? | AV-HuBERT implementation and preregistered E-035/E-036. The result slots are still empty. |
| Can the score prove a video is fake? | No. Thresholds can fail under distribution shift, and missing evidence can require an inconclusive finding. |
| Who receives credit? | All three members receive one third across the whole project and every current deliverable. |

## Demonstration

Before presenting, check the local environment and checkpoint requirements in `docs/SETUP.md` at tag `team/submission-final`.
Follow `docs/FINAL_DEMO_RUNBOOK.md` at tag `team/submission-final` for the detailed sequence.
Use an approved local example. Explain which modules ran and which could not produce evidence.
Keep mock mode visible if real checkpoints are missing. Do not present a mock output as a measured detector result.

For an optional three-minute demonstration, give each member one minute: input and scope, module output, then limitations.
Keep the core twelve-minute presentation usable without a network connection or a live model.
