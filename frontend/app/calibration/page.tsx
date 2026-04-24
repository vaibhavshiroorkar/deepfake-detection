import Masthead from "@/components/Masthead";
import Footer from "@/components/Footer";

export const metadata = {
  title: "Calibration · Veritas",
  description:
    "Where Veritas sits today, what it gets right, what it gets wrong, and how it compares to the published research.",
};

const DISTRIBUTIONS: Record<
  "image" | "video" | "audio" | "text",
  { real: number[]; fake: number[] }
> = {
  image: {
    real: [4, 9, 18, 22, 18, 12, 7, 5, 3, 2],
    fake: [1, 2, 4, 7, 11, 15, 18, 17, 14, 11],
  },
  video: {
    real: [6, 12, 19, 22, 17, 11, 6, 4, 2, 1],
    fake: [1, 2, 3, 6, 10, 14, 18, 19, 16, 11],
  },
  audio: {
    real: [3, 8, 17, 24, 20, 13, 7, 4, 3, 1],
    fake: [1, 2, 3, 5, 9, 14, 18, 20, 17, 11],
  },
  text: {
    real: [5, 11, 18, 21, 17, 12, 8, 4, 3, 1],
    fake: [1, 1, 3, 5, 9, 14, 18, 19, 18, 12],
  },
};

const BINS = 10;
const BIN_LABELS = Array.from({ length: BINS }, (_, i) => `${i * 10}–${i * 10 + 10}`);

type AccuracyRow = {
  kind: string;
  today: string;
  ceiling: string;
  blocker: string;
};

const ACCURACY: AccuracyRow[] = [
  {
    kind: "Image",
    today: "75 to 80% on SDXL-era generations. Drops to 55 to 65% on newer diffusion models (Flux, Midjourney v6) and heavy post-processing.",
    ceiling: "88 to 92% with a DINOv2 head trained on the FF++, DFDC, Celeb-DF, WildDeepFake union.",
    blocker: "No fine-tuned head yet. Current classifiers were trained on earlier generator families.",
  },
  {
    kind: "Video",
    today: "60 to 70% on FF++ and DFDC. Weaker on face-swap because we only run per-frame image checks with a light temporal pass.",
    ceiling: "85 to 90% with CNN-RNN or ViViT temporal fusion plus audio-visual dissonance.",
    blocker: "No trained temporal model. Video pipeline currently ignores the audio track.",
  },
  {
    kind: "Audio",
    today: "Around 90% on ASVspoof-style synthetic speech. Shakier on recent cloning systems (ElevenLabs, XTTS).",
    ceiling: "93 to 95% with a Whisper head fine-tuned on WaveFake plus 2024-era cloned voices.",
    blocker: "The Whisper classifier head scaffold is in place but untrained.",
  },
  {
    kind: "Text",
    today: "Around 65% on modern LLM output. RoBERTa + GPT-2 were calibrated against GPT-3 era writing.",
    ceiling: "80 to 85% with a discriminator fine-tuned on GPT-4, Claude, Gemini, and Llama 3 output.",
    blocker: "Needs a current corpus of labelled human vs LLM writing. We don't have one.",
  },
];

type CompareRow = {
  finding: string;
  veritas: "yes" | "partial" | "no";
  note: string;
};

const COMPARISON: CompareRow[] = [
  {
    finding: "CNN-based spatial methods dominate the field",
    veritas: "yes",
    note: "Primary image classifiers are Swin-v2 and the DINOv2 scaffold, both spatial.",
  },
  {
    finding: "Temporal methods are needed for video",
    veritas: "partial",
    note: "We have a temporal-flicker heuristic and a TemporalTransformer scaffold, but no trained weights yet.",
  },
  {
    finding: "Multimodal fusion (audio + visual) raises accuracy",
    veritas: "no",
    note: "Real gap. The video pipeline ignores the audio track. Chugh et al. show this is worth 15 to 20% on face-swap video.",
  },
  {
    finding: "Ensemble methods hit 88 to 97% on benchmarks",
    veritas: "yes",
    note: "Veritas is an ensemble. Two image classifiers, two audio models, two text models. Calibration and agreement-adjustment logic match the paper's recipe.",
  },
  {
    finding: "Detectors overfit to FF++, DFDC, Celeb-DF. Performance drops cross-dataset.",
    veritas: "partial",
    note: "Sidestepped by not training, but we inherit it from the upstream models. Organika/sdxl-detector was trained on SDXL, so newer generators slip through.",
  },
  {
    finding: "Capsule networks, disentangled representation, rPPG",
    veritas: "no",
    note: "None implemented. rPPG (heart rate from facial colour, Vinay et al.) would be a real forensic lift and isn't in the repo.",
  },
  {
    finding: "46.3% of reviewed studies claim their model generalises",
    veritas: "partial",
    note: "The paper's headline caveat is that state-of-the-art still degrades on unseen manipulations. We agree with the caveat more than the claim.",
  },
  {
    finding: "Standardised datasets with scoring systems are needed",
    veritas: "no",
    note: "This page gestures at it, but we don't have a real benchmark suite yet.",
  },
];

export default function CalibrationPage() {
  return (
    <main>
      <Masthead />
      <section className="page-frame py-16 max-w-5xl">
        <div className="flex items-baseline justify-between mb-10">
          <span className="running-head">Report 001 · Calibration</span>
          <span className="running-head hidden sm:inline">Where we actually are</span>
        </div>

        <h1 className="display-xl max-w-[18ch]">
          What the score means,
          <br />
          <span className="italic text-ink/70">and what it doesn&rsquo;t.</span>
        </h1>

        <p className="body-lead mt-10 max-w-[54ch] text-smoke">
          A number like 60 on its own is almost meaningless. The point of
          calibration is showing what the number has looked like, in
          practice, on things we already knew the answer to. That&rsquo;s
          what this page is about. There&rsquo;s also an honest comparison
          with the published research, because otherwise we&rsquo;d just
          be claiming things.
        </p>

        <p className="body-sm mt-5 max-w-[54ch] text-mute">
          The histograms below are illustrative while we collect a real
          labelled test set. Running <code className="text-ink">python
          scripts/calibrate.py</code> over a folder of known real and
          synthetic samples overwrites them with measurements.
        </p>

        <div className="mt-14 space-y-12">
          {(Object.keys(DISTRIBUTIONS) as (keyof typeof DISTRIBUTIONS)[]).map((kind) => (
            <Histogram key={kind} kind={kind} dist={DISTRIBUTIONS[kind]} />
          ))}
        </div>

        {/* Accuracy today vs ceiling */}
        <div className="mt-24">
          <div className="flex items-baseline justify-between mb-6">
            <span className="running-head">Report 002 · Accuracy</span>
            <span className="running-head hidden sm:inline">Today vs. what&rsquo;s possible</span>
          </div>
          <h2 className="display-lg max-w-[20ch]">
            Where it sits,
            <br />
            <span className="italic text-ink/70">and where it could go.</span>
          </h2>

          <p className="body mt-8 max-w-[58ch]">
            Rough estimates, not benchmark numbers. Until we run a real
            evaluation these are informed guesses based on the
            performance claimed by upstream models and our own spot
            testing. The ceiling column assumes a couple of weeks of
            training work per modality.
          </p>

          <div className="mt-10 border-y border-rule divide-y divide-rule">
            {ACCURACY.map((row) => (
              <div
                key={row.kind}
                className="grid grid-cols-1 md:grid-cols-[8rem_1fr] gap-x-10 gap-y-3 py-6"
              >
                <div className="font-display text-xl leading-tight">
                  {row.kind}
                </div>
                <div className="space-y-3 text-smoke">
                  <div>
                    <span className="eyebrow mr-3 text-ink">Today</span>
                    {row.today}
                  </div>
                  <div>
                    <span className="eyebrow mr-3 eyebrow-ember">Ceiling</span>
                    {row.ceiling}
                  </div>
                  <div>
                    <span className="eyebrow mr-3 text-mute">Blocker</span>
                    <span className="text-mute">{row.blocker}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Research comparison */}
        <div className="mt-24">
          <div className="flex items-baseline justify-between mb-6">
            <span className="running-head">Report 003 · Against the literature</span>
            <span className="running-head hidden sm:inline">What the research says</span>
          </div>
          <h2 className="display-lg max-w-[22ch]">
            How we stack up against
            <br />
            <span className="italic text-ink/70">a systematic review.</span>
          </h2>

          <p className="body mt-8 max-w-[60ch]">
            Ramanaharan, Guruge, and Agbinya published a systematic
            review of 108 deepfake video detection studies in{" "}
            <em>Data and Information Management</em> (2025). It covers
            work from 2018 to early 2024 and lands on a handful of clear
            conclusions. Here&rsquo;s how Veritas compares, honestly.
          </p>

          <div className="mt-10 border-y border-rule divide-y divide-rule">
            {COMPARISON.map((row) => (
              <div
                key={row.finding}
                className="grid grid-cols-[4rem_1fr] md:grid-cols-[6rem_22rem_1fr] gap-x-8 gap-y-2 py-5 items-start"
              >
                <div>
                  <StatusPill status={row.veritas} />
                </div>
                <div className="font-display text-lg leading-snug md:col-span-1 col-span-1">
                  {row.finding}
                </div>
                <div className="body-sm text-smoke md:col-start-3 col-span-2 md:col-span-1">
                  {row.note}
                </div>
              </div>
            ))}
          </div>

          <p className="body-sm text-mute mt-8 max-w-[58ch]">
            Reference: Ramanaharan, R., Guruge, D. B., &amp; Agbinya, J. I.
            (2025). DeepFake video detection: Insights into model
            generalisation. A systematic review.{" "}
            <em>Data and Information Management, 9</em>(2), 100099.
          </p>
        </div>

        {/* The hedge */}
        <div className="mt-24 border-t border-rule pt-12">
          <div className="flex items-baseline justify-between mb-4">
            <span className="running-head">Report 004 · The hedge</span>
          </div>
          <h2 className="display-md max-w-[24ch]">
            The paper&rsquo;s takeaway is our takeaway.
          </h2>
          <p className="body mt-6 max-w-[60ch]">
            Cross-dataset generalisation in deepfake detection is
            unsolved. The best published models hit 92 to 96% on the
            dataset they were trained on, and 65 to 75% on everything
            else. Until generation and detection hit a saturation point
            together, nobody is going to nail 95% on arbitrary in-the-wild
            fakes. A realistic ceiling for a careful ensemble like this
            one is around 85%, with honest hedging on the other 15%.
            That&rsquo;s why &ldquo;inconclusive&rdquo; is a real verdict
            here and not a failure state.
          </p>
        </div>
      </section>
      <Footer />
    </main>
  );
}

function StatusPill({ status }: { status: "yes" | "partial" | "no" }) {
  const cfg = {
    yes: { label: "Match", cls: "bg-forest text-paper" },
    partial: { label: "Partial", cls: "bg-amber text-ink" },
    no: { label: "Gap", cls: "bg-alert text-paper" },
  }[status];
  return (
    <span
      className={`inline-block px-2.5 py-0.5 font-mono text-[10px] tracking-widest uppercase ${cfg.cls}`}
    >
      {cfg.label}
    </span>
  );
}

function Histogram({
  kind,
  dist,
}: {
  kind: string;
  dist: { real: number[]; fake: number[] };
}) {
  const max = Math.max(...dist.real, ...dist.fake);
  const realArea = sum(dist.real);
  const fakeArea = sum(dist.fake);
  const overlap = sum(dist.real.map((r, i) => Math.min(r / realArea, dist.fake[i] / fakeArea)));

  return (
    <div className="border border-rule bg-paper">
      <div className="px-5 py-3 border-b border-rule flex items-baseline justify-between">
        <h2 className="font-display capitalize text-xl">{kind}</h2>
        <span className="text-xs text-mute">overlap {Math.round(overlap * 100)}%</span>
      </div>
      <div className="px-5 py-6">
        <div className="grid grid-cols-10 gap-2 items-end h-40">
          {dist.real.map((r, i) => {
            const f = dist.fake[i];
            return (
              <div key={i} className="flex flex-col items-center gap-1">
                <div className="relative w-full h-full flex items-end gap-0.5">
                  <div
                    className="flex-1 bg-forest/70"
                    style={{ height: `${(r / max) * 100}%` }}
                    title={`real: ${r}`}
                  />
                  <div
                    className="flex-1 bg-ember/70"
                    style={{ height: `${(f / max) * 100}%` }}
                    title={`fake: ${f}`}
                  />
                </div>
              </div>
            );
          })}
        </div>
        <div className="mt-2 grid grid-cols-10 gap-2 text-[10px] text-mute text-center tabular-nums">
          {BIN_LABELS.map((l) => (
            <span key={l}>{l}</span>
          ))}
        </div>
        <div className="mt-4 flex items-center gap-5 text-xs text-smoke">
          <span className="flex items-center gap-1.5">
            <span className="size-2.5 bg-forest/70" /> real
          </span>
          <span className="flex items-center gap-1.5">
            <span className="size-2.5 bg-ember/70" /> synthetic
          </span>
        </div>
      </div>
    </div>
  );
}

function sum(xs: number[]) {
  return xs.reduce((a, b) => a + b, 0);
}
