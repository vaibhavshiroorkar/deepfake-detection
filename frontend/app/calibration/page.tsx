import Masthead from "@/components/Masthead";
import Footer from "@/components/Footer";

// Synthetic illustrative distributions. Replace with measurements from a
// real labeled test set as you collect one — see `scripts/calibrate.py`.
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

export default function CalibrationPage() {
  return (
    <main>
      <Masthead />
      <section className="mx-auto max-w-4xl px-6 py-12">
        <h1 className="font-display text-3xl">Calibration</h1>
        <p className="mt-3 text-sm text-smoke max-w-prose leading-relaxed">
          A score of 60 means nothing on its own. The chart below shows the
          empirical distribution of suspicion scores on a labeled set of real
          and synthetic samples. Where the two curves cross is the honest
          decision boundary; how much they overlap tells you the residual
          ambiguity at any given score.
        </p>
        <p className="mt-3 text-xs text-mute max-w-prose">
          The data here is illustrative until a real test set is dropped in.
          Run <code className="text-ink">python scripts/calibrate.py</code> on
          a folder of labeled samples to overwrite these histograms.
        </p>

        <div className="mt-10 space-y-12">
          {(Object.keys(DISTRIBUTIONS) as (keyof typeof DISTRIBUTIONS)[]).map((kind) => (
            <Histogram key={kind} kind={kind} dist={DISTRIBUTIONS[kind]} />
          ))}
        </div>
      </section>
      <Footer />
    </main>
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
