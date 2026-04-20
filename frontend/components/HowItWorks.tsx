"use client";

import { motion } from "framer-motion";

const METHOD = [
  {
    kicker: "§ III · a",
    title: "For images",
    signals: [
      ["Error-level analysis", "Recompress and measure where the picture argues with itself."],
      ["Noise stationarity", "Sensor grain should be steady across the frame; synthesized regions are too smooth — or too uneven."],
      ["Spectral falloff", "Real lenses and sensors leave a specific radial signature in frequency space. Diffusion and GANs don't."],
      ["Face boundary", "When a face has been swapped onto a body, a faint seam of edges usually survives around the jaw, hairline, and ears."],
      ["Capture metadata", "EXIF tags are weak evidence on their own, but 'Software: Stable Diffusion' is not nothing."],
    ],
  },
  {
    kicker: "§ III · b",
    title: "For video",
    signals: [
      ["Per-frame forensics", "Every sampled frame goes through the image checks. A verdict is an average with an eye on the worst frame."],
      ["Temporal flicker", "Face-generators built frame-by-frame often wobble on fine detail — a mouth that trembles, a specular highlight that jumps."],
      ["Region stability", "Crop boundaries on a swapped face tend to drift a pixel or two across a scene. The eye rarely catches it. Math does."],
    ],
  },
  {
    kicker: "§ III · c",
    title: "For text",
    signals: [
      ["Burstiness", "Humans vary sentence length freely. Language models, even good ones, lean toward a steady middle."],
      ["Lexical rhythm", "Type–token ratios, function-word load, and the small grammar of how ideas connect."],
      ["Phrase tics", "Delve. Tapestry. Landscape. Seamless. Navigate the. A short list of phrases statistically over-represented in LLM output."],
      ["Punctuation entropy", "A comma-heavy, em-dash-light, contraction-light register is a house style. Some humans write that way. Most don't."],
    ],
  },
];

export default function HowItWorks() {
  return (
    <section id="how" className="relative border-t border-rule bg-paper/60">
      <div className="mx-auto max-w-[1400px] px-6 py-20 md:py-28">
        <div className="grid md:grid-cols-12 gap-10">
          <div className="md:col-span-4">
            <div className="font-mono text-[11px] uppercase tracking-[0.28em] text-ember">
              §III — Method
            </div>
            <h3
              className="mt-4 font-display tracking-tight text-ink"
              style={{ fontSize: "clamp(2rem, 3.6vw, 3rem)", lineHeight: 1.02 }}
            >
              We show our work,
              <br />
              <span className="italic text-smoke">line by line.</span>
            </h3>
            <p className="mt-6 text-smoke leading-[1.7] max-w-md">
              No single signal is decisive. A suspicious image may be a
              compressed one. A clean image may be a careful forgery. What
              holds up is agreement across independent channels — the same
              story told in five different forensic languages.
            </p>
            <p className="mt-4 text-smoke leading-[1.7] max-w-md">
              We publish every signal, every score, and a plain-language
              reading of what it means. You are the editor of your own belief.
            </p>
          </div>

          <div className="md:col-span-8 grid gap-10">
            {METHOD.map((m, i) => (
              <motion.div
                key={m.title}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ duration: 0.6, delay: i * 0.08 }}
              >
                <div className="flex items-baseline gap-4 border-b border-ink pb-3">
                  <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-ember">
                    {m.kicker}
                  </span>
                  <h4 className="font-display text-2xl md:text-3xl tracking-tight">
                    {m.title}
                  </h4>
                </div>
                <dl className="mt-5 divide-y divide-rule">
                  {m.signals.map(([name, body]) => (
                    <div key={name} className="py-4 grid md:grid-cols-12 gap-4">
                      <dt className="md:col-span-4 font-mono text-[12px] uppercase tracking-[0.18em] text-smoke">
                        {name}
                      </dt>
                      <dd className="md:col-span-8 text-[0.96rem] leading-[1.65] text-ink">
                        {body}
                      </dd>
                    </div>
                  ))}
                </dl>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
