"use client";

export default function Caveats() {
  return (
    <section id="caveats" className="relative">
      <div className="mx-auto max-w-[1400px] px-6 py-20 md:py-24">
        <div className="grid md:grid-cols-12 gap-10">
          <div className="md:col-span-4">
            <div className="font-mono text-[11px] uppercase tracking-[0.28em] text-ember">
              §IV — An honest note
            </div>
            <h3
              className="mt-4 font-display tracking-tight text-ink"
              style={{ fontSize: "clamp(1.8rem, 3.2vw, 2.6rem)", lineHeight: 1.05 }}
            >
              What this tool
              <br />
              <span className="italic text-smoke">cannot do.</span>
            </h3>
          </div>

          <div className="md:col-span-8">
            <div className="border-l-2 border-ember pl-6 md:pl-8 text-[1.02rem] leading-[1.75] text-smoke space-y-5">
              <p>
                Veritas is a <em>reading</em>, not a ruling. The same signals
                that light up a clever forgery also light up a re-saved JPEG,
                a low-bitrate video, or a careful copy editor's pass. We are
                in the business of measuring suspicion, not assigning guilt.
              </p>
              <p>
                Generative models change weekly. A method that works on
                yesterday's diffusion artifacts may miss tomorrow's. We keep
                our signals plural and our thresholds conservative for that
                reason — and we will be wrong, in both directions, sometimes.
              </p>
              <p>
                If the stakes are real, a single automated reading is not
                enough. Find the first-party publisher. Look for the higher
                resolution original. Ask who benefits from your belief. Then
                come back and see whether our reading agrees with you.
              </p>
            </div>

            <div className="mt-10 grid sm:grid-cols-3 gap-4 text-[12px] font-mono uppercase tracking-[0.2em] text-smoke">
              {[
                ["No models are stored", "Uploads are processed in memory and discarded."],
                ["No accounts", "No sign-in, no tracking, no newsletter."],
                ["Open method", "Every signal is documented. Read the method above."],
              ].map(([t, b]) => (
                <div key={t} className="border border-rule bg-paper p-5">
                  <div className="text-ember">{t}</div>
                  <div className="mt-2 text-mute normal-case tracking-normal leading-snug text-xs">
                    {b}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
