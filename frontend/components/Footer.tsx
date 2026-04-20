"use client";

export default function Footer() {
  return (
    <footer id="colophon" className="relative border-t border-ink bg-ink text-paper">
      <div className="mx-auto max-w-[1400px] px-6 py-14">
        <div className="grid md:grid-cols-12 gap-10">
          <div className="md:col-span-6">
            <div className="font-mono text-[11px] uppercase tracking-[0.28em] text-ember">
              Colophon
            </div>
            <h4
              className="mt-3 font-display tracking-tight"
              style={{ fontSize: "clamp(2rem, 4vw, 3.2rem)", lineHeight: 1 }}
            >
              Veritas<span className="text-ember">.</span>
            </h4>
            <p className="mt-4 max-w-prose text-mute leading-[1.7]">
              Set in Fraunces, Inter Tight, and JetBrains Mono. Backend in
              Python with Pillow, OpenCV, NumPy, and SciPy. Frontend in Next.js
              and Framer Motion. No user data is retained.
            </p>
          </div>

          <div className="md:col-span-6 grid grid-cols-2 gap-6 text-sm">
            <div>
              <div className="font-mono text-[10px] uppercase tracking-[0.24em] text-mute">Sections</div>
              <ul className="mt-3 space-y-1.5">
                <li><a href="#console" className="hover:text-ember">The Desk</a></li>
                <li><a href="#how" className="hover:text-ember">Method</a></li>
                <li><a href="#caveats" className="hover:text-ember">Caveats</a></li>
              </ul>
            </div>
            <div>
              <div className="font-mono text-[10px] uppercase tracking-[0.24em] text-mute">Further reading</div>
              <ul className="mt-3 space-y-1.5 text-paper/90">
                <li>Error-level analysis (Krawetz, 2007)</li>
                <li>GLTR: detecting machine prose (Gehrmann et al., 2019)</li>
                <li>Spectral signatures of GAN images (Durall et al., 2020)</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="mt-14 pt-6 border-t border-paper/10 flex flex-wrap items-center justify-between gap-4 font-mono text-[10px] uppercase tracking-[0.22em] text-mute">
          <span>© {new Date().getFullYear()} Veritas — for educational use.</span>
          <span>Truth · evidence · the slow second look.</span>
        </div>
      </div>
    </footer>
  );
}
