import Masthead from "@/components/Masthead";
import DetectorConsole from "@/components/DetectorConsole";
import Footer from "@/components/Footer";

export const metadata = {
  title: "Detect · Veritas",
  description: "Run text, image, audio, or video through Veritas.",
};

export default function DetectPage() {
  return (
    <main>
      <Masthead />
      <section className="section-screen">
        <div className="page-frame flex-1 flex flex-col pt-16 pb-10">
          <div className="flex items-baseline justify-between mb-8">
            <span className="running-head">Workspace · drop something in</span>
            <span className="running-head hidden sm:inline">Live check</span>
          </div>
          <h1 className="display-lg max-w-[22ch]">
            Let&rsquo;s have a <span className="italic text-ember">look</span>.
          </h1>
          <p className="body-lead mt-8 max-w-[54ch] text-smoke">
            Pick the kind of thing you&rsquo;re checking. Drop the file or
            paste the text. The answer shows up below with every signal
            that went into it.
          </p>
          <div className="mt-12 md:mt-14 flex-1">
            <DetectorConsole />
          </div>
        </div>
      </section>
      <Footer />
    </main>
  );
}
