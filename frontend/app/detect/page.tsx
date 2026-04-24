import Masthead from "@/components/Masthead";
import DetectorConsole from "@/components/DetectorConsole";
import Footer from "@/components/Footer";

export const metadata = {
  title: "Detect — Veritas",
  description: "Run text, image, audio, or video through Veritas.",
};

export default function DetectPage() {
  return (
    <main>
      <Masthead />
      <section className="section-screen">
        <div className="page-frame flex-1 flex flex-col pt-16 pb-10">
          <div className="flex items-baseline justify-between mb-8">
            <span className="running-head">Workspace — file a piece of evidence</span>
            <span className="running-head hidden sm:inline">Live inference</span>
          </div>
          <h1 className="display-lg max-w-[22ch]">
            Run an <span className="italic text-ember">examination</span>.
          </h1>
          <p className="body-lead mt-8 max-w-[54ch] text-smoke">
            Pick a modality. Drop the file or paste the text. The verdict
            appears below, with every contributing signal shown.
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
