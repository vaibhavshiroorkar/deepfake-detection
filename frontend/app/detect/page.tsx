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
      <section className="mx-auto max-w-3xl px-6 pt-12 pb-6">
        <p className="text-xs uppercase tracking-[0.2em] text-mute">Workspace</p>
        <h1
          className="font-display tracking-tight mt-2"
          style={{ fontSize: "clamp(1.8rem, 3.4vw, 2.6rem)", lineHeight: 1.1 }}
        >
          Run an examination
        </h1>
        <p className="mt-3 max-w-xl text-sm text-smoke leading-[1.7]">
          Pick a modality, drop the file or paste the text. The verdict appears
          below with the underlying signals shown.
        </p>
      </section>
      <DetectorConsole />
      <Footer />
    </main>
  );
}
