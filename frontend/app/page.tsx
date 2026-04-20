import Masthead from "@/components/Masthead";
import Hero from "@/components/Hero";
import DetectorConsole from "@/components/DetectorConsole";
import HowItWorks from "@/components/HowItWorks";
import Caveats from "@/components/Caveats";
import Footer from "@/components/Footer";

export default function Page() {
  return (
    <main>
      <Masthead />
      <Hero />
      <DetectorConsole />
      <HowItWorks />
      <Caveats />
      <Footer />
    </main>
  );
}
