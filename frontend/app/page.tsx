import Masthead from "@/components/Masthead";
import Hero from "@/components/Hero";
import Capabilities from "@/components/Capabilities";
import HowItWorks from "@/components/HowItWorks";
import Principles from "@/components/Principles";
import CallToAction from "@/components/CallToAction";
import Footer from "@/components/Footer";

export default function Page() {
  return (
    <main>
      <Masthead />
      <Hero />
      <Capabilities />
      <HowItWorks />
      <Principles />
      <CallToAction />
      <Footer />
    </main>
  );
}
