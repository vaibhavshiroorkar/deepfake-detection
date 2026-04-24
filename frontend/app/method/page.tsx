import Masthead from "@/components/Masthead";
import HowItWorks from "@/components/HowItWorks";
import Footer from "@/components/Footer";

export const metadata = {
  title: "Method · Veritas",
  description: "How Veritas looks at images, video, audio, and text.",
};

export default function MethodPage() {
  return (
    <main>
      <Masthead />
      <HowItWorks />
      <Footer />
    </main>
  );
}
