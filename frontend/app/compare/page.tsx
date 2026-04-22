import Masthead from "@/components/Masthead";
import Footer from "@/components/Footer";
import CompareClient from "./CompareClient";

export const dynamic = "force-dynamic";

export default function ComparePage() {
  return (
    <main>
      <Masthead />
      <CompareClient />
      <Footer />
    </main>
  );
}
