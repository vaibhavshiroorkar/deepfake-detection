import Masthead from "@/components/Masthead";
import Footer from "@/components/Footer";
import KeysClient from "./KeysClient";

export const dynamic = "force-dynamic";

export default function KeysPage() {
  return (
    <main>
      <Masthead />
      <KeysClient />
      <Footer />
    </main>
  );
}
