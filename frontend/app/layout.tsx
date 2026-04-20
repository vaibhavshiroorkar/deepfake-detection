import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Veritas — Forensic review for images, video, and text",
  description:
    "A careful second opinion on synthetic media. We examine images, video, and writing for the fingerprints of manipulation — and show our work.",
  openGraph: {
    title: "Veritas — Forensic review for images, video, and text",
    description:
      "A careful second opinion on synthetic media. We show our work.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="relative min-h-screen">
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  );
}
