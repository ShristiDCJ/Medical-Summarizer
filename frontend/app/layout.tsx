import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Medical Summarizer",
  description: "Summarize medical transcripts with a custom transformer model."
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

