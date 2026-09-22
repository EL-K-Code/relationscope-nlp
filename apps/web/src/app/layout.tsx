import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RelationScope NLP",
  description:
    "Production-grade relation extraction, calibration and unseen-relation detection.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
