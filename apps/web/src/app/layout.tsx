import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RelationScope · Relation Intelligence",
  description:
    "A production workspace for document relation intelligence, open-set prediction and review-ready evidence.",
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
