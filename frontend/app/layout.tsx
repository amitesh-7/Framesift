import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FrameSift | Semantic Video Search Engine",
  description:
    "Semantic Video Search powered by Hybrid AI. Find any moment in your videos using natural language queries.",
  keywords: [
    "video search",
    "semantic search",
    "AI",
    "CLIP",
    "NVIDIA NIM",
    "video analysis",
  ],
  authors: [{ name: "FrameSift Team" }],
  openGraph: {
    title: "FrameSift | Semantic Video Search Engine",
    description:
      "Find any moment in your videos using natural language queries.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
