import { Footer } from "@/components/layout";
import {
  Hero,
  Models,
  HowItWorks,
  Features,
  Technology,
  CTA,
} from "@/components/home";
import { FloatingNav, BackgroundPathsWrapper } from "@/components/ui";

const navItems = [
  { name: "Features", link: "#features" },
  { name: "How It Works", link: "#how-it-works" },
  { name: "Technology", link: "#tech" },
];

export function HomePage() {
  return (
    <div className="min-h-screen bg-black">
      {/* Permanent Floating Navbar */}
      <FloatingNav navItems={navItems} />

      {/* Hero with Raining Letters (no background paths) */}
      <Hero />

      {/* Rest of content with Background Paths */}
      <BackgroundPathsWrapper>
        <Models />
        <HowItWorks />
        <Features />
        <Technology />
        <CTA />
        <Footer />
      </BackgroundPathsWrapper>
    </div>
  );
}
