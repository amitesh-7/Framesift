import { Navbar, Footer } from "@/components/layout";
import {
  Hero,
  Models,
  HowItWorks,
  Features,
  Technology,
  CTA,
} from "@/components/home";

export function HomePage() {
  return (
    <div className="min-h-screen bg-black">
      <Navbar />
      <Hero />
      <Models />
      <HowItWorks />
      <Features />
      <Technology />
      <CTA />
      <Footer />
    </div>
  );
}
