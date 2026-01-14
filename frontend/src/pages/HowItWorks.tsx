import { FloatingNav, BackgroundPathsWrapper } from "@/components/ui";
import { HowItWorks as HowItWorksSection } from "@/components/home";
import { Footer } from "@/components/layout";

const navItems = [
  { name: "Home", link: "/" },
  { name: "Features", link: "/features" },
  { name: "Technology", link: "/technology" },
];

export function HowItWorksPage() {
  return (
    <div className="min-h-screen bg-black">
      <FloatingNav navItems={navItems} />

      <BackgroundPathsWrapper>
        <div className="pt-20">
          <HowItWorksSection />
        </div>
        <Footer />
      </BackgroundPathsWrapper>
    </div>
  );
}
