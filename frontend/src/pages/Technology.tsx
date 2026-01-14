import { FloatingNav, BackgroundPathsWrapper } from "@/components/ui";
import { Technology as TechnologySection } from "@/components/home";
import { Footer } from "@/components/layout";

const navItems = [
  { name: "Home", link: "/" },
  { name: "Features", link: "/features" },
  { name: "How It Works", link: "/how-it-works" },
];

export function TechnologyPage() {
  return (
    <div className="min-h-screen bg-black">
      <FloatingNav navItems={navItems} />

      <BackgroundPathsWrapper>
        <div className="pt-20">
          <TechnologySection />
        </div>
        <Footer />
      </BackgroundPathsWrapper>
    </div>
  );
}
