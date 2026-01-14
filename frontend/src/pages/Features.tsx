import { FloatingNav, BackgroundPathsWrapper } from "@/components/ui";
import { Features as FeaturesSection } from "@/components/home";
import { Footer } from "@/components/layout";

const navItems = [
  { name: "Home", link: "/" },
  { name: "How It Works", link: "/how-it-works" },
  { name: "Technology", link: "/technology" },
];

export function FeaturesPage() {
  return (
    <div className="min-h-screen bg-black">
      <FloatingNav navItems={navItems} />

      <BackgroundPathsWrapper>
        <div className="pt-20">
          <FeaturesSection />
        </div>
        <Footer />
      </BackgroundPathsWrapper>
    </div>
  );
}
