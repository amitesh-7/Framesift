import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Search } from "lucide-react";
import { useGoogleLogin } from "@react-oauth/google";
import { Button } from "@/components/ui";
import { useAuthStore } from "@/store";

export function CTA() {
  const { isAuthenticated, login } = useAuthStore();

  const googleLogin = useGoogleLogin({
    onSuccess: async (response) => {
      try {
        const res = await fetch(
          "https://www.googleapis.com/oauth2/v3/userinfo",
          {
            headers: { Authorization: `Bearer ${response.access_token}` },
          }
        );
        const data = await res.json();
        login(
          {
            id: data.sub,
            email: data.email,
            name: data.name,
            picture: data.picture,
          },
          response.access_token
        );
      } catch (error) {
        console.error("Login failed:", error);
      }
    },
  });

  return (
    <section className="py-20">
      <div className="max-w-3xl mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="p-10 rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/80 to-zinc-900/40 text-center"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4">
            Ready to Get Started?
          </h2>
          <p className="text-zinc-400 mb-8">
            Transform your video library into a searchable knowledge base.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            {isAuthenticated ? (
              <Link to="/search">
                <Button size="lg">
                  <Search className="w-4 h-4 mr-2" />
                  Go to Dashboard
                </Button>
              </Link>
            ) : (
              <Button size="lg" onClick={() => googleLogin()}>
                Start Free with Google
              </Button>
            )}
          </div>
        </motion.div>
      </div>
    </section>
  );
}

export default CTA;
