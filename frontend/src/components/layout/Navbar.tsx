import { Link, useLocation } from "react-router-dom";
import { Video, Search, LogOut } from "lucide-react";
import { useGoogleLogin } from "@react-oauth/google";
import { Button } from "@/components/ui";
import { useAuthStore } from "@/store";
import { cn } from "@/lib/utils";

export function Navbar() {
  const location = useLocation();
  const { user, isAuthenticated, login, logout } = useAuthStore();
  const isHome = location.pathname === "/";

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
    onError: () => console.error("Google login failed"),
  });

  return (
    <nav
      className={cn(
        "fixed top-0 left-0 right-0 z-50 h-16 border-b border-zinc-800/50",
        isHome ? "bg-black/80 backdrop-blur-xl" : "bg-zinc-950"
      )}
    >
      <div className="max-w-6xl mx-auto h-full px-4 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
            <Video className="w-4 h-4 text-white" />
          </div>
          <span className="font-semibold text-white">FrameSift</span>
        </Link>

        {/* Nav Links */}
        {isHome && (
          <div className="hidden md:flex items-center gap-6">
            <a
              href="#features"
              className="text-sm text-zinc-400 hover:text-white transition-colors"
            >
              Features
            </a>
            <a
              href="#how-it-works"
              className="text-sm text-zinc-400 hover:text-white transition-colors"
            >
              How It Works
            </a>
            <a
              href="#tech"
              className="text-sm text-zinc-400 hover:text-white transition-colors"
            >
              Technology
            </a>
          </div>
        )}

        {/* Auth */}
        <div className="flex items-center gap-3">
          {isAuthenticated && user ? (
            <>
              <Link to="/search">
                <Button variant="ghost" size="sm">
                  <Search className="w-4 h-4 mr-1.5" />
                  Search
                </Button>
              </Link>
              <img
                src={user.picture}
                alt={user.name}
                className="w-8 h-8 rounded-full"
              />
              <Button variant="ghost" size="sm" onClick={logout}>
                <LogOut className="w-4 h-4" />
              </Button>
            </>
          ) : (
            <Button onClick={() => googleLogin()} size="sm">
              Sign In
            </Button>
          )}
        </div>
      </div>
    </nav>
  );
}
