import React from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Video, LogOut } from "lucide-react";
import { useGoogleLogin } from "@react-oauth/google";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/store";
import { config } from "@/config";

export const FloatingNav = ({
  navItems,
  className,
}: {
  navItems: {
    name: string;
    link: string;
    icon?: React.ReactNode;
  }[];
  className?: string;
}) => {
  const { user, isAuthenticated, login, logout } = useAuthStore();

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

        const userData = {
          id: data.sub,
          email: data.email,
          name: data.name,
          picture: data.picture,
        };

        // Store in frontend
        login(userData, response.access_token);

        // Send to backend for admin tracking
        try {
          await fetch(`${config.apiUrl}/admin/track-login`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "X-Admin-Key": config.adminKey,
            },
            body: JSON.stringify(userData),
          });
        } catch (trackError) {
          console.error("Failed to track login:", trackError);
          // Continue anyway - don't block user login
        }
      } catch (error) {
        console.error("Login failed:", error);
      }
    },
    onError: () => console.error("Google login failed"),
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={cn(
        "flex w-full max-w-4xl fixed top-4 inset-x-0 mx-auto border border-white/[0.1] rounded-full bg-black/80 backdrop-blur-xl shadow-[0px_2px_3px_-1px_rgba(0,0,0,0.1),0px_1px_0px_0px_rgba(25,28,33,0.02),0px_0px_0px_1px_rgba(25,28,33,0.08)] z-[5000] px-4 py-2 items-center justify-between",
        className
      )}
    >
      {/* Logo */}
      <Link to="/" className="flex items-center gap-2">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
          <Video className="w-4 h-4 text-white" />
        </div>
        <span className="font-semibold text-white hidden sm:inline">
          FrameSift
        </span>
      </Link>

      {/* Nav Links */}
      <div className="hidden md:flex items-center gap-6">
        {navItems.map((navItem, idx: number) => (
          <Link
            key={`link-${idx}`}
            to={navItem.link}
            className="relative text-zinc-400 text-sm hover:text-white transition-colors"
          >
            {navItem.name}
          </Link>
        ))}
      </div>

      {/* Auth */}
      <div className="flex items-center gap-2">
        {isAuthenticated && user ? (
          <>
            {user.picture ? (
              <img
                src={user.picture}
                alt={user.name}
                className="w-7 h-7 rounded-full"
                onError={(e) => {
                  e.currentTarget.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(
                    user.name
                  )}&background=7c3aed&color=fff&size=128`;
                }}
              />
            ) : (
              <div className="w-7 h-7 rounded-full bg-violet-500 flex items-center justify-center text-white text-xs font-medium">
                {user.name.charAt(0).toUpperCase()}
              </div>
            )}
            <button
              onClick={logout}
              className="p-2 text-zinc-400 hover:text-white transition-colors"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </>
        ) : (
          <button
            onClick={() => googleLogin()}
            className="border text-sm font-medium relative border-white/[0.2] text-white px-4 py-1.5 rounded-full hover:bg-white/10 transition-colors"
          >
            <span>Sign In</span>
            <span className="absolute inset-x-0 w-1/2 mx-auto -bottom-px bg-gradient-to-r from-transparent via-violet-500 to-transparent h-px" />
          </button>
        )}
      </div>
    </motion.div>
  );
};
