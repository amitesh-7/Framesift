import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface User {
  id: string;
  email: string;
  name: string;
  picture: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (user: User, token: string) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      login: (user, token) => set({ user, token, isAuthenticated: true }),
      logout: async () => {
        // Clear Pinecone database on logout
        try {
          const apiUrl =
            import.meta.env.VITE_API_URL || "http://localhost:8000";
          await fetch(`${apiUrl}/clear-database`, {
            method: "POST",
          });
          console.log("Database cleared on logout");
        } catch (error) {
          console.error("Failed to clear database:", error);
        }
        // Clear local auth state
        set({ user: null, token: null, isAuthenticated: false });
      },
    }),
    {
      name: "framesift-auth",
    }
  )
);
