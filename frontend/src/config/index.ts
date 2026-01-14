export const config = {
  apiUrl: import.meta.env.VITE_API_URL || "http://localhost:8000",
  googleClientId: import.meta.env.VITE_GOOGLE_CLIENT_ID,
  adminKey: import.meta.env.VITE_ADMIN_KEY || "admin-secret-key",
  adminEmails: (import.meta.env.VITE_ADMIN_EMAILS || "")
    .split(",")
    .filter(Boolean),
  app: {
    name: "FrameSift",
    description: "Semantic video search powered by AI",
    author: "Amitesh Vishwakarma",
  },
  features: {
    embedDimensions: 4096,
    modelSize: "90B",
    searchLatency: "<1s",
  },
};
