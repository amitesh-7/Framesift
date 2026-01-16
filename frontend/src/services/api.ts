import axios from "axios";
import { config } from "@/config";

const api = axios.create({
  baseURL: config.apiUrl,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor - adds auth token and user ID for multi-tenancy
api.interceptors.request.use((cfg) => {
  const token = localStorage.getItem("framesift-auth");
  if (token) {
    const parsed = JSON.parse(token);
    if (parsed?.state?.token) {
      cfg.headers.Authorization = `Bearer ${parsed.state.token}`;
    }
    // Add user ID for multi-user support
    if (parsed?.state?.user?.id) {
      cfg.headers["X-User-Id"] = parsed.state.user.id;
    }
  }
  return cfg;
});

export default api;
