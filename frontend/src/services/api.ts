import axios from "axios";
import { config } from "@/config";

const api = axios.create({
  baseURL: config.apiUrl,
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor
api.interceptors.request.use((cfg) => {
  const token = localStorage.getItem("framesift-auth");
  if (token) {
    const parsed = JSON.parse(token);
    if (parsed?.state?.token) {
      cfg.headers.Authorization = `Bearer ${parsed.state.token}`;
    }
  }
  return cfg;
});

export default api;
