import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the Python API runs on :8000; Vite proxies API, live events and media to it.
const api = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: api, changeOrigin: true },
      "/media": { target: api, changeOrigin: true },
    },
  },
});
