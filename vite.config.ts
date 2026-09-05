import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/rl": {
        target: "http://127.0.0.1:8765",
        changeOrigin: true,
        timeout: 1_800_000,
        proxyTimeout: 1_800_000,
        rewrite: (p) => p.replace(/^\/rl/, ""),
      },
    },
  },
  optimizeDeps: {
    exclude: ["@dimforge/rapier3d-compat"],
  },
});
