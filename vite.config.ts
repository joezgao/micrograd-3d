import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// BASE_PATH is set only by the GitHub Pages workflow (/micrograd-3d/); local dev and tests keep "/".
export default defineConfig({
  base: process.env.BASE_PATH ?? "/",
  plugins: [react()],
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
