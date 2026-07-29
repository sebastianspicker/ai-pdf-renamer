import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => {
  const demo = mode === "demo";
  return {
    plugins: [react()],
    publicDir: demo ? "demo-public" : false,
    resolve: {
      alias: {
        "@api": new URL(demo ? "./src/demo-api.ts" : "./src/api.ts", import.meta.url).pathname,
      },
    },
    test: {
      environment: "jsdom",
      setupFiles: "./src/test-setup.ts",
    },
    build: {
      outDir: "../src/folionym/web_dist",
      emptyOutDir: true,
      sourcemap: false,
    },
    server: {
      host: "127.0.0.1",
      port: 5173,
      proxy: {
        "/api": "http://127.0.0.1:8765",
      },
    },
  };
});
