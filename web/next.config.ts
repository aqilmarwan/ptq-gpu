import type { NextConfig } from "next";
import { fileURLToPath } from "node:url";

const nextConfig: NextConfig = {
  // Self-contained server bundle for a slim Docker runtime image.
  output: "standalone",
  // Pin the workspace root to this app so the dev/build server doesn't pick up
  // an unrelated lockfile higher up the filesystem.
  turbopack: {
    root: fileURLToPath(new URL(".", import.meta.url)),
  },
};

export default nextConfig;
