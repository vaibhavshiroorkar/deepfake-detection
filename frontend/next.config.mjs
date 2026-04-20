/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    serverActions: { bodySizeLimit: "200mb" },
  },
  env: {
    BACKEND_URL: process.env.BACKEND_URL || "http://localhost:8000",
  },
};

export default nextConfig;
