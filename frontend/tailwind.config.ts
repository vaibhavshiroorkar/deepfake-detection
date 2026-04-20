import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bone: "#F4EFE6",
        paper: "#FBF8F1",
        ink: "#141413",
        smoke: "#3A3835",
        mute: "#8A8478",
        rule: "#D9D1C2",
        ember: "#D84727",
        amber: "#E8A33D",
        forest: "#2F5D45",
        alert: "#B8341F",
      },
      fontFamily: {
        display: ['"Fraunces"', "ui-serif", "Georgia", "serif"],
        sans: ['"Inter Tight"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      letterSpacing: {
        tightest: "-0.04em",
        tighter: "-0.025em",
      },
      boxShadow: {
        paper: "0 1px 0 rgba(20,20,19,0.06), 0 10px 30px -18px rgba(20,20,19,0.25)",
      },
    },
  },
  plugins: [],
};
export default config;
