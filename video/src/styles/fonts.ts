import { loadFont as loadInter } from "@remotion/google-fonts/Inter";
import { loadFont as loadJetBrainsMono } from "@remotion/google-fonts/JetBrainsMono";

// Load Inter with specific weights
export const { fontFamily: interFamily } = loadInter("normal", {
  weights: ["400", "500", "600", "700"],
  subsets: ["latin"],
});

// Load JetBrains Mono for code (better readability than Fira Code)
export const { fontFamily: jetbrainsFamily } = loadJetBrainsMono("normal", {
  weights: ["400", "500", "600", "700"],
  subsets: ["latin"],
});

export const fonts = {
  sans: interFamily,
  mono: jetbrainsFamily,
} as const;

// Typography presets
export const typography = {
  hero: {
    fontFamily: interFamily,
    fontSize: 72,
    fontWeight: 700,
    lineHeight: 1.1,
    letterSpacing: "-0.02em",
  },
  title: {
    fontFamily: interFamily,
    fontSize: 48,
    fontWeight: 700,
    lineHeight: 1.2,
    letterSpacing: "-0.01em",
  },
  subtitle: {
    fontFamily: interFamily,
    fontSize: 36,
    fontWeight: 500,
    lineHeight: 1.3,
  },
  body: {
    fontFamily: interFamily,
    fontSize: 28,
    fontWeight: 400,
    lineHeight: 1.5,
  },
  code: {
    fontFamily: jetbrainsFamily,
    fontSize: 22,
    fontWeight: 400,
    lineHeight: 1.6,
  },
  codeLarge: {
    fontFamily: jetbrainsFamily,
    fontSize: 28,
    fontWeight: 500,
    lineHeight: 1.5,
  },
} as const;
