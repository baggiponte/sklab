export const colors = {
  // Base
  background: "#09090b",
  backgroundAlt: "#18181b",
  codeBackground: "#1c1c1e",

  // Brand
  accentBlue: "#3b82f6",
  accentBlueBright: "#60a5fa",
  accentBlueGlow: "rgba(59, 130, 246, 0.4)",

  // Semantic
  successGreen: "#22c55e",
  successGreenBright: "#4ade80",
  successGreenGlow: "rgba(34, 197, 94, 0.3)",
  warningRed: "#ef4444",
  warningRedBright: "#f87171",

  // Text
  text: "#fafafa",
  textMuted: "#a1a1aa",
  textDim: "#71717a",

  // Code syntax
  codePurple: "#c084fc",
  codeYellow: "#fde047",
  codeOrange: "#fb923c",
  codeCyan: "#22d3ee",
  codeGreen: "#86efac",
  codePink: "#f472b6",

  // Effects
  gradientStart: "#09090b",
  gradientEnd: "#1e1b4b",
} as const;

// Gradient backgrounds
export const gradients = {
  radialGlow: `radial-gradient(ellipse 80% 50% at 50% 50%, ${colors.accentBlueGlow} 0%, transparent 70%)`,
  radialGlowGreen: `radial-gradient(ellipse 60% 40% at 50% 50%, ${colors.successGreenGlow} 0%, transparent 60%)`,
  subtleVignette: `radial-gradient(ellipse 100% 100% at 50% 50%, transparent 40%, rgba(0,0,0,0.4) 100%)`,
  diagonalSheen: `linear-gradient(135deg, ${colors.background} 0%, ${colors.gradientEnd} 100%)`,
} as const;
