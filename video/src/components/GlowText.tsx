import React from "react";
import { spring, useCurrentFrame, useVideoConfig } from "remotion";
import { fonts } from "../styles/fonts";
import { colors } from "../styles/colors";

interface GlowTextProps {
  children: React.ReactNode;
  color?: string;
  glowColor?: string;
  fontSize?: number;
  fontWeight?: number;
  startFrame?: number;
  style?: React.CSSProperties;
}

export const GlowText: React.FC<GlowTextProps> = ({
  children,
  color = colors.accentBlue,
  glowColor,
  fontSize = 72,
  fontWeight = 700,
  startFrame = 0,
  style,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const relativeFrame = frame - startFrame;

  if (relativeFrame < 0) return null;

  const opacity = spring({
    frame: relativeFrame,
    fps,
    config: { damping: 200 },
  });

  const scale = spring({
    frame: relativeFrame,
    fps,
    config: { damping: 12, stiffness: 100 },
  });

  const effectiveGlowColor = glowColor || color;

  return (
    <span
      style={{
        position: "relative",
        display: "inline-block",
        fontFamily: fonts.sans,
        fontSize,
        fontWeight,
        color,
        opacity,
        transform: `scale(${scale})`,
        textShadow: `0 0 40px ${effectiveGlowColor}, 0 0 80px ${effectiveGlowColor}`,
        ...style,
      }}
    >
      {children}
    </span>
  );
};
