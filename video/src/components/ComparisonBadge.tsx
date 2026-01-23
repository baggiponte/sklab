import React from "react";
import { spring, useCurrentFrame, useVideoConfig, interpolate } from "remotion";
import { fonts } from "../styles/fonts";
import { colors } from "../styles/colors";

interface ComparisonBadgeProps {
  text: string;
  startFrame?: number;
  style?: React.CSSProperties;
}

export const ComparisonBadge: React.FC<ComparisonBadgeProps> = ({
  text,
  startFrame = 0,
  style,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const relativeFrame = frame - startFrame;

  if (relativeFrame < 0) return null;

  // Bouncy entrance
  const scale = spring({
    frame: relativeFrame,
    fps,
    config: {
      damping: 8,
      stiffness: 200,
    },
  });

  const opacity = spring({
    frame: relativeFrame,
    fps,
    config: { damping: 200 },
  });

  // Subtle pulse after entrance
  const pulseScale = relativeFrame > 30
    ? interpolate(
        Math.sin(relativeFrame * 0.1),
        [-1, 1],
        [1, 1.02]
      )
    : 1;

  return (
    <div
      style={{
        position: "relative",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        transform: `scale(${scale * pulseScale})`,
        opacity,
        ...style,
      }}
    >
      {/* Glow effect */}
      <div
        style={{
          position: "absolute",
          inset: -8,
          background: colors.successGreenGlow,
          borderRadius: 60,
          filter: "blur(16px)",
          opacity: 0.6,
        }}
      />
      <div
        style={{
          position: "relative",
          backgroundColor: colors.successGreen,
          color: colors.background,
          fontFamily: fonts.sans,
          fontWeight: 700,
          fontSize: 26,
          padding: "14px 28px",
          borderRadius: 50,
          boxShadow: `0 4px 20px ${colors.successGreenGlow}`,
          letterSpacing: "-0.01em",
        }}
      >
        {text}
      </div>
    </div>
  );
};
