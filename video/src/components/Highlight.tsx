import React from "react";
import { spring, useCurrentFrame, useVideoConfig } from "remotion";
import { colors } from "../styles/colors";

interface HighlightProps {
  children: React.ReactNode;
  color?: string;
  delay?: number;
  durationInFrames?: number;
  style?: React.CSSProperties;
}

/**
 * Animated highlight effect that wipes across text like a highlighter pen.
 */
export const Highlight: React.FC<HighlightProps> = ({
  children,
  color = colors.accentBlueGlow,
  delay = 0,
  durationInFrames = 18,
  style,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const highlightProgress = spring({
    fps,
    frame,
    config: { damping: 200 },
    delay,
    durationInFrames,
  });

  const scaleX = Math.max(0, Math.min(1, highlightProgress));

  return (
    <span style={{ position: "relative", display: "inline-block", ...style }}>
      <span
        style={{
          position: "absolute",
          left: -4,
          right: -4,
          top: "50%",
          height: "1.1em",
          transform: `translateY(-50%) scaleX(${scaleX})`,
          transformOrigin: "left center",
          backgroundColor: color,
          borderRadius: "0.15em",
          zIndex: 0,
        }}
      />
      <span style={{ position: "relative", zIndex: 1 }}>{children}</span>
    </span>
  );
};
