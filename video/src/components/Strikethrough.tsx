import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { colors } from "../styles/colors";

interface StrikethroughProps {
  startFrame?: number;
  duration?: number;
  width: number | string;
  style?: React.CSSProperties;
}

export const Strikethrough: React.FC<StrikethroughProps> = ({
  startFrame = 0,
  duration = 15,
  width,
  style,
}) => {
  const frame = useCurrentFrame();
  const relativeFrame = frame - startFrame;

  if (relativeFrame < 0) return null;

  const progress = interpolate(
    relativeFrame,
    [0, duration],
    [0, 100],
    { extrapolateRight: "clamp" }
  );

  return (
    <div
      style={{
        position: "absolute",
        top: "50%",
        left: 0,
        height: 4,
        width: typeof width === "number" ? `${width}px` : width,
        background: `linear-gradient(90deg, ${colors.warningRed} ${progress}%, transparent ${progress}%)`,
        transform: "translateY(-50%)",
        borderRadius: 2,
        ...style,
      }}
    />
  );
};
