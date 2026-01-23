import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { fonts } from "../styles/fonts";
import { colors } from "../styles/colors";

interface TypewriterTextProps {
  text: string;
  startFrame?: number;
  framesPerChar?: number;
  fontSize?: number;
  color?: string;
  fontFamily?: string;
  fontWeight?: number;
  style?: React.CSSProperties;
  showCursor?: boolean;
  cursorChar?: string;
}

const CURSOR_BLINK_FRAMES = 16;

export const TypewriterText: React.FC<TypewriterTextProps> = ({
  text,
  startFrame = 0,
  framesPerChar = 2,
  fontSize = 48,
  color = colors.text,
  fontFamily = fonts.sans,
  fontWeight = 400,
  style,
  showCursor = true,
  cursorChar = "\u258C", // Block cursor
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const relativeFrame = frame - startFrame;

  if (relativeFrame < 0) return null;

  const charsToShow = Math.floor(relativeFrame / framesPerChar);
  const displayedText = text.slice(0, charsToShow);
  const isTyping = charsToShow < text.length;

  // Proper blinking cursor animation
  const cursorOpacity = interpolate(
    relativeFrame % CURSOR_BLINK_FRAMES,
    [0, CURSOR_BLINK_FRAMES / 2, CURSOR_BLINK_FRAMES],
    [1, 0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <span
      style={{
        fontSize,
        color,
        fontFamily,
        fontWeight,
        ...style,
      }}
    >
      {displayedText}
      {showCursor && (isTyping || relativeFrame < text.length * framesPerChar + fps * 2) && (
        <span
          style={{
            opacity: cursorOpacity,
            color,
            marginLeft: 2,
          }}
        >
          {cursorChar}
        </span>
      )}
    </span>
  );
};
