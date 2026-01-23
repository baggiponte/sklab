import React from "react";
import {
  interpolate,
  useCurrentFrame,
  spring,
  useVideoConfig,
} from "remotion";
import { fonts, typography } from "../styles/fonts";
import { colors } from "../styles/colors";

interface CodeBlockProps {
  code: string;
  startFrame?: number;
  framesPerChar?: number;
  fontSize?: number;
  style?: React.CSSProperties;
  typewriter?: boolean;
  showLineNumbers?: boolean;
  glowColor?: string;
}

// Enhanced syntax highlighting for Python
const highlightPython = (code: string): React.ReactNode[] => {
  const keywords = [
    "from", "import", "def", "return", "if", "else", "for", "in",
    "with", "as", "class", "try", "except", "raise", "and", "or",
    "not", "True", "False", "None", "lambda", "yield", "async", "await"
  ];
  const builtins = [
    "print", "range", "len", "str", "int", "float", "list", "dict",
    "set", "tuple", "max", "min", "sum", "abs", "round", "sorted"
  ];

  const lines = code.split("\n");

  return lines.map((line, lineIdx) => {
    const parts: React.ReactNode[] = [];
    let remaining = line;
    let keyIdx = 0;

    while (remaining.length > 0) {
      // Check for comments
      const commentMatch = remaining.match(/^(#.*)$/);
      if (commentMatch) {
        parts.push(
          <span key={`${lineIdx}-${keyIdx++}`} style={{ color: colors.textDim, fontStyle: "italic" }}>
            {commentMatch[1]}
          </span>
        );
        remaining = "";
        continue;
      }

      // Check for strings (including f-strings and triple quotes)
      const stringMatch = remaining.match(/^(f?["'](?:[^"'\\]|\\.)*["']|f?"""[\s\S]*?"""|f?'''[\s\S]*?''')/);
      if (stringMatch) {
        parts.push(
          <span key={`${lineIdx}-${keyIdx++}`} style={{ color: colors.codeGreen }}>
            {stringMatch[1]}
          </span>
        );
        remaining = remaining.slice(stringMatch[1].length);
        continue;
      }

      // Check for numbers (including scientific notation)
      const numberMatch = remaining.match(/^(\d+\.?\d*(?:e[+-]?\d+)?)/i);
      if (numberMatch) {
        parts.push(
          <span key={`${lineIdx}-${keyIdx++}`} style={{ color: colors.codeOrange }}>
            {numberMatch[1]}
          </span>
        );
        remaining = remaining.slice(numberMatch[1].length);
        continue;
      }

      // Check for function calls
      const funcMatch = remaining.match(/^(\w+)(\()/);
      if (funcMatch) {
        const word = funcMatch[1];
        const isBuiltin = builtins.includes(word);
        const isKeyword = keywords.includes(word);
        parts.push(
          <span
            key={`${lineIdx}-${keyIdx++}`}
            style={{
              color: isKeyword ? colors.codePurple : isBuiltin ? colors.codeCyan : colors.codeYellow
            }}
          >
            {word}
          </span>
        );
        parts.push(
          <span key={`${lineIdx}-${keyIdx++}`} style={{ color: colors.text }}>
            (
          </span>
        );
        remaining = remaining.slice(word.length + 1);
        continue;
      }

      // Check for keywords and identifiers
      const keywordMatch = remaining.match(/^(\w+)/);
      if (keywordMatch) {
        const word = keywordMatch[1];
        if (keywords.includes(word)) {
          parts.push(
            <span key={`${lineIdx}-${keyIdx++}`} style={{ color: colors.codePurple, fontWeight: 500 }}>
              {word}
            </span>
          );
        } else if (builtins.includes(word)) {
          parts.push(
            <span key={`${lineIdx}-${keyIdx++}`} style={{ color: colors.codeCyan }}>
              {word}
            </span>
          );
        } else {
          parts.push(
            <span key={`${lineIdx}-${keyIdx++}`} style={{ color: colors.text }}>
              {word}
            </span>
          );
        }
        remaining = remaining.slice(word.length);
        continue;
      }

      // Operators and punctuation
      const opMatch = remaining.match(/^([=<>!+\-*/%&|^~]+|[.,;:()[\]{}])/);
      if (opMatch) {
        parts.push(
          <span key={`${lineIdx}-${keyIdx++}`} style={{ color: colors.textMuted }}>
            {opMatch[1]}
          </span>
        );
        remaining = remaining.slice(opMatch[1].length);
        continue;
      }

      // Default: single character (whitespace, etc.)
      parts.push(
        <span key={`${lineIdx}-${keyIdx++}`} style={{ color: colors.text }}>
          {remaining[0]}
        </span>
      );
      remaining = remaining.slice(1);
    }

    return (
      <div key={lineIdx} style={{ minHeight: "1.6em" }}>
        {parts.length > 0 ? parts : "\u00A0"}
      </div>
    );
  });
};

export const CodeBlock: React.FC<CodeBlockProps> = ({
  code,
  startFrame = 0,
  framesPerChar = 1,
  fontSize = typography.code.fontSize,
  style,
  typewriter = true,
  showLineNumbers = false,
  glowColor,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const relativeFrame = frame - startFrame;

  if (relativeFrame < 0) return null;

  const charsToShow = typewriter
    ? Math.floor(relativeFrame / framesPerChar)
    : code.length;
  const displayedCode = code.slice(0, charsToShow);

  const opacity = spring({
    frame: relativeFrame,
    fps,
    config: { damping: 200 },
  });

  const scale = spring({
    frame: relativeFrame,
    fps,
    config: { damping: 200 },
    from: 0.98,
    to: 1,
  });

  return (
    <div
      style={{
        position: "relative",
        transform: `scale(${scale})`,
        opacity,
      }}
    >
      {/* Glow effect */}
      {glowColor && (
        <div
          style={{
            position: "absolute",
            inset: -20,
            background: `radial-gradient(ellipse at center, ${glowColor} 0%, transparent 70%)`,
            opacity: 0.3,
            filter: "blur(20px)",
            zIndex: -1,
          }}
        />
      )}
      <div
        style={{
          backgroundColor: colors.codeBackground,
          borderRadius: 16,
          padding: "28px 32px",
          fontFamily: fonts.mono,
          fontSize,
          lineHeight: 1.6,
          overflow: "hidden",
          border: `1px solid ${colors.backgroundAlt}`,
          boxShadow: "0 4px 24px rgba(0, 0, 0, 0.4)",
          ...style,
        }}
      >
        <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>
          {highlightPython(displayedCode)}
        </pre>
      </div>
    </div>
  );
};
