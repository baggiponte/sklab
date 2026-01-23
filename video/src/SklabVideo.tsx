import React from "react";
import { AbsoluteFill } from "remotion";
import {
  TransitionSeries,
  linearTiming,
  springTiming,
} from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";
import { OpeningHook } from "./scenes/OpeningHook";
import { ProblemStatement } from "./scenes/ProblemStatement";
import { IntroducingSklab } from "./scenes/IntroducingSklab";
import { ComparisonIntro } from "./scenes/ComparisonIntro";
import {
  CrossValidationComparison,
  GridSearchComparison,
  OptunaSearchComparison,
} from "./scenes/CodeComparison";
import { ThePattern } from "./scenes/ThePattern";
import { ClosingCTA } from "./scenes/ClosingCTA";
import { colors } from "./styles/colors";

// Frame calculations at 30 FPS
const FPS = 30;
const seconds = (s: number) => s * FPS;

// Scene durations (adjusted for transitions)
const OPENING_HOOK = seconds(4);
const PROBLEM_STATEMENT = seconds(4);
const INTRODUCING_SKLAB = seconds(4);
const COMPARISON_INTRO = seconds(2);
const CROSS_VALIDATION = seconds(12);
const GRID_SEARCH = seconds(12);
const OPTUNA_SEARCH = seconds(14);
const THE_PATTERN = seconds(6);
const CLOSING_CTA = seconds(7);

// Transition timing
const FADE_TIMING = linearTiming({ durationInFrames: 20 });
const SLIDE_TIMING = springTiming({ config: { damping: 200 }, durationInFrames: 25 });

export const SklabVideo: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: colors.background }}>
      <TransitionSeries>
        {/* Opening Hook */}
        <TransitionSeries.Sequence durationInFrames={OPENING_HOOK}>
          <OpeningHook />
        </TransitionSeries.Sequence>

        <TransitionSeries.Transition
          presentation={fade()}
          timing={FADE_TIMING}
        />

        {/* Problem Statement */}
        <TransitionSeries.Sequence durationInFrames={PROBLEM_STATEMENT}>
          <ProblemStatement />
        </TransitionSeries.Sequence>

        <TransitionSeries.Transition
          presentation={fade()}
          timing={FADE_TIMING}
        />

        {/* Introducing sklab */}
        <TransitionSeries.Sequence durationInFrames={INTRODUCING_SKLAB}>
          <IntroducingSklab />
        </TransitionSeries.Sequence>

        <TransitionSeries.Transition
          presentation={fade()}
          timing={FADE_TIMING}
        />

        {/* Comparison Intro */}
        <TransitionSeries.Sequence durationInFrames={COMPARISON_INTRO}>
          <ComparisonIntro />
        </TransitionSeries.Sequence>

        <TransitionSeries.Transition
          presentation={slide({ direction: "from-right" })}
          timing={SLIDE_TIMING}
        />

        {/* Cross-Validation Comparison */}
        <TransitionSeries.Sequence durationInFrames={CROSS_VALIDATION}>
          <CrossValidationComparison />
        </TransitionSeries.Sequence>

        <TransitionSeries.Transition
          presentation={slide({ direction: "from-right" })}
          timing={SLIDE_TIMING}
        />

        {/* Grid Search Comparison */}
        <TransitionSeries.Sequence durationInFrames={GRID_SEARCH}>
          <GridSearchComparison />
        </TransitionSeries.Sequence>

        <TransitionSeries.Transition
          presentation={slide({ direction: "from-right" })}
          timing={SLIDE_TIMING}
        />

        {/* Optuna Search Comparison */}
        <TransitionSeries.Sequence durationInFrames={OPTUNA_SEARCH}>
          <OptunaSearchComparison />
        </TransitionSeries.Sequence>

        <TransitionSeries.Transition
          presentation={fade()}
          timing={FADE_TIMING}
        />

        {/* The Pattern */}
        <TransitionSeries.Sequence durationInFrames={THE_PATTERN}>
          <ThePattern />
        </TransitionSeries.Sequence>

        <TransitionSeries.Transition
          presentation={fade()}
          timing={FADE_TIMING}
        />

        {/* Closing CTA */}
        <TransitionSeries.Sequence durationInFrames={CLOSING_CTA}>
          <ClosingCTA />
        </TransitionSeries.Sequence>
      </TransitionSeries>
    </AbsoluteFill>
  );
};
