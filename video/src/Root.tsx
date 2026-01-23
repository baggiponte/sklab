import React from "react";
import { Composition } from "remotion";
import { SklabVideo } from "./SklabVideo";

// Total duration: 4+4+4+2+12+12+14+6+7 = 65 seconds at 30 FPS
const FPS = 30;
const DURATION_SECONDS = 65;
const TOTAL_FRAMES = DURATION_SECONDS * FPS;

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="SklabPromo"
        component={SklabVideo}
        durationInFrames={TOTAL_FRAMES}
        fps={FPS}
        width={1920}
        height={1080}
      />
    </>
  );
};
