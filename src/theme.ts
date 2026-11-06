import { useSyncExternalStore } from "react";

/** DESIGN.md tokens. Light = inverted monochrome; the one accent darkens for contrast. */
export interface Theme {
  dark: boolean;
  bg: string;
  surface: string;
  hairline: string;
  text: string;
  muted: string;
  accent: string;
}

export const DARK: Theme = {
  dark: true,
  bg: "#0A0A0A",
  surface: "#141414",
  hairline: "#262626",
  text: "#EDEDED",
  muted: "#8A8A8A",
  accent: "#F5B46A",
};

export const LIGHT: Theme = {
  dark: false,
  bg: "#F5F5F4",
  surface: "#EBEBEA",
  hairline: "#D6D6D4",
  text: "#121212",
  muted: "#666666",
  accent: "#B4661A",
};

const darkQuery = () => window.matchMedia("(prefers-color-scheme: dark)");
const motionQuery = () => window.matchMedia("(prefers-reduced-motion: reduce)");

function useMedia(query: () => MediaQueryList, fallback: boolean): boolean {
  return useSyncExternalStore(
    (cb) => {
      const q = query();
      q.addEventListener("change", cb);
      return () => q.removeEventListener("change", cb);
    },
    () => query().matches,
    () => fallback,
  );
}

export function useTheme(): Theme {
  return useMedia(darkQuery, true) ? DARK : LIGHT;
}

export function useReducedMotion(): boolean {
  return useMedia(motionQuery, false);
}
