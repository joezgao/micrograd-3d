import { createRng } from "./random";

export interface Point {
  x: number;
  y: number;
  label: 1 | -1;
}

/**
 * Two interleaving half circles ("moons"), like sklearn.datasets.make_moons.
 * Points are shifted and scaled to sit roughly inside [-1.5, 1.5]² around the origin.
 */
export function makeMoons(n = 100, noise = 0.1, seed = 7): Point[] {
  const rng = createRng(seed);
  const half = Math.floor(n / 2);
  const pts: Point[] = [];
  for (let i = 0; i < n; i++) {
    const upper = i < half;
    const k = upper ? i : i - half;
    const m = upper ? half : n - half;
    const t = (Math.PI * k) / Math.max(1, m - 1);
    let x = upper ? Math.cos(t) : 1 - Math.cos(t);
    let y = upper ? Math.sin(t) : 0.5 - Math.sin(t);
    x += noise * rng.normal();
    y += noise * rng.normal();
    // Centre on the origin and scale so tanh units start in their useful range.
    pts.push({ x: (x - 0.5) * 1.2, y: (y - 0.25) * 1.2, label: upper ? 1 : -1 });
  }
  return pts;
}
