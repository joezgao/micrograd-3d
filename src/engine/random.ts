/** Seeded random numbers, so every run (and every screenshot) is reproducible. */
export interface Rng {
  next(): number; // uniform in [0, 1)
  uniform(lo: number, hi: number): number;
  normal(): number; // standard normal
}

/** mulberry32: a tiny, fast, good-enough 32-bit generator. */
export function createRng(seed: number): Rng {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    next,
    uniform: (lo, hi) => lo + (hi - lo) * next(),
    normal: () => {
      // Box–Muller transform.
      const u = 1 - next();
      const v = next();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    },
  };
}
