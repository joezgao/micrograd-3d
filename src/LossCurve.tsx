/** Loss per step on a log scale, as one SVG polyline. */
interface Props {
  history: readonly number[];
}

const W = 320;
const H = 96;
const MAX_POINTS = 320;

export function LossCurve({ history }: Props) {
  const n = history.length;
  let d = "";
  let dSmooth = "";
  if (n > 1) {
    const logs = history.map((v) => Math.log10(Math.max(v, 1e-4)));
    // Exponential moving average: plain SGD is noisy step to step; the trend is what matters.
    const ema: number[] = [];
    logs.forEach((v, i) => ema.push(i === 0 ? v : ema[i - 1] * 0.9 + v * 0.1));
    const hi = Math.max(...logs.slice(0, Math.min(n, 5)), logs[n - 1] + 0.1);
    const lo = Math.min(...logs) - 0.05;
    const path = (vals: number[]) => {
      const stride = Math.max(1, Math.ceil(n / MAX_POINTS));
      const pts: string[] = [];
      for (let i = 0; i < n; i += stride) pts.push(pt(i, vals[i], n, lo, hi));
      pts.push(pt(n - 1, vals[n - 1], n, lo, hi));
      return "M" + pts.join("L");
    };
    d = path(logs);
    dSmooth = path(ema);
  }
  if (!d)
    return (
      <div className="loss loss-empty mono">
        <span>press Play, or → for one step</span>
      </div>
    );
  return (
    <svg
      className="loss"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={n ? `Loss curve over ${n} steps, now ${history[n - 1].toFixed(4)}` : "Loss curve, no steps yet"}
    >
      <line x1="0" y1={H - 0.5} x2={W} y2={H - 0.5} className="axis" />
      <path d={d} className="raw" vectorEffect="non-scaling-stroke" />
      <path d={dSmooth} className="curve" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function pt(i: number, v: number, n: number, lo: number, hi: number): string {
  const x = (i / Math.max(1, n - 1)) * W;
  const y = 4 + (1 - (v - lo) / Math.max(1e-6, hi - lo)) * (H - 8);
  return `${x.toFixed(1)},${Math.min(H, Math.max(0, y)).toFixed(1)}`;
}
