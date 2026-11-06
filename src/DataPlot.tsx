/**
 * The moons data and what the model currently believes.
 * Background: a soft monochrome field, lighter where the model says +1.
 * Thin line: the decision boundary (score = 0), traced with marching squares. It is drawn in
 * the text colour on purpose: the accent is reserved for gradients in the 3D view.
 * Points: filled = class +1, ring = class −1 (shape, not just colour). Click one to probe it.
 */
import { useEffect, useRef } from "react";
import type { Trainer } from "./engine/trainer";
import { scoreNumeric } from "./engine/fast";
import type { Theme } from "./theme";

// The moons span about x ∈ [-1.8, 1.8], y ∈ [-0.9, 0.9]; leave a margin around them.
export const EX = 2.2;
export const EY = 1.3;
const GX = 66;
const GY = Math.round((GX * EY) / EX);

interface Props {
  trainer: Trainer;
  version: number;
  probe: number;
  onProbe: (i: number) => void;
  theme: Theme;
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function DataPlot({ trainer, version, probe, onProbe, theme }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const field = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const W = Math.round(canvas.clientWidth * dpr);
    const H = Math.round((W * EY) / EX);
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }
    const toX = (v: number) => ((v + EX) / (2 * EX)) * W;
    const toY = (v: number) => ((EY - v) / (2 * EY)) * H;
    const model = trainer.model;

    // 1. Score grid.
    const s = new Float32Array((GX + 1) * (GY + 1));
    for (let j = 0; j <= GY; j++)
      for (let i = 0; i <= GX; i++) {
        const x = -EX + (2 * EX * i) / GX;
        const y = EY - (2 * EY * j) / GY;
        s[j * (GX + 1) + i] = scoreNumeric(model, x, y);
      }

    // 2. Soft field: draw a tiny image and let the browser smooth it up.
    const f = (field.current ??= document.createElement("canvas"));
    f.width = GX + 1;
    f.height = GY + 1;
    const fctx = f.getContext("2d");
    if (!fctx) return;
    const img = fctx.createImageData(GX + 1, GY + 1);
    const bg = hexToRgb(theme.bg);
    const hi = hexToRgb(theme.dark ? "#2A2A2A" : "#FFFFFF");
    const lo = hexToRgb(theme.dark ? "#0E0E0E" : "#E2E2E0");
    for (let k = 0; k < s.length; k++) {
      const t = Math.tanh(s[k] * 1.5) * 0.5 + 0.5; // 0 = surely −1, 1 = surely +1
      for (let c = 0; c < 3; c++) img.data[k * 4 + c] = lo[c] + (hi[c] - lo[c]) * t;
      img.data[k * 4 + 3] = 255;
    }
    fctx.putImageData(img, 0, 0);
    ctx.fillStyle = `rgb(${bg.join(",")})`;
    ctx.fillRect(0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(f, 0, 0, W, H);

    // 3. Decision boundary (marching squares on score = 0).
    const cw = W / GX;
    const ch = H / GY;
    const at = (i: number, j: number) => s[j * (GX + 1) + i];
    ctx.strokeStyle = theme.text;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1.5 * dpr;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let j = 0; j < GY; j++)
      for (let i = 0; i < GX; i++) {
        const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1);
        const pts: [number, number][] = [];
        const cross = (v0: number, v1: number, x0: number, y0: number, x1: number, y1: number) => {
          if (v0 > 0 !== v1 > 0) {
            const t = v0 / (v0 - v1);
            pts.push([(x0 + (x1 - x0) * t) * cw, (y0 + (y1 - y0) * t) * ch]);
          }
        };
        cross(a, b, i, j, i + 1, j);
        cross(b, c, i + 1, j, i + 1, j + 1);
        cross(c, d, i + 1, j + 1, i, j + 1);
        cross(d, a, i, j + 1, i, j);
        for (let k = 0; k + 1 < pts.length; k += 2) {
          ctx.moveTo(pts[k][0], pts[k][1]);
          ctx.lineTo(pts[k + 1][0], pts[k + 1][1]);
        }
      }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // 4. Points.
    const r = 3.2 * dpr;
    trainer.data.forEach((p, i) => {
      const x = toX(p.x);
      const y = toY(p.y);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      if (p.label > 0) {
        ctx.fillStyle = theme.text;
        ctx.fill();
      } else {
        ctx.strokeStyle = theme.muted;
        ctx.lineWidth = 1.4 * dpr;
        ctx.stroke();
      }
      if (i === probe) {
        ctx.beginPath();
        ctx.arc(x, y, r * 2.6, 0, Math.PI * 2);
        ctx.strokeStyle = theme.text;
        ctx.lineWidth = 1 * dpr;
        ctx.stroke();
      }
    });
  }, [trainer, version, probe, theme]);

  const pick = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 * EX - EX;
    const y = EY - ((e.clientY - rect.top) / rect.height) * 2 * EY;
    let best = 0;
    let bestD = Infinity;
    trainer.data.forEach((p, i) => {
      const d = (p.x - x) ** 2 + (p.y - y) ** 2;
      if (d < bestD) [best, bestD] = [i, d];
    });
    onProbe(best);
  };

  // Keyboard: ←/→ move the probe through the points.
  const onKey = (e: React.KeyboardEvent<HTMLCanvasElement>) => {
    const n = trainer.data.length;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") onProbe((probe + 1) % n);
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") onProbe((probe - 1 + n) % n);
    else return;
    e.preventDefault();
    e.stopPropagation();
  };
  const pt = trainer.data[probe];

  return (
    <canvas
      ref={ref}
      className="plot"
      onPointerDown={pick}
      onKeyDown={onKey}
      tabIndex={0}
      role="slider"
      aria-label="Probe point: moons dataset with the model's decision boundary. Click a point, or use arrow keys."
      aria-valuemin={1}
      aria-valuemax={trainer.data.length}
      aria-valuenow={probe + 1}
      aria-valuetext={`point ${probe + 1} of ${trainer.data.length}, class ${pt.label > 0 ? "+1" : "−1"}`}
    />
  );
}
