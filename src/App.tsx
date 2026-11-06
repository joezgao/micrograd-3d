import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { Trainer } from "./engine/trainer";
import { NetworkView } from "./NetworkView";
import { DataPlot } from "./DataPlot";
import { LossCurve } from "./LossCurve";
import { useReducedMotion, useTheme } from "./theme";

const trainer = new Trainer();
const DEFAULT_PROBE = 12; // a point on the upper (+1) moon

export function App() {
  const version = useSyncExternalStore(trainer.subscribe, trainer.getVersion);
  const theme = useTheme();
  const reducedMotion = useReducedMotion();
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(15); // steps per second
  const [probe, setProbe] = useState(DEFAULT_PROBE);
  const speedRef = useRef(speed);
  speedRef.current = speed;

  // Training clock: accumulate fractional steps so any speed works at any frame rate.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const tick = (now: number) => {
      acc += ((now - last) / 1000) * speedRef.current;
      last = now;
      const n = Math.min(4, Math.floor(acc));
      acc -= n;
      if (n > 0) trainer.run(n);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const step = useCallback(() => {
    setPlaying(false);
    trainer.run(1);
  }, []);
  const reset = useCallback(() => {
    setPlaying(false);
    trainer.reset();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Leave keys alone on focused controls: Space should press the focused button.
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.target instanceof HTMLElement && e.target.closest("button, input, [tabindex]")) return;
      if (e.code === "Space") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key === "ArrowRight") step();
      else if (e.key === "r" || e.key === "R") reset();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, reset]);

  const loss = trainer.history.at(-1);

  return (
    <div className="app">
      <header className="head">
        <h1>micrograd 3D</h1>
        <p className="lede">
          A tiny neural network learning two moons. Every line is a weight; the brighter it glows, the larger the
          gradient backprop just sent through it.
        </p>
      </header>
      <section className="stage" aria-label="Network">
        <NetworkView trainer={trainer} probe={probe} theme={theme} reducedMotion={reducedMotion} />
        <div className="legend mono" aria-hidden="true">
          <span>|∂L/∂w|</span>
          <span className="ramp" />
          <span className="hint">drag to orbit</span>
        </div>
      </section>

      <aside className="panel">

        <div className="controls" role="group" aria-label="Training controls">
          <button className="primary" onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
            {playing ? "Pause" : "Play"}
          </button>
          <button onClick={step} aria-label="Step one training step">
            Step
          </button>
          <button onClick={reset} aria-label="Reset weights">
            Reset
          </button>
        </div>
        <label className="speed mono">
          <span>speed</span>
          <input
            type="range"
            min={1}
            max={60}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            aria-valuetext={`${speed} steps per second`}
          />
          <span className="val">{speed}/s</span>
        </label>

        <dl className="stats mono">
          <div>
            <dt>step</dt>
            <dd>{trainer.step}</dd>
          </div>
          <div>
            <dt>loss</dt>
            <dd>{loss === undefined ? "—" : loss.toFixed(4)}</dd>
          </div>
          <div>
            <dt>accuracy</dt>
            <dd>{trainer.step ? `${Math.round(trainer.accuracy * 100)}%` : "—"}</dd>
          </div>
        </dl>

        <figure>
          <figcaption className="mono">loss · log scale</figcaption>
          <LossCurve history={trainer.history} />
        </figure>

        <figure>
          <figcaption className="mono">data · decision boundary · click a point to probe</figcaption>
          <DataPlot trainer={trainer} version={version} probe={probe} onProbe={setProbe} theme={theme} />
        </figure>

        <footer className="mono">
          2 → 8 → 8 → 1 · tanh · hinge loss · SGD
          <br />
          engine: micrograd in TypeScript, after Karpathy
        </footer>
      </aside>
    </div>
  );
}
