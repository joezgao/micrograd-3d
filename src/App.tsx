import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import { Trainer } from "./engine/trainer";
import { forwardNumeric } from "./engine/fast";
import { buildWalkthrough, pointLoss } from "./engine/walkthrough";
import { DataPlot } from "./DataPlot";
import { LossCurve } from "./LossCurve";
import { WalkPanel } from "./WalkPanel";
import { useReducedMotion, useTheme } from "./theme";

// three.js + R3F are ~90 % of the JavaScript: load them after the page is up.
const NetworkView = lazy(() => import("./NetworkView").then((m) => ({ default: m.NetworkView })));

const trainer = new Trainer();
const DEFAULT_PROBE = 12; // a point on the upper (+1) moon

const lossAt = (i: number) => {
  const p = trainer.data[i];
  const acts = forwardNumeric(trainer.model, [p.x, p.y]);
  return pointLoss(acts[acts.length - 1][0], p.label);
};

export function App() {
  const version = useSyncExternalStore(trainer.subscribe, trainer.getVersion);
  const theme = useTheme();
  const reducedMotion = useReducedMotion();
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(15); // steps per second
  const [probe, setProbe] = useState(DEFAULT_PROBE);
  /** Backprop walk-through: null = normal training mode, otherwise the current step index. */
  const [walkStep, setWalkStep] = useState<number | null>(null);
  const speedRef = useRef(speed);
  speedRef.current = speed;

  const walking = walkStep !== null;
  // Training is paused while walking, so the model is frozen; rebuild only when the point changes.
  const walk = useMemo(() => (walking ? buildWalkthrough(trainer.model, trainer.data[probe]) : null), [walking, probe]);
  const walkView = useMemo(() => (walk && walkStep !== null ? { walk, step: walkStep } : null), [walk, walkStep]);

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
    setWalkStep(null);
    trainer.reset();
  }, []);

  const startWalk = useCallback(() => {
    setPlaying(false);
    // A point beyond the margin has no gradient to show; prefer the point with the largest loss.
    if (lossAt(probe) === 0) {
      let best = -1;
      let bestLoss = 0;
      trainer.data.forEach((_, i) => {
        const l = lossAt(i);
        if (l > bestLoss) [best, bestLoss] = [i, l];
      });
      if (best >= 0) setProbe(best);
    }
    setWalkStep(0);
  }, [probe]);
  const nSteps = walk?.steps.length ?? 0;
  const next = useCallback(() => setWalkStep((s) => (s === null ? s : Math.min(nSteps - 1, s + 1))), [nSteps]);
  const prev = useCallback(() => setWalkStep((s) => (s === null ? s : Math.max(0, s - 1))), []);
  const exitWalk = useCallback(() => setWalkStep(null), []);
  const onProbe = useCallback(
    (i: number) => {
      setProbe(i);
      if (walking) setWalkStep(0);
    },
    [walking],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target instanceof HTMLElement ? e.target : null;
      if (walking) {
        // ← → Esc work even after clicking Back / Next; only the slider and the data plot
        // (which moves its probe with the arrows) keep their own keys.
        if (target?.closest("input, [role=slider]")) return;
        if (e.key === "ArrowRight") next();
        else if (e.key === "ArrowLeft") prev();
        else if (e.key === "Escape") exitWalk();
        return;
      }
      // Leave keys alone on focused controls: Space should press the focused button.
      if (target?.closest("button, input, [tabindex]")) return;
      if (e.code === "Space") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key === "ArrowRight") step();
      else if (e.key === "r" || e.key === "R") reset();
      else if (e.key === "b" || e.key === "B") startWalk();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [walking, next, prev, exitWalk, step, reset, startWalk]);

  // When the walk ends (Exit, Done, Esc) its panel unmounts; put focus back on the button that opened it.
  const walkOpenRef = useRef<HTMLButtonElement>(null);
  const wasWalking = useRef(false);
  useEffect(() => {
    if (wasWalking.current && !walking) walkOpenRef.current?.focus();
    wasWalking.current = walking;
  }, [walking]);

  const loss = trainer.history.at(-1);

  return (
    <main className="app">
      <header className="head">
        <h1>micrograd 3D</h1>
        <p className="lede">
          A tiny neural network learning two moons. Every line is a weight; the brighter it glows, the larger the
          gradient backprop just sent through it.
        </p>
      </header>
      <section className="stage" aria-label="Network">
        <Suspense fallback={<p className="stage-note mono">preparing the 3D view…</p>}>
          <NetworkView trainer={trainer} probe={probe} theme={theme} reducedMotion={reducedMotion} walk={walkView} />
        </Suspense>
        <div className={walking ? "legend mono walking" : "legend mono"} aria-hidden="true">
          <span>{walking ? "this step's ∂L/∂w" : "|∂L/∂w|"}</span>
          <span className="ramp" />
          {walking && (
            <>
              <span>computed</span>
              <span className="done-line" />
            </>
          )}
          <span className="hint">drag to orbit</span>
        </div>
      </section>

      <aside className="panel">
        {walk && walkStep !== null ? (
          <WalkPanel walk={walk} step={walkStep} onPrev={prev} onNext={next} onExit={exitWalk} />
        ) : (
          <>
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
            <button ref={walkOpenRef} className="walk-open" onClick={startWalk} aria-keyshortcuts="B">
              Step through backprop
            </button>
          </>
        )}

        {/* One idea at a time: the training stats come back when the walk ends. */}
        {!walking && (
          <>
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
          </>
        )}

        <figure>
          <figcaption className="mono">
            {walking ? "data · the point being traced · click another" : "data · decision boundary · click a point to probe"}
          </figcaption>
          <DataPlot trainer={trainer} version={version} probe={probe} onProbe={onProbe} theme={theme} />
        </figure>

        <footer className="mono">
          2 → 8 → 8 → 1 · tanh · hinge loss · SGD
          <br />
          engine: micrograd in TypeScript, after Karpathy
          <br />
          built with Claude for Joe Gao
        </footer>
      </aside>
    </main>
  );
}
