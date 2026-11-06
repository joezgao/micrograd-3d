/** Caption card for the backprop walk-through: where we are, the numbers, back / next / done. */
import { useEffect, useRef } from "react";
import type { Walkthrough } from "./engine/walkthrough";

interface Props {
  walk: Walkthrough;
  step: number;
  onPrev: () => void;
  onNext: () => void;
  onExit: () => void;
}

/** "expression = 0.1234" → ["expression", "0.1234"], so the number sits in its own column and never wraps. */
function splitValue(line: string): [string, string | null] {
  const k = line.lastIndexOf(" = ");
  if (k < 0) return [line, null];
  const value = line.slice(k + 3);
  return /^[−-]?\d/.test(value) ? [line.slice(0, k) + " =", value] : [line, null];
}

export function WalkPanel({ walk, step, onPrev, onNext, onExit }: Props) {
  const s = walk.steps[step];
  const n = walk.steps.length;
  const first = step === 0;
  const last = step === n - 1;
  const nextRef = useRef<HTMLButtonElement>(null);
  // Entering the mode puts keyboard users straight on "Next".
  useEffect(() => nextRef.current?.focus(), []);

  return (
    <section className="walk" aria-label="Step through backprop">
      <div className="controls" role="group" aria-label="Backprop steps">
        {/* aria-disabled, not disabled: a disabled button would drop keyboard focus to the page. */}
        <button onClick={first ? undefined : onPrev} aria-disabled={first} aria-label="Previous backprop step">
          ← Back
        </button>
        <button
          ref={nextRef}
          className="primary"
          onClick={last ? onExit : onNext}
          aria-label={last ? "Finish" : "Next backprop step"}
        >
          {last ? "Done" : "Next →"}
        </button>
        <button onClick={onExit} aria-label="Leave backprop mode">
          Exit
        </button>
      </div>
      <div className="walk-progress" aria-hidden="true">
        <span style={{ width: `${((step + 1) / n) * 100}%` }} />
      </div>
      <div className="walk-card" aria-live="polite">
        <p className="walk-count mono">
          backward pass · {step + 1} / {n}
        </p>
        <h2>{s.title}</h2>
        <ul className="mono">
          {s.lines.map((line) => {
            const [expr, value] = splitValue(line);
            return (
              <li key={line}>
                <span>{expr}</span>
                {value !== null && <span className="walk-val">{value}</span>}
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
