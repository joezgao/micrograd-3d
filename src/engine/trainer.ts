/**
 * The training loop, kept outside React so the UI can run it at any speed.
 * One step = forward pass on all points → loss → zeroGrad → backward → SGD update.
 */
import { MLP } from "./nn";
import { Value, sum } from "./value";
import { createRng } from "./random";
import { makeMoons, type Point } from "./data";

export const SIZES = [2, 8, 8, 1] as const;
const ALPHA = 1e-4; // L2 regularisation strength
const LR0 = 0.5;

export interface StepResult {
  loss: number;
  accuracy: number;
}

export function lossOf(model: MLP, data: readonly Point[]): { loss: Value; accuracy: number } {
  const scores = data.map((p) => model.call([p.x, p.y]));
  // Max-margin (hinge) loss: zero once a point is on the right side with margin 1.
  const losses = scores.map((s, i) => s.mul(-data[i].label).add(1).relu());
  const dataLoss = sum(losses).mul(1 / data.length);
  // L2 regularisation keeps weights small and the boundary smooth.
  const reg = sum(model.parameters().map((p) => p.mul(p))).mul(ALPHA);
  const correct = scores.filter((s, i) => s.data > 0 === data[i].label > 0).length;
  return { loss: dataLoss.add(reg), accuracy: correct / data.length };
}

export class Trainer {
  model!: MLP;
  readonly data: Point[];
  step = 0;
  history: number[] = [];
  accuracy = 0;
  /** Bumps on every change so React can subscribe cheaply. */
  version = 0;
  private listeners = new Set<() => void>();
  private readonly seed: number;

  constructor(seed = 1337) {
    this.seed = seed;
    this.data = makeMoons(100, 0.1, 7);
    this.reset();
  }

  reset(): void {
    this.model = new MLP(SIZES, createRng(this.seed));
    this.step = 0;
    this.history = [];
    // Evaluate once so the view has a loss and gradients to show before training starts.
    this.evaluate();
    this.emit();
  }

  /** Forward + backward without updating weights. */
  private evaluate(): StepResult {
    const { loss, accuracy } = lossOf(this.model, this.data);
    this.model.zeroGrad();
    loss.backward();
    this.accuracy = accuracy;
    return { loss: loss.data, accuracy };
  }

  learningRate(): number {
    return LR0 / (1 + this.step / 300);
  }

  /** One SGD step. Gradients of this step stay on the parameters for drawing. */
  trainStep(): StepResult {
    const r = this.evaluate();
    const lr = this.learningRate();
    for (const p of this.model.parameters()) p.data -= lr * p.grad;
    this.step++;
    this.history.push(r.loss);
    return r;
  }

  run(n: number): void {
    for (let i = 0; i < n; i++) this.trainStep();
    if (n > 0) this.emit();
  }


  subscribe = (fn: () => void): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  getVersion = (): number => this.version;

  private emit(): void {
    this.version++;
    for (const fn of this.listeners) fn();
  }
}
