/**
 * Forward pass on plain numbers (no Value graph), reading the model's current weights.
 * Used only for drawing — the decision field needs thousands of evaluations per frame.
 * Training always goes through Value so backward() can run.
 */
import type { MLP } from "./nn";

export function forwardNumeric(model: MLP, x: readonly number[]): number[][] {
  let cur = x.slice();
  const acts: number[][] = [cur];
  for (const layer of model.layers) {
    const next = new Array<number>(layer.neurons.length);
    for (let j = 0; j < layer.neurons.length; j++) {
      const n = layer.neurons[j];
      let s = n.b.data;
      for (let i = 0; i < n.w.length; i++) s += n.w[i].data * cur[i];
      next[j] = n.nonlin ? Math.tanh(s) : s;
    }
    cur = next;
    acts.push(cur);
  }
  return acts;
}

export function scoreNumeric(model: MLP, x: number, y: number): number {
  const acts = forwardNumeric(model, [x, y]);
  return acts[acts.length - 1][0];
}
