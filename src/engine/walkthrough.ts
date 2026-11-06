/**
 * "Step through backprop": one point's backward pass, replayed neuron by neuron.
 *
 * Training uses the mean loss over 100 points, whose graph has ~20k nodes — too many to
 * read. Here we take a single point, build its own small graph with the same Value engine,
 * call the same backward(), and then read the gradients off the graph in the order backward()
 * visits them: loss → output neuron → hidden layer 2 → hidden layer 1 → inputs.
 *
 * For every neuron j the chain rule is the same three lines:
 *   dL/da_j = Σ_k w_jk · δ_k        (gradient arriving from the next layer)
 *   δ_j     = dL/da_j · f'(z_j)     (times the neuron's local gradient; tanh' = 1 − a²)
 *   dL/dw_ij = a_i · δ_j            (each incoming weight: input value × δ)
 */
import type { MLP } from "./nn";
import type { Point } from "./data";
import type { Value } from "./value";

export interface WalkEdge {
  layer: number; // index into model.layers (edge goes from sizes[layer] to sizes[layer+1])
  from: number;
  to: number;
  input: number; // a_i, the value flowing into the weight
  delta: number; // δ_j of the receiving neuron
  grad: number; // dL/dw = a_i · δ_j, as backward() computed it
}

export interface WalkStep {
  kind: "loss" | "neuron";
  /** Layer in the drawing (0 = inputs … sizes.length − 1 = output); −1 for the loss step. */
  layer: number;
  index: number;
  title: string;
  lines: string[];
  /** Edges whose gradient this step computes (incoming weights of the neuron). */
  edges: WalkEdge[];
}

export interface Walkthrough {
  point: Point;
  score: number;
  loss: number;
  /** false when the point is already beyond the margin: loss 0, every gradient 0. */
  flows: boolean;
  steps: WalkStep[];
}

const f = (x: number) => (Math.abs(x) < 5e-5 && x !== 0 ? x.toExponential(1) : x.toFixed(4));

export function buildWalkthrough(model: MLP, point: Point): Walkthrough {
  const params = model.parameters();
  // backward() adds into parameter grads; keep the training step's grads intact for the view.
  const saved = params.map((p) => p.grad);
  for (const p of params) p.grad = 0;

  const acts = model.forward([point.x, point.y]);
  const out = acts[acts.length - 1][0];
  const loss = out.mul(-point.label).add(1).relu();
  loss.backward();

  // For tanh neurons the pre-activation z is the tanh node's only parent.
  const pre = (v: Value): Value => (v.op === "tanh" ? v._prev[0] : v);
  const sizes = model.sizes;
  const L = sizes.length - 1;
  const y = point.label;
  const steps: WalkStep[] = [];

  steps.push({
    kind: "loss",
    layer: -1,
    index: 0,
    title: "Start at the loss",
    lines: [
      `this point: (${point.x.toFixed(2)}, ${point.y.toFixed(2)}), label y = ${y > 0 ? "+1" : "−1"}`,
      `score s = ${f(out.data)}`,
      `L = max(0, 1 − y·s) = ${f(loss.data)}`,
      loss.data > 0
        ? "backward() begins here: dL/dL = 1"
        : "y·s ≥ 1: already beyond the margin, so L = 0 and every gradient below is 0",
    ],
    edges: [],
  });

  for (let l = L; l >= 0; l--) {
    for (let j = 0; j < sizes[l]; j++) {
      const a = acts[l][j];
      const edges: WalkEdge[] = [];
      const lines: string[] = [];
      let title: string;
      if (l === L) {
        title = "Output neuron (linear)";
        lines.push(`dL/ds = ${loss.data > 0 ? `−y = ${f(-y)}` : "0 (relu blocked it)"}`);
        lines.push("local gradient of a linear neuron = 1, so δ = dL/ds");
        lines.push(`δ = ${f(pre(a).grad)}`);
      } else if (l === 0) {
        title = `Input x${j === 0 ? "₁" : "₂"}`;
        lines.push(`dL/dx = Σ w·δ over its ${sizes[1]} outgoing weights = ${f(a.grad)}`);
        lines.push("inputs are data, not parameters: nothing to update, backprop stops here");
      } else {
        const z = pre(a);
        title = `Hidden layer ${l}, neuron ${j + 1}`;
        lines.push(`dL/da = Σ w·δ (next layer) = ${f(a.grad)}`);
        // Bracket a negative a, so "1 − (−0.05)²" can't be read as 1 − (−(0.05²)).
        const aText = a.data < 0 ? `(${f(a.data)})` : f(a.data);
        lines.push(`tanh′(z) = 1 − a² = 1 − ${aText}² = ${f(1 - a.data * a.data)}`);
        lines.push(`δ = dL/da · tanh′(z) = ${f(z.grad)}`);
      }
      if (l > 0) {
        const neuron = model.layers[l - 1].neurons[j];
        const delta = pre(a).grad;
        neuron.w.forEach((w, i) =>
          edges.push({ layer: l - 1, from: i, to: j, input: acts[l - 1][i].data, delta, grad: w.grad }),
        );
        const top = edges.reduce((m, e) => (Math.abs(e.grad) > Math.abs(m.grad) ? e : m), edges[0]);
        lines.push(`each of its ${edges.length} weights: dL/dw = a·δ`);
        lines.push(`largest: ${f(top.input)} × ${f(top.delta)} = ${f(top.grad)}`);
      }
      steps.push({ kind: "neuron", layer: l, index: j, title, lines, edges });
    }
  }

  params.forEach((p, i) => (p.grad = saved[i]));
  return { point, score: out.data, loss: loss.data, flows: loss.data > 0, steps };
}

/** Per-point hinge loss, without building a graph (to pick an interesting point). */
export function pointLoss(score: number, label: 1 | -1): number {
  return Math.max(0, 1 - label * score);
}
