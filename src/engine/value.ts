/**
 * Value — one number that remembers how it was made.
 *
 * A neural network is just a big formula. If every number in that formula remembers
 * (1) which numbers it was computed from and (2) how to pass a gradient back to them,
 * then we can ask "if I nudge this weight, how much does the loss change?" for every
 * weight at once. That question is the gradient, and answering it is backprop.
 *
 * This is a TypeScript port of Andrej Karpathy's micrograd (github.com/karpathy/micrograd).
 */

export type Op = "" | "+" | "*" | "**" | "tanh" | "relu" | "exp";

export class Value {
  /** The number itself, computed on the forward pass. */
  data: number;
  /**
   * d(final output) / d(this value). Starts at 0 and is filled in by backward().
   * "If this number went up by a tiny ε, the output would go up by grad · ε."
   */
  grad = 0;
  /**
   * How to push this node's grad down to its parents (the chain rule, one step).
   * Each operation below sets its own version. Leaves (inputs, weights) do nothing.
   */
  _backward: () => void = () => {};
  /** The values this one was computed from — the edges of the graph. */
  readonly _prev: readonly Value[];
  /** Which operation produced it, for debugging and drawing. */
  readonly op: Op;
  /** Optional name, e.g. "w[1][3][0]". */
  label: string;

  constructor(data: number, prev: readonly Value[] = [], op: Op = "", label = "") {
    this.data = data;
    this._prev = prev;
    this.op = op;
    this.label = label;
  }

  // ── Operations ──────────────────────────────────────────────────────────────
  // Each one does two things: compute the new number now (forward), and store a
  // closure that later says how the output's grad flows back into the inputs.
  // Note the `+=`: a value used twice (like a + a) must collect gradient from both uses.

  add(other: Value | number): Value {
    const o = toValue(other);
    const out = new Value(this.data + o.data, [this, o], "+");
    out._backward = () => {
      // d(a+b)/da = 1 and d(a+b)/db = 1: addition passes the gradient through unchanged.
      this.grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  mul(other: Value | number): Value {
    const o = toValue(other);
    const out = new Value(this.data * o.data, [this, o], "*");
    out._backward = () => {
      // d(a·b)/da = b and d(a·b)/db = a: each side gets the other side's value.
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  /** this ** k, for a plain number k (not a Value). */
  pow(k: number): Value {
    const out = new Value(this.data ** k, [this], "**");
    out._backward = () => {
      // Power rule: d(x^k)/dx = k · x^(k-1).
      this.grad += k * this.data ** (k - 1) * out.grad;
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], "tanh");
    out._backward = () => {
      // d tanh(x)/dx = 1 − tanh(x)². We already have tanh(x) = t, so reuse it.
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(this.data < 0 ? 0 : this.data, [this], "relu");
    out._backward = () => {
      // ReLU lets the gradient through where the input was positive, blocks it otherwise.
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  exp(): Value {
    const e = Math.exp(this.data);
    const out = new Value(e, [this], "exp");
    out._backward = () => {
      // d e^x/dx = e^x, which is the output itself.
      this.grad += e * out.grad;
    };
    return out;
  }

  // These are built from the ones above, so they need no backward of their own.
  neg(): Value {
    return this.mul(-1);
  }
  sub(other: Value | number): Value {
    return this.add(toValue(other).neg());
  }
  div(other: Value | number): Value {
    return this.mul(toValue(other).pow(-1));
  }

  // ── Backprop ────────────────────────────────────────────────────────────────

  /**
   * Fill in .grad for every value this one depends on.
   *
   * 1. List every node in topological order: a node appears only after all the nodes it
   *    was computed from. (Depth-first search; add a node once all its parents are added.)
   * 2. The output's gradient with respect to itself is 1.
   * 3. Walk the list backwards — from the output toward the inputs — and let each node
   *    run its _backward(). Going backwards guarantees a node's grad is complete (every
   *    use of it has reported in) before it passes that grad on to its own parents.
   *
   * Call zeroGrad() on the parameters before each new backward(), because grads add up.
   */
  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    // Iterative DFS (a recursive one would overflow the stack on big graphs).
    const stack: [Value, boolean][] = [[this, false]];
    while (stack.length) {
      const [v, childrenDone] = stack.pop()!;
      if (childrenDone) {
        topo.push(v);
        continue;
      }
      if (visited.has(v)) continue;
      visited.add(v);
      stack.push([v, true]);
      for (const p of v._prev) if (!visited.has(p)) stack.push([p, false]);
    }

    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }

  toString(): string {
    return `Value(data=${this.data}, grad=${this.grad})`;
  }
}

export function toValue(x: Value | number): Value {
  return x instanceof Value ? x : new Value(x);
}

/** Sum a list of Values without building a long chain of temporaries by hand. */
export function sum(values: readonly Value[]): Value {
  let acc = new Value(0);
  for (const v of values) acc = acc.add(v);
  return acc;
}
