import { describe, expect, it } from "vitest";
import { Value } from "./value";
import { MLP } from "./nn";
import { createRng } from "./random";
import { makeMoons } from "./data";
import { Trainer, lossOf } from "./trainer";
import { forwardNumeric } from "./fast";
import { buildWalkthrough } from "./walkthrough";

describe("Value backprop against hand-computed gradients", () => {
  it("f = a*b + c", () => {
    // df/da = b = -3, df/db = a = 2, df/dc = 1
    const a = new Value(2);
    const b = new Value(-3);
    const c = new Value(10);
    const f = a.mul(b).add(c);
    f.backward();
    expect(f.data).toBe(4);
    expect(a.grad).toBe(-3);
    expect(b.grad).toBe(2);
    expect(c.grad).toBe(1);
  });

  it("a value used twice accumulates gradient: b = a + a", () => {
    const a = new Value(3);
    const b = a.add(a);
    b.backward();
    expect(a.grad).toBe(2);
  });

  it("d = a*b, e = a+b, f = d*e (shared parents)", () => {
    // f = a²b + ab²; df/da = 2ab + b² = 2(-2)(3) + 9 = -3; df/db = a² + 2ab = 4 - 12 = -8
    const a = new Value(-2);
    const b = new Value(3);
    const f = a.mul(b).mul(a.add(b));
    f.backward();
    expect(f.data).toBe(-6);
    expect(a.grad).toBe(-3);
    expect(b.grad).toBe(-8);
  });

  it("Karpathy's neuron: o = tanh(x1*w1 + x2*w2 + b)", () => {
    // n = 2*(-3) + 0*1 + 6.88137... = 0.88137..., tanh(n) = 1/√2, 1 - tanh² = 0.5
    const x1 = new Value(2);
    const x2 = new Value(0);
    const w1 = new Value(-3);
    const w2 = new Value(1);
    const b = new Value(6.881373587019543);
    const o = x1.mul(w1).add(x2.mul(w2)).add(b).tanh();
    o.backward();
    expect(o.data).toBeCloseTo(Math.SQRT1_2, 6);
    expect(x1.grad).toBeCloseTo(-1.5, 6); // w1 * 0.5
    expect(w1.grad).toBeCloseTo(1.0, 6); // x1 * 0.5
    expect(x2.grad).toBeCloseTo(0.5, 6); // w2 * 0.5
    expect(w2.grad).toBeCloseTo(0.0, 6); // x2 * 0.5
    expect(b.grad).toBeCloseTo(0.5, 6);
  });

  it("pow, div, sub, neg", () => {
    // f = (a² - b) / b with a = 4, b = 2 → f = 7
    // df/da = 2a/b = 4; df/db = -1/b - (a² - b)/b² ... = d/db (a²/b - 1) = -a²/b² = -4
    const a = new Value(4);
    const b = new Value(2);
    const f = a.pow(2).sub(b).div(b);
    f.backward();
    expect(f.data).toBeCloseTo(7, 12);
    expect(a.grad).toBeCloseTo(4, 12);
    expect(b.grad).toBeCloseTo(-4, 12);
  });

  it("relu blocks gradient for negative inputs, exp passes e^x", () => {
    const p = new Value(1.5);
    const n = new Value(-1.5);
    const f = p.relu().add(n.relu()).add(p.exp());
    f.backward();
    expect(p.grad).toBeCloseTo(1 + Math.exp(1.5), 12);
    expect(n.grad).toBe(0);
  });
});

describe("MLP gradients match finite differences", () => {
  it("every parameter of a 2-4-4-1 net", () => {
    const model = new MLP([2, 4, 4, 1], createRng(3));
    const data = makeMoons(10, 0.1, 5);
    const f = () => lossOf(model, data).loss;
    model.zeroGrad();
    f().backward();
    const h = 1e-6;
    for (const p of model.parameters()) {
      const analytic = p.grad;
      const orig = p.data;
      p.data = orig + h;
      const up = f().data;
      p.data = orig - h;
      const down = f().data;
      p.data = orig;
      const numeric = (up - down) / (2 * h);
      expect(analytic).toBeCloseTo(numeric, 5);
    }
  });
});

describe("numeric forward (drawing fast path)", () => {
  it("matches the Value forward pass layer by layer", () => {
    const model = new MLP([2, 8, 8, 1], createRng(9));
    const acts = model.forward([0.3, -0.7]);
    const fast = forwardNumeric(model, [0.3, -0.7]);
    acts.forEach((layer, l) => layer.forEach((v, j) => expect(fast[l][j]).toBeCloseTo(v.data, 12)));
  });
});

describe("backprop walkthrough (one point, neuron by neuron)", () => {
  const setup = () => {
    const model = new MLP([2, 8, 8, 1], createRng(1337));
    const data = makeMoons(100, 0.1, 7);
    // A point the untrained model gets wrong, so gradient flows.
    const p = data.find((q) => q.label * forwardNumeric(model, [q.x, q.y])[3][0] < 1)!;
    return { model, p };
  };

  it("visits loss, output, hidden 2, hidden 1, inputs in backward order (20 steps)", () => {
    const { model, p } = setup();
    const w = buildWalkthrough(model, p);
    expect(w.flows).toBe(true);
    expect(w.steps.map((s) => s.layer)).toEqual([-1, 3, ...Array(8).fill(2), ...Array(8).fill(1), 0, 0]);
  });

  it("every edge obeys the chain rule: dL/dw = a_i · δ_j", () => {
    const { model, p } = setup();
    const w = buildWalkthrough(model, p);
    const edges = w.steps.flatMap((s) => s.edges);
    expect(edges).toHaveLength(2 * 8 + 8 * 8 + 8);
    for (const e of edges) expect(e.grad).toBeCloseTo(e.input * e.delta, 12);
  });

  it("output δ is −y when the margin is violated, and grads match finite differences", () => {
    const { model, p } = setup();
    const w = buildWalkthrough(model, p);
    const outEdges = w.steps[1].edges;
    for (const e of outEdges) expect(e.delta).toBe(-p.label);
    const h = 1e-6;
    const lossAt = () => Math.max(0, 1 - p.label * forwardNumeric(model, [p.x, p.y])[3][0]);
    for (const e of w.steps.flatMap((s) => s.edges)) {
      const wv = model.layers[e.layer].neurons[e.to].w[e.from];
      const orig = wv.data;
      wv.data = orig + h;
      const up = lossAt();
      wv.data = orig - h;
      const down = lossAt();
      wv.data = orig;
      expect(e.grad).toBeCloseTo((up - down) / (2 * h), 5);
    }
  });

  it("leaves the training gradients on the parameters untouched", () => {
    const { model, p } = setup();
    model.parameters().forEach((q, i) => (q.grad = i * 0.01));
    buildWalkthrough(model, p);
    model.parameters().forEach((q, i) => expect(q.grad).toBe(i * 0.01));
  });

  it("a point beyond the margin has zero loss and no gradient", () => {
    const t = new Trainer();
    t.run(300);
    const p = t.data.find((q) => q.label * forwardNumeric(t.model, [q.x, q.y])[3][0] > 1.2)!;
    const w = buildWalkthrough(t.model, p);
    expect(w.flows).toBe(false);
    for (const e of w.steps.flatMap((s) => s.edges)) expect(e.grad).toBe(0);
  });
});

describe("training on moons", () => {
  it("loss falls and accuracy reaches ≥ 95%", () => {
    const t = new Trainer();
    const first = t.trainStep().loss;
    t.run(400);
    expect(t.history.at(-1)!).toBeLessThan(first * 0.25);
    expect(t.accuracy).toBeGreaterThanOrEqual(0.95);
  });

  it("reset replays the same run (deterministic)", () => {
    const t = new Trainer();
    t.run(20);
    const a = t.history.slice();
    t.reset();
    t.run(20);
    expect(t.history).toEqual(a);
  });
});
