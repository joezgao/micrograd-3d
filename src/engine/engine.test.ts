import { describe, expect, it } from "vitest";
import { Value } from "./value";
import { MLP } from "./nn";
import { createRng } from "./random";
import { makeMoons } from "./data";
import { Trainer, lossOf } from "./trainer";
import { forwardNumeric } from "./fast";

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
