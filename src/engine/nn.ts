/**
 * A tiny neural-network library on top of Value.
 * Neuron: weighted sum of inputs plus a bias, then (optionally) tanh.
 * Layer: a row of neurons that all read the same inputs.
 * MLP: layers in a chain, each reading the previous layer's outputs.
 */
import { Value, sum } from "./value";
import type { Rng } from "./random";

export class Neuron {
  readonly w: Value[];
  readonly b: Value;
  readonly nonlin: boolean;

  constructor(nin: number, rng: Rng, nonlin = true) {
    // Start weights small and random so neurons differ from each other.
    this.w = Array.from({ length: nin }, () => new Value(rng.uniform(-1, 1)));
    this.b = new Value(0);
    this.nonlin = nonlin;
  }

  call(x: readonly Value[]): Value {
    const act = sum(this.w.map((wi, i) => wi.mul(x[i]))).add(this.b);
    return this.nonlin ? act.tanh() : act;
  }

  parameters(): Value[] {
    return [...this.w, this.b];
  }
}

export class Layer {
  readonly neurons: Neuron[];

  constructor(nin: number, nout: number, rng: Rng, nonlin = true) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, rng, nonlin));
  }

  call(x: readonly Value[]): Value[] {
    return this.neurons.map((n) => n.call(x));
  }

  parameters(): Value[] {
    return this.neurons.flatMap((n) => n.parameters());
  }
}

export class MLP {
  readonly sizes: readonly number[];
  readonly layers: Layer[];

  /** sizes = [inputs, hidden..., outputs], e.g. [2, 8, 8, 1]. The last layer is linear. */
  constructor(sizes: readonly number[], rng: Rng) {
    this.sizes = sizes;
    this.layers = sizes
      .slice(1)
      .map((nout, i) => new Layer(sizes[i], nout, rng, i !== sizes.length - 2));
  }

  /** Forward pass that also returns every layer's outputs (for drawing activations). */
  forward(x: readonly (Value | number)[]): Value[][] {
    let cur = x.map((xi) => (xi instanceof Value ? xi : new Value(xi)));
    const acts: Value[][] = [cur];
    for (const layer of this.layers) {
      cur = layer.call(cur);
      acts.push(cur);
    }
    return acts;
  }

  call(x: readonly (Value | number)[]): Value {
    const acts = this.forward(x);
    return acts[acts.length - 1][0];
  }

  parameters(): Value[] {
    return this.layers.flatMap((l) => l.parameters());
  }

  zeroGrad(): void {
    for (const p of this.parameters()) p.grad = 0;
  }
}
