# micrograd 3D

![micrograd 3D: stepping through backprop, one neuron's incoming weights lit by their gradient](docs/screenshot.png)

Watch a tiny neural network learn, in 3D. A 2 → 8 → 8 → 1 network learns to separate two
interleaving half-moons while every weight is drawn as a line that glows when backprop sends a
large gradient through it. Pause at any moment and step through one point's backward pass,
neuron by neuron, with the chain rule written out.

For CS students learning backprop. The autograd engine is
[Karpathy's micrograd](https://github.com/karpathy/micrograd), rewritten from scratch in TypeScript.

## Run it

**Live:** [joezgao.github.io/micrograd-3d/](https://joezgao.github.io/micrograd-3d/) (nothing to install).

```bash
npm install
npm run dev      # http://localhost:5173
npm test         # engine tests: gradients by hand, finite differences, chain rule
npm run build    # typecheck + production build into dist/
```

Keys: <kbd>Space</kbd> play / pause · <kbd>→</kbd> one training step · <kbd>R</kbd> reset ·
<kbd>B</kbd> step through backprop (then <kbd>←</kbd> <kbd>→</kbd>, <kbd>Esc</kbd> to leave).

## How it works

1. `Value` (`src/engine/value.ts`) is one number that remembers which numbers made it and how to pass a gradient back to them.
2. `backward()` sorts the graph so every node comes after its inputs, sets the output's gradient to 1, and walks the list in reverse, applying the chain rule at each node.
3. `Neuron`, `Layer` and `MLP` (`nn.ts`) are just weighted sums of `Value`s followed by `tanh`.
4. Each training step builds the hinge loss over all 100 points, calls `backward()`, and nudges every weight against its gradient (plain SGD).
5. The 3D view (React Three Fiber, loaded lazily) colours each edge by |∂L/∂w| from that step; bloom makes the largest ones glow.
6. Backprop mode rebuilds one point's graph, runs the same `backward()`, and replays it neuron by neuron: δ = (Σ w·δ) · tanh′(z), ∂L/∂w = a·δ.


## What the tests check

`npm test` runs 15 tests (`src/engine/engine.test.ts`):

- gradients of small expressions against hand-computed values, including a value used twice and shared parents;
- `pow`, `div`, `sub`, `neg`, `relu` and `exp` gradients;
- every parameter of a 2-4-4-1 network against finite differences;
- the fast numeric forward pass used for drawing matches the `Value` forward pass, layer by layer;
- the backprop walkthrough: backward order, the chain rule on every edge, the margin cases, and that it leaves the training gradients untouched;
- training on the moons: the loss falls and accuracy reaches at least 95%, and reset replays the same run.

## Limits

- Scalar autograd: every weight is its own `Value` object, so it is slow by design. It suits a 2 → 8 → 8 → 1 network and 100 points, not real datasets.
- A toy problem: two half-moons in 2D. The accuracy shown is on the same 100 points the network trains on; there is no held-out set.
- One training recipe: hinge loss, `tanh`, L2 regularisation (1e-4) and plain full-batch SGD, with no momentum or learning-rate schedule.

## License

MIT. See [LICENSE](LICENSE).

Built with Claude for Joe Gao.
