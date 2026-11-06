/**
 * The 3D network. Each layer is a ring of neurons; rings sit one after another along the
 * network's axis. Every weight is a line from a neuron to a neuron in the next ring.
 * Line colour = how large that weight's gradient was on the last backward pass.
 * Neuron brightness = |activation| for the probe point.
 * Nothing here re-renders on a training step: useFrame reads the trainer directly.
 */
import { Component, memo, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Segment, Segments, type SegmentObject } from "@react-three/drei";
import { Bloom, EffectComposer } from "@react-three/postprocessing";
import * as THREE from "three";
import type { Trainer } from "./engine/trainer";
import { forwardNumeric } from "./engine/fast";
import type { Theme } from "./theme";

interface Props {
  trainer: Trainer;
  probe: number;
  theme: Theme;
  reducedMotion: boolean;
}

const LAYER_GAP = 1.9;
const LAYER_GAP_TALL = 1.45; // portrait: shorter axis so the rings can be drawn larger
const RING_RADIUS = 1.25;
const NEURON_RADIUS = 0.085;

/** Neuron positions per layer; the axis is x on wide screens and y on tall ones. */
function layout(sizes: readonly number[], vertical: boolean): THREE.Vector3[][] {
  const mid = (sizes.length - 1) / 2;
  return sizes.map((n, l) => {
    const along = (l - mid) * (vertical ? LAYER_GAP_TALL : LAYER_GAP);
    const r = n === 1 ? 0 : n === 2 ? 0.55 : RING_RADIUS;
    return Array.from({ length: n }, (_, i) => {
      const a = (i / n) * Math.PI * 2 + (n === 2 ? Math.PI / 2 : Math.PI / n);
      const u = Math.cos(a) * r;
      const w = Math.sin(a) * r;
      return vertical ? new THREE.Vector3(u, -along, w) : new THREE.Vector3(along, u, w);
    });
  });
}

interface Edge {
  layer: number; // index into model.layers
  from: number;
  to: number;
  start: THREE.Vector3;
  end: THREE.Vector3;
}

function Network({ trainer, probe, theme }: Omit<Props, "reducedMotion">) {
  const size = useThree((s) => s.size);
  const vertical = size.height > size.width;
  const sizes = trainer.model.sizes;
  const pos = useMemo(() => layout(sizes, vertical), [sizes, vertical]);
  const edges = useMemo<Edge[]>(() => {
    const out: Edge[] = [];
    for (let l = 0; l < sizes.length - 1; l++)
      for (let j = 0; j < sizes[l + 1]; j++)
        for (let i = 0; i < sizes[l]; i++)
          out.push({ layer: l, from: i, to: j, start: pos[l][i], end: pos[l + 1][j] });
    return out;
  }, [pos, sizes]);

  const segRefs = useRef<(SegmentObject | null)[]>([]);
  const neuronMats = useRef<(THREE.MeshBasicMaterial | null)[][]>(sizes.map(() => []));
  const smoothed = useRef<Float32Array>(new Float32Array(edges.length));

  const colors = useMemo(() => {
    const base = new THREE.Color(theme.hairline);
    const accent = new THREE.Color(theme.accent);
    const dim = new THREE.Color(theme.muted).lerp(new THREE.Color(theme.bg), theme.dark ? 0.55 : 0.35);
    return { base, accent, dim, on: new THREE.Color(theme.text), tmp: new THREE.Color() };
  }, [theme]);

  useFrame((_, delta) => {
    const model = trainer.model;

    // Edges: normalise |grad| by the largest one this step, ease toward it so colour moves calmly.
    let maxG = 1e-12;
    for (const e of edges) maxG = Math.max(maxG, Math.abs(model.layers[e.layer].neurons[e.to].w[e.from].grad));
    const ease = 1 - Math.exp(-delta * 10);
    edges.forEach((e, k) => {
      const w = model.layers[e.layer].neurons[e.to].w[e.from];
      const g = (Math.abs(w.grad) / maxG) ** 1.5; // steep: only the largest gradients light up
      smoothed.current[k] += (g - smoothed.current[k]) * ease;
      // Below a threshold an edge stays pure grey, so the accent means only "large gradient".
      const s = Math.max(0, (smoothed.current[k] - 0.15) / 0.85);
      const seg = segRefs.current[k];
      if (!seg) return;
      // Faint base set by |w|, then the accent rises with the gradient.
      const wMag = Math.min(1, Math.abs(w.data) / 1.5);
      colors.tmp.copy(colors.base).lerp(colors.dim, wMag).lerp(colors.accent, s);
      // In dark mode push bright edges past 1.0 so only they bloom.
      if (theme.dark) colors.tmp.multiplyScalar(1 + 2.2 * s * s);
      seg.color.copy(colors.tmp);
    });

    // Neurons: brightness = |activation| at the probe point.
    const p = trainer.data[probe];
    const acts = forwardNumeric(model, [p.x, p.y]);
    acts.forEach((layer, l) =>
      layer.forEach((a, i) => {
        const mat = neuronMats.current[l][i];
        if (!mat) return;
        // Output is linear (unbounded), so squash it for display only.
        const m = Math.min(1, Math.abs(l === acts.length - 1 ? Math.tanh(a) : a));
        mat.color.copy(colors.base).lerp(colors.on, 0.25 + 0.75 * m);
      }),
    );
  });

  return (
    <group>
      <Segments limit={edges.length} lineWidth={1.6} toneMapped={false} transparent={!theme.dark}>
        {edges.map((e, k) => (
          <Segment
            key={`${e.layer}-${e.from}-${e.to}-${vertical}`}
            ref={(r) => {
              segRefs.current[k] = r;
            }}
            start={e.start}
            end={e.end}
            color={theme.hairline}
          />
        ))}
      </Segments>
      {pos.map((layer, l) =>
        layer.map((p, i) => (
          <mesh
            key={`${l}-${i}-${vertical}`}
            position={p}
          >
            <sphereGeometry args={[l === 0 || l === sizes.length - 1 ? NEURON_RADIUS * 1.35 : NEURON_RADIUS, 24, 16]} />
            <meshBasicMaterial
              ref={(m) => {
                neuronMats.current[l][i] = m;
              }}
              color={theme.text}
              toneMapped={false}
            />
          </mesh>
        )),
      )}
    </group>
  );
}

function Rig({ reducedMotion }: { reducedMotion: boolean }) {
  const size = useThree((s) => s.size);
  const vertical = size.height > size.width;
  const camera = useThree((s) => s.camera);
  // Fit: the network spans ~5.7 units along x (wide) or ~4.4 along y (tall).
  const [spin, setSpin] = useState(true);
  useLayoutEffect(() => {
    const d = vertical ? 7.6 : Math.max(7.2, 7.2 * (1.25 / Math.min(1.25, size.width / size.height)));
    camera.position.set(vertical ? 1.4 : 3.6, vertical ? 0.8 : 2.2, d);
    camera.lookAt(0, 0, 0);
  }, [vertical, size.width, size.height, camera]);
  return (
    <OrbitControls
      makeDefault
      enablePan={false}
      minDistance={4}
      maxDistance={16}
      autoRotate={spin && !reducedMotion}
      autoRotateSpeed={0.35}
      onStart={() => setSpin(false)}
    />
  );
}

/** If WebGL cannot start, say so and leave the rest of the page working. */
class GlBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <p className="stage-note mono">3D view unavailable (WebGL failed to start). The data plot and loss curve still work.</p>
    ) : (
      this.props.children
    );
  }
}

/** Flips `ready` after the first rendered frame (shader compile can take a moment). */
function FirstFrame({ onReady }: { onReady: () => void }) {
  const done = useRef(false);
  useFrame(() => {
    if (!done.current) {
      done.current = true;
      onReady();
    }
  });
  return null;
}

export const NetworkView = memo(function NetworkView(props: Props) {
  const [ready, setReady] = useState(false);
  return (
    <GlBoundary>
      {!ready && <p className="stage-note mono">preparing the 3D view…</p>}
      <Scene {...props} onReady={() => setReady(true)} />
    </GlBoundary>
  );
});

function Scene({ trainer, probe, theme, reducedMotion, onReady }: Props & { onReady: () => void }) {
  return (
    <Canvas
      flat
      dpr={[1, 2]}
      camera={{ fov: 40, near: 0.1, far: 100 }}
      gl={{ antialias: true }}
      aria-label="3D view of the neural network; brighter lines carry larger gradients"
      role="img"
    >
      <color attach="background" args={[theme.bg]} />
      <Network trainer={trainer} probe={probe} theme={theme} />
      <Rig reducedMotion={reducedMotion} />
      {theme.dark && (
        <EffectComposer multisampling={4}>
          <Bloom mipmapBlur luminanceThreshold={1} luminanceSmoothing={0.2} intensity={0.9} radius={0.7} />
        </EffectComposer>
      )}
      <FirstFrame onReady={onReady} />
    </Canvas>
  );
}
