import { createFileRoute, Link } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Eye, Sparkles, Trash2, Circle, Square, Triangle, Minus, Upload } from "lucide-react";
import {
  KNN, SHAPE_LABELS, extractShapeFeatures, featuresToVec,
  makeShapeDataset, standardize, applyStandardize, trainTestSplit,
} from "@/lib/ml";
import { AnimatedNumber } from "@/components/AnimatedNumber";

export const Route = createFileRoute("/shapes")({
  component: ShapesPage,
});

const W = 280;
const H = 280;

const PALETTE: { name: string; hex: string }[] = [
  { name: "cyan", hex: "oklch(0.85 0.18 195)" },
  { name: "violet", hex: "oklch(0.8 0.22 310)" },
  { name: "green", hex: "oklch(0.86 0.2 155)" },
  { name: "amber", hex: "oklch(0.85 0.18 75)" },
  { name: "red", hex: "oklch(0.78 0.22 22)" },
];

function ShapesPage() {
  const { model, mean, std, testAcc } = useMemo(() => {
    const { X, y } = makeShapeDataset(800);
    const { X: Xs, mean, std } = standardize(X);
    const { XTrain, yTrain, XTest, yTest } = trainTestSplit(Xs, y, 0.2);
    const model = new KNN(7);
    model.fit(XTrain, yTrain, SHAPE_LABELS);
    return { model, mean, std, testAcc: model.accuracy(XTest, yTest) };
  }, []);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);
  const [color, setColor] = useState(PALETTE[0]);
  const [brush, setBrush] = useState(12);
  const [version, setVersion] = useState(0);

  // initial fill
  useEffect(() => {
    const c = canvasRef.current!;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "oklch(0.13 0.02 265)";
    ctx.fillRect(0, 0, W, H);
  }, []);

  const getPos = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * W,
      y: ((e.clientY - r.top) / r.height) * H,
    };
  };

  const start = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drawing.current = true;
    last.current = getPos(e);
  };
  const move = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return;
    const p = getPos(e);
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.strokeStyle = color.hex;
    ctx.lineWidth = brush;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(last.current!.x, last.current!.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    last.current = p;
    setVersion((v) => v + 1);
  };
  const end = () => { drawing.current = false; last.current = null; };

  const clear = () => {
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.fillStyle = "oklch(0.13 0.02 265)";
    ctx.fillRect(0, 0, W, H);
    setVersion((v) => v + 1);
  };

  // Extract features whenever canvas changes (throttled by RAF).
  const [features, setFeatures] = useState<ReturnType<typeof extractShapeFeatures> | null>(null);
  const [prediction, setPrediction] = useState<{ label: string; confidence: number; votes: Record<string, number> } | null>(null);

  const rafRef = useRef(0);
  const recompute = useCallback(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    const img = ctx.getImageData(0, 0, W, H);
    const data = img.data;

    // Sample background color from the four corners so the mask works for
    // both the dark-themed drawing surface AND uploaded photos with any bg.
    const sample = (x: number, y: number) => {
      const i = (y * W + x) * 4;
      return [data[i], data[i + 1], data[i + 2]];
    };
    const corners = [sample(2, 2), sample(W - 3, 2), sample(2, H - 3), sample(W - 3, H - 3)];
    const bg = [0, 1, 2].map((k) => corners.reduce((s, c) => s + c[k], 0) / corners.length);

    const mask = new Uint8Array(W * H);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      const dr = data[i] - bg[0];
      const dg = data[i + 1] - bg[1];
      const db = data[i + 2] - bg[2];
      const dist = Math.sqrt(dr * dr + dg * dg + db * db);
      mask[j] = dist > 55 ? 1 : 0;
    }

    const pv = previewRef.current;
    if (pv) {
      const pctx = pv.getContext("2d")!;
      const out = pctx.createImageData(W, H);
      for (let i = 0, j = 0; i < mask.length; i++, j += 4) {
        const v = mask[i] ? 220 : 30;
        out.data[j] = v; out.data[j + 1] = v; out.data[j + 2] = v; out.data[j + 3] = 255;
      }
      pctx.putImageData(out, 0, 0);
    }

    const f = extractShapeFeatures(mask, W, H);
    setFeatures(f);
    if (f) {
      const vec = applyStandardize(featuresToVec(f), mean, std);
      setPrediction(model.predict(vec));
    } else {
      setPrediction(null);
    }
  }, [mean, std, model]);

  const onUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const ctx = canvasRef.current!.getContext("2d")!;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, W, H);
      // contain-fit
      const scale = Math.min(W / img.width, H / img.height);
      const dw = img.width * scale;
      const dh = img.height * scale;
      ctx.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
      URL.revokeObjectURL(url);
      setVersion((v) => v + 1);
    };
    img.src = url;
    e.target.value = "";
  };


  useEffect(() => {
    cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(recompute);
    return () => cancelAnimationFrame(rafRef.current);
  }, [version, recompute]);

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none fixed inset-0 -z-10 neural-grid animate-grid-shift opacity-30" />
      <div className="pointer-events-none fixed -top-40 left-1/3 -z-10 h-[460px] w-[460px] rounded-full bg-accent/25 blur-3xl animate-blob" />

      <header className="relative z-10 mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
        <Link to="/" className="inline-flex items-center gap-2 rounded-full border border-border bg-surface/60 px-4 py-2 text-sm text-muted-foreground backdrop-blur transition hover:text-foreground">
          <ArrowLeft className="h-4 w-4" /> Back
        </Link>
        <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-muted-foreground">
          <Eye className="h-3.5 w-3.5 text-accent" /> image · shape recognizer
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-7xl px-6 pb-24">
        <div className="animate-fade-up">
          <div className="inline-flex items-center gap-2 rounded-full border border-border bg-surface/60 px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-accent backdrop-blur">
            <Sparkles className="h-3 w-3" /> module 03 / vision pipeline
          </div>
          <h1 className="mt-4 text-4xl font-bold tracking-tight text-foreground sm:text-5xl">
            Draw a shape — let the model <span className="text-gradient">see it.</span>
          </h1>
          <p className="mt-3 max-w-2xl text-muted-foreground">
            Pixels are reduced to six geometric features — fill, aspect, circularity,
            symmetry, corner energy — then a k-NN classifier votes on what you drew.
          </p>
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-[auto_1fr]">
          {/* Canvas card */}
          <div className="rounded-3xl border border-border bg-surface/40 p-6 backdrop-blur animate-fade-up" style={{ animationDelay: "80ms" }}>
            <div className="flex items-center justify-between gap-2">
              <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">// canvas · 280×280</div>
              <div className="flex items-center gap-2">
                <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-full border border-border bg-surface-2/60 px-3 py-1 text-xs text-muted-foreground transition hover:border-accent hover:text-foreground">
                  <Upload className="h-3 w-3" /> upload
                  <input type="file" accept="image/*" className="hidden" onChange={onUpload} />
                </label>
                <button
                  onClick={clear}
                  className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-2/60 px-3 py-1 text-xs text-muted-foreground transition hover:border-danger hover:text-foreground"
                >
                  <Trash2 className="h-3 w-3" /> clear
                </button>
              </div>
            </div>

            <div className="relative mt-4">
              <canvas
                ref={canvasRef}
                width={W}
                height={H}
                onPointerDown={start}
                onPointerMove={move}
                onPointerUp={end}
                onPointerLeave={end}
                className="rounded-2xl border border-border-strong bg-[oklch(0.13_0.02_265)] touch-none cursor-crosshair shadow-[var(--shadow-card)]"
                style={{ width: W, height: H }}
              />
              <div className="pointer-events-none absolute inset-0 rounded-2xl shimmer-border" />
            </div>

            {/* Tools */}
            <div className="mt-5 space-y-4">
              <div>
                <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">color</div>
                <div className="mt-2 flex gap-2">
                  {PALETTE.map((p) => (
                    <button
                      key={p.name}
                      onClick={() => setColor(p)}
                      className={`h-8 w-8 rounded-full border transition ${color.name === p.name ? "border-foreground scale-110" : "border-border"}`}
                      style={{ background: p.hex, boxShadow: color.name === p.name ? `0 0 18px ${p.hex}` : "none" }}
                      aria-label={p.name}
                    />
                  ))}
                </div>
              </div>
              <div>
                <div className="flex items-baseline justify-between">
                  <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">brush</div>
                  <div className="font-mono text-xs text-foreground">{brush}px</div>
                </div>
                <input
                  type="range" min={4} max={30} value={brush}
                  onChange={(e) => setBrush(Number(e.target.value))}
                  className="mt-2 w-full accent-accent"
                />
              </div>
              <div>
                <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">try drawing</div>
                <div className="mt-2 grid grid-cols-4 gap-2">
                  {[
                    { i: <Circle className="h-4 w-4" />, n: "Circle" },
                    { i: <Square className="h-4 w-4" />, n: "Square" },
                    { i: <Triangle className="h-4 w-4" />, n: "Triangle" },
                    { i: <Minus className="h-4 w-4" />, n: "Line" },
                  ].map((h) => (
                    <div key={h.n} className="flex flex-col items-center gap-1 rounded-xl border border-border bg-surface-2/40 py-2 text-[10px] text-muted-foreground">
                      {h.i}{h.n}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Prediction + features */}
          <div className="space-y-6 animate-fade-up" style={{ animationDelay: "160ms" }}>
            <div className="relative overflow-hidden rounded-3xl border border-accent/40 bg-gradient-to-br from-accent/10 via-surface/40 to-primary/10 p-8 backdrop-blur">
              <div className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-accent/30 blur-3xl animate-pulse-glow" />
              <div className="relative font-mono text-[10px] uppercase tracking-widest text-accent">predicted shape</div>
              <div className="relative mt-2 flex items-end gap-4">
                <div key={prediction?.label ?? "none"} className="text-4xl font-bold text-foreground animate-scale-in sm:text-5xl">
                  {prediction?.label ?? "—"}
                </div>
                <div className="pb-2 font-mono text-sm text-muted-foreground">
                  {prediction ? `confidence ${(prediction.confidence * 100).toFixed(0)}%` : "draw something"}
                </div>
              </div>

              <div className="relative mt-6 space-y-2.5">
                {SHAPE_LABELS.map((lbl) => {
                  const v = prediction?.votes[lbl] || 0;
                  const pct = (v / 7) * 100;
                  const isWinner = lbl === prediction?.label;
                  return (
                    <div key={lbl} className="flex items-center gap-3">
                      <div className={`w-24 font-mono text-xs ${isWinner ? "text-foreground" : "text-muted-foreground"}`}>{lbl}</div>
                      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${isWinner ? "bg-gradient-to-r from-accent to-accent-glow" : "bg-muted-foreground/40"}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <div className="w-6 text-right font-mono text-xs text-foreground tabular-nums">{v}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <Metric label="Test acc" value={testAcc * 100} />
              <Metric label="Features" value={6} suffix="" decimals={0} />
              <Metric label="Classes" value={4} suffix="" decimals={0} />
            </div>

            {/* Feature panel */}
            <div className="grid gap-4 sm:grid-cols-[auto_1fr]">
              <div className="rounded-3xl border border-border bg-surface/40 p-4 backdrop-blur">
                <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">// binary mask</div>
                <canvas
                  ref={previewRef}
                  width={W}
                  height={H}
                  className="mt-3 rounded-xl border border-border"
                  style={{ width: 160, height: 160, imageRendering: "pixelated" }}
                />
              </div>
              <div className="rounded-3xl border border-border bg-surface/40 p-5 backdrop-blur">
                <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">// extracted features</div>
                <div className="mt-3 space-y-3">
                  {features ? (
                    [
                      { k: "Fill ratio", v: features.fillRatio },
                      { k: "Aspect", v: features.aspect },
                      { k: "Circularity", v: features.circularity },
                      { k: "V-symmetry", v: features.vSym },
                      { k: "H-symmetry", v: features.hSym },
                      { k: "Corner score", v: features.cornerScore },
                    ].map((row) => (
                      <FeatureRow key={row.k} k={row.k} v={row.v} />
                    ))
                  ) : (
                    <div className="font-mono text-xs text-muted-foreground">awaiting input…</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function FeatureRow({ k, v }: { k: string; v: number }) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <div className="text-xs text-muted-foreground">{k}</div>
        <div className="font-mono text-xs text-foreground tabular-nums">{v.toFixed(3)}</div>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-2">
        <div
          className="h-full rounded-full bg-gradient-to-r from-primary to-accent transition-all duration-300"
          style={{ width: `${Math.max(2, v * 100)}%` }}
        />
      </div>
    </div>
  );
}

function Metric({ label, value, suffix = "%", decimals = 1 }: { label: string; value: number; suffix?: string; decimals?: number }) {
  return (
    <div className="rounded-2xl border border-border bg-surface/40 p-4 backdrop-blur">
      <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
      <div className="mt-1 font-mono text-2xl font-bold text-foreground">
        <AnimatedNumber value={value} decimals={decimals} suffix={suffix} />
      </div>
    </div>
  );
}
