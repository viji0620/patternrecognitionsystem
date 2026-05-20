import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { ArrowLeft, Activity, Sparkles } from "lucide-react";
import {
  HEART_LABELS, KNN, applyStandardize, makeHeartbeatDataset, standardize, trainTestSplit,
} from "@/lib/ml";
import { AnimatedNumber } from "@/components/AnimatedNumber";

export const Route = createFileRoute("/heartbeat")({
  component: HeartbeatPage,
});

const FEATURES = [
  { key: "hr", label: "Heart rate", unit: "bpm", default: 72, min: 30, max: 180, step: 1 },
  { key: "qrs", label: "QRS width", unit: "ms", default: 90, min: 60, max: 180, step: 1 },
  { key: "qt", label: "QT interval", unit: "ms", default: 400, min: 280, max: 520, step: 1 },
  { key: "st", label: "ST deviation", unit: "mV", default: 0, min: -0.3, max: 0.5, step: 0.01 },
  { key: "var", label: "Beat variability", unit: "", default: 0.04, min: 0, max: 0.3, step: 0.005 },
] as const;

const PRESETS: { name: string; vals: Record<string, number>; dot: string }[] = [
  { name: "Normal", vals: { hr: 72, qrs: 90, qt: 400, st: 0, var: 0.04 }, dot: "bg-success shadow-[0_0_10px_oklch(0.86_0.2_155)]" },
  { name: "Bradycardia", vals: { hr: 45, qrs: 95, qt: 430, st: 0.02, var: 0.05 }, dot: "bg-primary shadow-[0_0_10px_oklch(0.85_0.18_195)]" },
  { name: "Tachycardia", vals: { hr: 135, qrs: 85, qt: 340, st: 0.05, var: 0.06 }, dot: "bg-accent shadow-[0_0_10px_oklch(0.8_0.22_310)]" },
  { name: "Arrhythmia", vals: { hr: 90, qrs: 135, qt: 460, st: 0.3, var: 0.22 }, dot: "bg-danger shadow-[0_0_10px_oklch(0.78_0.22_22)]" },
];

function HeartbeatPage() {
  const { model, mean, std, testAcc } = useMemo(() => {
    const { X, y } = makeHeartbeatDataset(800);
    const { X: Xs, mean, std } = standardize(X);
    const { XTrain, yTrain, XTest, yTest } = trainTestSplit(Xs, y, 0.2);
    const model = new KNN(7);
    model.fit(XTrain, yTrain, HEART_LABELS);
    return { model, mean, std, testAcc: model.accuracy(XTest, yTest) };
  }, []);

  const [values, setValues] = useState<Record<string, number>>(
    Object.fromEntries(FEATURES.map((f) => [f.key, f.default])),
  );

  const raw = FEATURES.map((f) => Number(values[f.key]) || 0);
  const scaled = applyStandardize(raw, mean, std);
  const pred = model.predict(scaled);

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none fixed inset-0 -z-10 neural-grid animate-grid-shift opacity-30" />
      <div className="pointer-events-none fixed -bottom-40 right-1/4 -z-10 h-[420px] w-[420px] rounded-full bg-success/20 blur-3xl animate-blob" />

      <header className="relative z-10 mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
        <Link to="/" className="inline-flex items-center gap-2 rounded-full border border-border bg-surface/60 px-4 py-2 text-sm text-muted-foreground backdrop-blur transition hover:text-foreground">
          <ArrowLeft className="h-4 w-4" /> Back
        </Link>
        <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-muted-foreground">
          <Activity className="h-3.5 w-3.5 text-success" /> medical · k-nearest neighbors
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-7xl px-6 pb-24">
        <div className="animate-fade-up">
          <div className="inline-flex items-center gap-2 rounded-full border border-border bg-surface/60 px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-accent backdrop-blur">
            <Sparkles className="h-3 w-3" /> module 02 / ecg classifier
          </div>
          <h1 className="mt-4 text-4xl font-bold tracking-tight text-foreground sm:text-5xl">
            Classify a heartbeat as it <span className="text-gradient">beats.</span>
          </h1>
          <p className="mt-3 max-w-2xl text-muted-foreground">
            k = 7 neighbors vote on whether your inputs describe a normal sinus rhythm or something the heart shouldn't be doing.
          </p>
        </div>

        {/* Live ECG strip */}
        <div className="relative mt-8 overflow-hidden rounded-3xl border border-border bg-[oklch(0.1_0.02_265)] p-6 animate-fade-up" style={{ animationDelay: "80ms" }}>
          <div className="absolute inset-0 pointer-events-none" style={{
            backgroundImage: "linear-gradient(oklch(0.78_0.19_155/0.07) 1px, transparent 1px), linear-gradient(90deg, oklch(0.78_0.19_155/0.07) 1px, transparent 1px)",
            backgroundSize: "20px 20px",
          }} />
          <div className="relative flex items-baseline justify-between">
            <div className="font-mono text-xs uppercase tracking-widest text-success">// live trace · lead II</div>
            <div className="font-mono text-xs text-muted-foreground">{values.hr} bpm</div>
          </div>
          <ECGWave hr={values.hr} st={values.st} qrs={values.qrs} variability={values["var"]} />
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_1.1fr]">
          {/* Controls */}
          <div className="rounded-3xl border border-border bg-surface/40 p-6 backdrop-blur animate-fade-up" style={{ animationDelay: "160ms" }}>
            <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">// presets</div>
            <div className="mt-3 grid grid-cols-2 gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.name}
                  onClick={() => setValues(p.vals)}
                  className="group flex items-center justify-between rounded-xl border border-border bg-surface-2/40 px-3 py-2.5 text-left text-sm font-medium text-foreground transition hover:border-primary hover:bg-surface-2"
                >
                  <span>{p.name}</span>
                  <span className={`h-2 w-2 rounded-full bg-${p.color} shadow-[0_0_10px] shadow-${p.color}`} />
                </button>
              ))}
            </div>

            <div className="mt-6 font-mono text-xs uppercase tracking-widest text-muted-foreground">// ecg features</div>
            <div className="mt-3 space-y-4">
              {FEATURES.map((f) => (
                <div key={f.key}>
                  <div className="flex items-baseline justify-between">
                    <label className="text-sm font-medium text-foreground">{f.label}</label>
                    <span className="font-mono text-sm text-success">
                      {values[f.key].toFixed(f.step < 1 ? 2 : 0)}{f.unit && ` ${f.unit}`}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    value={values[f.key]}
                    onChange={(e) => setValues((v) => ({ ...v, [f.key]: Number(e.target.value) }))}
                    className="mt-2 w-full accent-success"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Diagnosis */}
          <div className="space-y-6 animate-fade-up" style={{ animationDelay: "240ms" }}>
            <div className="relative overflow-hidden rounded-3xl border border-success/40 bg-gradient-to-br from-success/10 via-surface/40 to-accent/10 p-8 backdrop-blur">
              <div className="pointer-events-none absolute -right-20 -top-20 h-56 w-56 rounded-full bg-success/30 blur-3xl animate-pulse-glow" />
              <div className="relative font-mono text-[10px] uppercase tracking-widest text-success">predicted class</div>
              <div className="relative mt-2 flex items-end gap-4">
                <div key={pred.label} className="text-4xl font-bold text-foreground animate-scale-in sm:text-5xl">
                  {pred.label}
                </div>
                <div className="pb-2 font-mono text-sm text-muted-foreground">
                  confidence {(pred.confidence * 100).toFixed(0)}%
                </div>
              </div>

              {/* Vote bars */}
              <div className="relative mt-6 space-y-2.5">
                {HEART_LABELS.map((lbl) => {
                  const v = pred.votes[lbl] || 0;
                  const pct = (v / 7) * 100;
                  const isWinner = lbl === pred.label;
                  return (
                    <div key={lbl} className="flex items-center gap-3">
                      <div className={`w-28 font-mono text-xs ${isWinner ? "text-foreground" : "text-muted-foreground"}`}>{lbl}</div>
                      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${isWinner ? "bg-gradient-to-r from-success to-success-glow" : "bg-muted-foreground/40"}`}
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
              <Metric label="Neighbors" value={7} suffix="" decimals={0} />
              <Metric label="Classes" value={4} suffix="" decimals={0} />
            </div>
          </div>
        </div>
      </main>
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

function ECGWave({ hr, st, qrs, variability }: { hr: number; st: number; qrs: number; variability: number }) {
  const w = 800;
  const h = 140;
  // Build two copies for seamless marquee scroll
  const beats = Math.max(3, Math.round(hr / 12));
  const beatW = w / beats;
  const qrsScale = Math.min(2, qrs / 90);

  const buildPath = (offset: number) => {
    const path: string[] = [`M ${offset} ${h / 2}`];
    for (let i = 0; i < beats; i++) {
      const x0 = offset + i * beatW + (Math.sin(i * 7.3) * variability * 30);
      const mid = h / 2 - st * 60;
      path.push(`L ${x0 + beatW * 0.25} ${mid}`);
      path.push(`L ${x0 + beatW * 0.32} ${mid - 10}`); // P
      path.push(`L ${x0 + beatW * 0.4} ${mid}`);
      path.push(`L ${x0 + beatW * 0.46} ${mid + 14}`); // Q
      path.push(`L ${x0 + beatW * 0.5} ${mid - 55 * qrsScale}`); // R
      path.push(`L ${x0 + beatW * 0.54} ${mid + 18}`); // S
      path.push(`L ${x0 + beatW * 0.62} ${mid}`);
      path.push(`L ${x0 + beatW * 0.78} ${mid - 8}`); // T
      path.push(`L ${x0 + beatW} ${mid}`);
    }
    return path.join(" ");
  };

  const d = `${buildPath(0)} ${buildPath(w)}`;

  return (
    <div className="relative mt-4 h-[140px] overflow-hidden">
      <svg viewBox={`0 0 ${w * 2} ${h}`} preserveAspectRatio="none" className="absolute inset-0 h-full w-[200%] animate-marquee">
        <defs>
          <linearGradient id="ecgGlow" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="oklch(0.86 0.2 155 / 0.3)" />
            <stop offset="100%" stopColor="oklch(0.78 0.19 155 / 0)" />
          </linearGradient>
        </defs>
        <path d={d} fill="none" stroke="oklch(0.86 0.2 155)" strokeWidth="2" strokeLinejoin="round" />
        <path d={d} fill="none" stroke="oklch(0.86 0.2 155 / 0.4)" strokeWidth="6" strokeLinejoin="round" />
      </svg>
      {/* sweeping cursor */}
      <div className="pointer-events-none absolute inset-y-0 left-1/2 w-px bg-gradient-to-b from-transparent via-success to-transparent opacity-80" />
    </div>
  );
}
