import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { ArrowLeft, ShieldAlert, CheckCircle2, AlertTriangle, Sparkles } from "lucide-react";
import {
  LogisticRegression, applyStandardize, makeFraudDataset, standardize, trainTestSplit,
} from "@/lib/ml";
import { AnimatedNumber } from "@/components/AnimatedNumber";

export const Route = createFileRoute("/fraud")({
  component: FraudPage,
});

const FEATURES = [
  { key: "amount", label: "Amount", unit: "$", default: 75, min: 0, max: 3000, step: 5 },
  { key: "hour", label: "Hour of day", unit: "h", default: 14, min: 0, max: 23, step: 1 },
  { key: "distance", label: "Distance from home", unit: "km", default: 10, min: 0, max: 3000, step: 10 },
  { key: "txPerHour", label: "Transactions / hour", unit: "tx", default: 2, min: 1, max: 15, step: 1 },
  { key: "foreign", label: "Foreign transaction", unit: "", default: 0, min: 0, max: 1, step: 1 },
] as const;

function FraudPage() {
  const { model, mean, std, trainAcc, testAcc, losses } = useMemo(() => {
    const { X, y } = makeFraudDataset(800);
    const { X: Xs, mean, std } = standardize(X);
    const { XTrain, yTrain, XTest, yTest } = trainTestSplit(Xs, y, 0.2);
    const model = new LogisticRegression(0.2, 400);
    model.fit(XTrain, yTrain);
    return {
      model, mean, std,
      trainAcc: model.accuracy(XTrain, yTrain),
      testAcc: model.accuracy(XTest, yTest),
      losses: model.losses,
    };
  }, []);

  const [values, setValues] = useState<Record<string, number>>(
    Object.fromEntries(FEATURES.map((f) => [f.key, f.default])),
  );

  const raw = FEATURES.map((f) => Number(values[f.key]) || 0);
  const scaled = applyStandardize(raw, mean, std);
  const proba = model.predictProba(scaled);
  const isFraud = proba >= 0.5;

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none fixed inset-0 -z-10 neural-grid animate-grid-shift opacity-30" />
      <div className="pointer-events-none fixed -top-40 left-1/4 -z-10 h-[420px] w-[420px] rounded-full bg-danger/20 blur-3xl animate-blob" />

      <header className="relative z-10 mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
        <Link to="/" className="inline-flex items-center gap-2 rounded-full border border-border bg-surface/60 px-4 py-2 text-sm text-muted-foreground backdrop-blur transition hover:text-foreground">
          <ArrowLeft className="h-4 w-4" /> Back
        </Link>
        <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-muted-foreground">
          <ShieldAlert className="h-3.5 w-3.5 text-danger" /> behavioral · logistic regression
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-7xl px-6 pb-24">
        <div className="animate-fade-up">
          <div className="inline-flex items-center gap-2 rounded-full border border-border bg-surface/60 px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-primary backdrop-blur">
            <Sparkles className="h-3 w-3" /> module 01 / fraud engine
          </div>
          <h1 className="mt-4 text-4xl font-bold tracking-tight text-foreground sm:text-5xl">
            Score a transaction in <span className="text-gradient">real time.</span>
          </h1>
          <p className="mt-3 max-w-2xl text-muted-foreground">
            Tune the inputs and watch the model recompute its fraud probability the moment you move a slider.
          </p>
        </div>

        <div className="mt-10 grid gap-6 lg:grid-cols-[1fr_1.05fr]">
          {/* Inputs */}
          <div className="rounded-3xl border border-border bg-surface/40 p-6 backdrop-blur animate-fade-up" style={{ animationDelay: "80ms" }}>
            <div className="flex items-center justify-between">
              <h2 className="font-mono text-xs uppercase tracking-widest text-muted-foreground">// inputs</h2>
              <div className="flex gap-2">
                <PresetBtn onClick={() => setValues({ amount: 35, hour: 13, distance: 4, txPerHour: 1, foreign: 0 })}>
                  Coffee buy
                </PresetBtn>
                <PresetBtn onClick={() => setValues({ amount: 1200, hour: 3, distance: 1800, txPerHour: 7, foreign: 1 })}>
                  Suspicious
                </PresetBtn>
              </div>
            </div>

            <div className="mt-6 space-y-5">
              {FEATURES.map((f) => (
                <div key={f.key}>
                  <div className="flex items-baseline justify-between">
                    <label className="text-sm font-medium text-foreground">{f.label}</label>
                    <span className="font-mono text-sm text-primary">
                      {f.unit === "$" && "$"}{values[f.key]}{f.unit && f.unit !== "$" ? ` ${f.unit}` : ""}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    value={values[f.key]}
                    onChange={(e) => setValues((v) => ({ ...v, [f.key]: Number(e.target.value) }))}
                    className="mt-2 w-full accent-primary"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Prediction */}
          <div className="space-y-6 animate-fade-up" style={{ animationDelay: "160ms" }}>
            <div className={`relative overflow-hidden rounded-3xl border p-8 backdrop-blur transition ${isFraud ? "border-danger/50 bg-danger/5" : "border-success/50 bg-success/5"}`}>
              <div className={`pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full blur-3xl ${isFraud ? "bg-danger/40" : "bg-success/40"} animate-pulse-glow`} />

              <div className="relative flex items-center gap-3">
                <div className={`flex h-12 w-12 items-center justify-center rounded-xl ${isFraud ? "bg-danger/20 text-danger" : "bg-success/20 text-success"}`}>
                  {isFraud ? <AlertTriangle className="h-6 w-6" /> : <CheckCircle2 className="h-6 w-6" />}
                </div>
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">verdict</div>
                  <div className="text-2xl font-bold text-foreground">
                    {isFraud ? "Likely fraud" : "Looks legitimate"}
                  </div>
                </div>
              </div>

              <div className="relative mt-6">
                <div className="flex items-baseline justify-between font-mono text-xs">
                  <span className="text-muted-foreground">fraud probability</span>
                  <span className="text-2xl font-bold text-foreground">
                    <AnimatedNumber value={proba * 100} decimals={1} suffix="%" duration={400} />
                  </span>
                </div>
                <div className="mt-2 h-3 overflow-hidden rounded-full bg-surface-2">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${isFraud ? "bg-gradient-to-r from-danger to-danger-glow" : "bg-gradient-to-r from-success to-success-glow"}`}
                    style={{ width: `${proba * 100}%` }}
                  />
                </div>
                <div className="mt-1 flex justify-between font-mono text-[10px] text-muted-foreground">
                  <span>0%</span><span>threshold 50%</span><span>100%</span>
                </div>
              </div>
            </div>

            {/* Metrics row */}
            <div className="grid grid-cols-3 gap-4">
              <Metric label="Train acc" value={trainAcc * 100} />
              <Metric label="Test acc" value={testAcc * 100} />
              <Metric label="Epochs" value={model.losses.length} suffix="" decimals={0} />
            </div>

            {/* Loss chart */}
            <div className="rounded-3xl border border-border bg-surface/40 p-6 backdrop-blur">
              <div className="flex items-baseline justify-between">
                <div className="font-mono text-xs uppercase tracking-widest text-muted-foreground">// training loss</div>
                <div className="font-mono text-xs text-primary">final: {losses[losses.length - 1].toFixed(4)}</div>
              </div>
              <LossChart losses={losses} />
            </div>

            {/* Weights */}
            <details className="rounded-3xl border border-border bg-surface/40 p-6 backdrop-blur">
              <summary className="cursor-pointer font-mono text-xs uppercase tracking-widest text-muted-foreground">
                // learned weights
              </summary>
              <div className="mt-4 space-y-2 font-mono text-xs">
                {FEATURES.map((f, i) => {
                  const w = model.weights[i];
                  const pct = Math.min(100, (Math.abs(w) / Math.max(...model.weights.map(Math.abs))) * 100);
                  return (
                    <div key={f.key} className="flex items-center gap-3">
                      <span className="w-28 text-muted-foreground">{f.key}</span>
                      <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                        <div
                          className={`absolute top-0 h-full ${w >= 0 ? "left-1/2 bg-danger" : "right-1/2 bg-success"}`}
                          style={{ width: `${pct / 2}%` }}
                        />
                        <div className="absolute left-1/2 top-0 h-full w-px bg-border-strong" />
                      </div>
                      <span className="w-16 text-right text-foreground">{w.toFixed(3)}</span>
                    </div>
                  );
                })}
              </div>
            </details>
          </div>
        </div>
      </main>
    </div>
  );
}

function PresetBtn({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="rounded-full border border-border bg-surface-2/60 px-3 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground transition hover:border-primary hover:text-primary"
    >
      {children}
    </button>
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

function LossChart({ losses }: { losses: number[] }) {
  const w = 400;
  const h = 100;
  const max = Math.max(...losses);
  const min = Math.min(...losses);
  const pts = losses.map((l, i) => {
    const x = (i / (losses.length - 1)) * w;
    const y = h - ((l - min) / (max - min || 1)) * (h - 8) - 4;
    return `${x},${y}`;
  });
  const line = pts.join(" ");
  const area = `0,${h} ${line} ${w},${h}`;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="mt-3 w-full">
      <defs>
        <linearGradient id="lossfill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="oklch(0.78 0.17 200 / 0.5)" />
          <stop offset="100%" stopColor="oklch(0.78 0.17 200 / 0)" />
        </linearGradient>
      </defs>
      <polygon points={area} fill="url(#lossfill)" />
      <polyline points={line} fill="none" stroke="oklch(0.85 0.18 195)" strokeWidth="1.5" />
    </svg>
  );
}
