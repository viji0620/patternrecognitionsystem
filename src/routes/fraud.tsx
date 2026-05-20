import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { ArrowLeft, ShieldAlert, CheckCircle2, AlertTriangle } from "lucide-react";
import {
  LogisticRegression,
  applyStandardize,
  makeFraudDataset,
  standardize,
  trainTestSplit,
} from "@/lib/ml";

export const Route = createFileRoute("/fraud")({
  component: FraudPage,
});

const FEATURES = [
  { key: "amount", label: "Amount ($)", default: 75 },
  { key: "hour", label: "Hour of day (0-23)", default: 14 },
  { key: "distance", label: "Distance from home (km)", default: 10 },
  { key: "txPerHour", label: "Transactions / hour", default: 2 },
  { key: "foreign", label: "Foreign transaction (0 or 1)", default: 0 },
] as const;

function FraudPage() {
  const { model, mean, std, trainAcc, testAcc, losses } = useMemo(() => {
    const { X, y } = makeFraudDataset(800);
    const { X: Xs, mean, std } = standardize(X);
    const { XTrain, yTrain, XTest, yTest } = trainTestSplit(Xs, y, 0.2);
    const model = new LogisticRegression(0.2, 400);
    model.fit(XTrain, yTrain);
    return {
      model,
      mean,
      std,
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
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link to="/" className="inline-flex items-center gap-2 text-sm text-slate-600 hover:text-slate-900">
            <ArrowLeft className="h-4 w-4" /> Back
          </Link>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <ShieldAlert className="h-4 w-4 text-rose-600" />
            Behavioral pattern recognition
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-10">
        <h1 className="text-3xl font-bold text-slate-900">Transaction fraud detection</h1>
        <p className="mt-2 max-w-2xl text-slate-600">
          A logistic regression classifier trained on synthetic transaction data. Adjust
          the inputs to see how the model scores the probability of fraud.
        </p>

        <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_1fr]">
          <div className="rounded-2xl border border-slate-200 bg-white p-6">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              Transaction features
            </h2>
            <div className="mt-4 space-y-4">
              {FEATURES.map((f) => (
                <label key={f.key} className="block">
                  <span className="text-sm font-medium text-slate-700">{f.label}</span>
                  <input
                    type="number"
                    value={values[f.key]}
                    onChange={(e) =>
                      setValues((v) => ({ ...v, [f.key]: Number(e.target.value) }))
                    }
                    className="mt-1 block w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  />
                </label>
              ))}
            </div>

            <div className="mt-6 flex gap-2">
              <PresetBtn onClick={() => setValues({ amount: 35, hour: 13, distance: 4, txPerHour: 1, foreign: 0 })}>
                Normal coffee buy
              </PresetBtn>
              <PresetBtn onClick={() => setValues({ amount: 1200, hour: 3, distance: 1800, txPerHour: 7, foreign: 1 })}>
                Suspicious foreign
              </PresetBtn>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-6">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              Prediction
            </h2>

            <div
              className={`mt-4 flex items-center gap-3 rounded-xl p-4 ${
                isFraud ? "bg-rose-50 text-rose-900" : "bg-emerald-50 text-emerald-900"
              }`}
            >
              {isFraud ? (
                <AlertTriangle className="h-6 w-6 text-rose-600" />
              ) : (
                <CheckCircle2 className="h-6 w-6 text-emerald-600" />
              )}
              <div>
                <div className="text-lg font-semibold">
                  {isFraud ? "Likely fraud" : "Looks legitimate"}
                </div>
                <div className="text-sm opacity-80">
                  Fraud probability: {(proba * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            <div className="mt-4">
              <div className="mb-1 flex justify-between text-xs text-slate-500">
                <span>0%</span>
                <span>fraud score</span>
                <span>100%</span>
              </div>
              <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200">
                <div
                  className={`h-full ${isFraud ? "bg-rose-500" : "bg-emerald-500"}`}
                  style={{ width: `${proba * 100}%` }}
                />
              </div>
            </div>

            <div className="mt-6 grid grid-cols-2 gap-4">
              <MetricBox label="Training accuracy" value={`${(trainAcc * 100).toFixed(1)}%`} />
              <MetricBox label="Test accuracy" value={`${(testAcc * 100).toFixed(1)}%`} />
            </div>

            <div className="mt-6">
              <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Training loss
              </div>
              <LossChart losses={losses} />
            </div>

            <details className="mt-6 rounded-lg bg-slate-50 p-4 text-sm">
              <summary className="cursor-pointer font-medium text-slate-700">
                Learned weights
              </summary>
              <div className="mt-3 grid grid-cols-2 gap-2 font-mono text-xs">
                {FEATURES.map((f, i) => (
                  <div key={f.key} className="flex justify-between rounded bg-white px-2 py-1">
                    <span className="text-slate-500">{f.key}</span>
                    <span className="text-slate-900">{model.weights[i].toFixed(3)}</span>
                  </div>
                ))}
                <div className="flex justify-between rounded bg-white px-2 py-1">
                  <span className="text-slate-500">bias</span>
                  <span className="text-slate-900">{model.bias.toFixed(3)}</span>
                </div>
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
      className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
    >
      {children}
    </button>
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="mt-1 text-xl font-semibold text-slate-900">{value}</div>
    </div>
  );
}

function LossChart({ losses }: { losses: number[] }) {
  const w = 320;
  const h = 80;
  const max = Math.max(...losses);
  const min = Math.min(...losses);
  const pts = losses
    .map((l, i) => {
      const x = (i / (losses.length - 1)) * w;
      const y = h - ((l - min) / (max - min || 1)) * h;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="mt-2 w-full rounded bg-slate-50">
      <polyline fill="none" stroke="#6366f1" strokeWidth="2" points={pts} />
    </svg>
  );
}
