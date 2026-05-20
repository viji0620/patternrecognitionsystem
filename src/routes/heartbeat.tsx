import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { ArrowLeft, Activity } from "lucide-react";
import {
  HEART_LABELS,
  KNN,
  applyStandardize,
  makeHeartbeatDataset,
  standardize,
  trainTestSplit,
} from "@/lib/ml";

export const Route = createFileRoute("/heartbeat")({
  component: HeartbeatPage,
});

const FEATURES = [
  { key: "hr", label: "Heart rate (bpm)", default: 72 },
  { key: "qrs", label: "QRS width (ms)", default: 90 },
  { key: "qt", label: "QT interval (ms)", default: 400 },
  { key: "st", label: "ST deviation (mV)", default: 0 },
  { key: "var", label: "Beat variability", default: 0.04 },
] as const;

const PRESETS: { name: string; vals: Record<string, number> }[] = [
  { name: "Normal", vals: { hr: 72, qrs: 90, qt: 400, st: 0, var: 0.04 } },
  { name: "Bradycardia", vals: { hr: 45, qrs: 95, qt: 430, st: 0.02, var: 0.05 } },
  { name: "Tachycardia", vals: { hr: 135, qrs: 85, qt: 340, st: 0.05, var: 0.06 } },
  { name: "Arrhythmia", vals: { hr: 90, qrs: 135, qt: 460, st: 0.3, var: 0.22 } },
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
    <div className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link to="/" className="inline-flex items-center gap-2 text-sm text-slate-600 hover:text-slate-900">
            <ArrowLeft className="h-4 w-4" /> Back
          </Link>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <Activity className="h-4 w-4 text-emerald-600" />
            Medical pattern recognition
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-10">
        <h1 className="text-3xl font-bold text-slate-900">Heartbeat ECG classifier</h1>
        <p className="mt-2 max-w-2xl text-slate-600">
          A k-nearest neighbors model (k = 7) trained on synthetic ECG-derived features.
          Choose a preset or tune values to classify the beat.
        </p>

        <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_1fr]">
          <div className="rounded-2xl border border-slate-200 bg-white p-6">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              ECG features
            </h2>
            <div className="mt-4 flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.name}
                  onClick={() => setValues(p.vals)}
                  className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
                >
                  {p.name}
                </button>
              ))}
            </div>
            <div className="mt-4 space-y-4">
              {FEATURES.map((f) => (
                <label key={f.key} className="block">
                  <span className="text-sm font-medium text-slate-700">{f.label}</span>
                  <input
                    type="number"
                    step="0.01"
                    value={values[f.key]}
                    onChange={(e) =>
                      setValues((v) => ({ ...v, [f.key]: Number(e.target.value) }))
                    }
                    className="mt-1 block w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                  />
                </label>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-6">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
              Diagnosis
            </h2>

            <div className="mt-4 rounded-xl bg-gradient-to-br from-emerald-50 to-cyan-50 p-5">
              <div className="text-xs font-medium uppercase tracking-wide text-emerald-700">
                Predicted class
              </div>
              <div className="mt-1 text-3xl font-bold text-slate-900">{pred.label}</div>
              <div className="mt-1 text-sm text-slate-600">
                Confidence: {(pred.confidence * 100).toFixed(0)}% ({Math.round(pred.confidence * 7)}/7 neighbors agree)
              </div>
            </div>

            <ECGWave hr={values.hr} st={values.st} qrs={values.qrs} variability={values.var} />

            <div className="mt-6">
              <div className="text-xs font-medium uppercase tracking-wide text-slate-500">
                Vote distribution
              </div>
              <div className="mt-2 space-y-2">
                {HEART_LABELS.map((lbl) => {
                  const v = pred.votes[lbl] || 0;
                  return (
                    <div key={lbl} className="flex items-center gap-3">
                      <div className="w-28 text-sm text-slate-700">{lbl}</div>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-100">
                        <div
                          className="h-full bg-emerald-500"
                          style={{ width: `${(v / 7) * 100}%` }}
                        />
                      </div>
                      <div className="w-8 text-right text-sm tabular-nums text-slate-600">{v}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="mt-6 rounded-lg border border-slate-200 p-3">
              <div className="text-xs text-slate-500">Model test accuracy</div>
              <div className="mt-1 text-xl font-semibold text-slate-900">
                {(testAcc * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function ECGWave({ hr, st, qrs, variability }: { hr: number; st: number; qrs: number; variability: number }) {
  const w = 400;
  const h = 90;
  const beats = Math.max(2, Math.min(8, Math.round(hr / 15)));
  const beatW = w / beats;
  const qrsScale = Math.min(2, qrs / 90);
  const jitter = Math.min(10, variability * 80);

  const path: string[] = [`M 0 ${h / 2}`];
  for (let i = 0; i < beats; i++) {
    const x0 = i * beatW + (Math.random() - 0.5) * jitter;
    const mid = h / 2 - st * 30;
    path.push(`L ${x0 + beatW * 0.25} ${mid}`);
    path.push(`L ${x0 + beatW * 0.35} ${mid - 6}`); // P
    path.push(`L ${x0 + beatW * 0.45} ${mid}`);
    path.push(`L ${x0 + beatW * 0.48} ${mid + 8}`); // Q
    path.push(`L ${x0 + beatW * 0.5} ${mid - 35 * qrsScale}`); // R
    path.push(`L ${x0 + beatW * 0.53} ${mid + 12}`); // S
    path.push(`L ${x0 + beatW * 0.6} ${mid}`);
    path.push(`L ${x0 + beatW * 0.75} ${mid - 5}`); // T
    path.push(`L ${x0 + beatW} ${mid}`);
  }
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="mt-4 w-full rounded bg-slate-900">
      <path d={path.join(" ")} fill="none" stroke="#34d399" strokeWidth="1.5" />
    </svg>
  );
}
