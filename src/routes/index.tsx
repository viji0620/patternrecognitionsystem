import { createFileRoute, Link } from "@tanstack/react-router";
import { Brain, Activity, ShieldAlert, ArrowRight } from "lucide-react";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      <header className="border-b border-slate-200/60 bg-white/60 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-indigo-600" />
            <span className="font-semibold text-slate-900">PatternLab</span>
          </div>
          <nav className="flex items-center gap-6 text-sm text-slate-600">
            <Link to="/fraud" className="hover:text-slate-900">Fraud detection</Link>
            <Link to="/heartbeat" className="hover:text-slate-900">Heartbeat ECG</Link>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-16">
        <section className="max-w-3xl">
          <span className="inline-flex items-center rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700">
            Machine learning in your browser
          </span>
          <h1 className="mt-4 text-5xl font-bold tracking-tight text-slate-900">
            A pattern recognition system, trained live in your tab.
          </h1>
          <p className="mt-5 text-lg text-slate-600">
            Two real ML pipelines — logistic regression for behavioral fraud detection
            and k-nearest neighbors for medical heartbeat classification — implemented
            from scratch and trained on synthetic data right in the browser.
          </p>
        </section>

        <section className="mt-12 grid gap-6 md:grid-cols-2">
          <DemoCard
            to="/fraud"
            icon={<ShieldAlert className="h-6 w-6 text-rose-600" />}
            tag="Behavioral patterns"
            title="Fraud detection"
            desc="Logistic regression learns the difference between normal and unusual transactions, then scores any payment you enter."
            accent="from-rose-50 to-orange-50"
          />
          <DemoCard
            to="/heartbeat"
            icon={<Activity className="h-6 w-6 text-emerald-600" />}
            tag="Medical patterns"
            title="Heartbeat ECG classifier"
            desc="A k-nearest neighbors model classifies a heartbeat as normal, bradycardia, tachycardia, or arrhythmia from ECG-like features."
            accent="from-emerald-50 to-cyan-50"
          />
        </section>

        <section className="mt-16 grid gap-4 rounded-2xl border border-slate-200 bg-white p-8 md:grid-cols-3">
          <Stat n="2" label="ML algorithms implemented from scratch" />
          <Stat n="1,200" label="Synthetic training samples generated" />
          <Stat n="0" label="External ML libraries used" />
        </section>
      </main>
    </div>
  );
}

function DemoCard({
  to,
  icon,
  tag,
  title,
  desc,
  accent,
}: {
  to: string;
  icon: React.ReactNode;
  tag: string;
  title: string;
  desc: string;
  accent: string;
}) {
  return (
    <Link
      to={to}
      className={`group relative overflow-hidden rounded-2xl border border-slate-200 bg-gradient-to-br ${accent} p-7 transition hover:border-slate-300 hover:shadow-lg`}
    >
      <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-white shadow-sm">
        {icon}
      </div>
      <div className="mt-5 text-xs font-medium uppercase tracking-wide text-slate-500">{tag}</div>
      <h3 className="mt-1 text-2xl font-semibold text-slate-900">{title}</h3>
      <p className="mt-2 text-sm text-slate-600">{desc}</p>
      <div className="mt-5 inline-flex items-center gap-1 text-sm font-medium text-slate-900">
        Open demo <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
      </div>
    </Link>
  );
}

function Stat({ n, label }: { n: string; label: string }) {
  return (
    <div>
      <div className="text-3xl font-bold text-slate-900">{n}</div>
      <div className="mt-1 text-sm text-slate-600">{label}</div>
    </div>
  );
}
