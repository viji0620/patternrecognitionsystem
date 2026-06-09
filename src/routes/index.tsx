import { createFileRoute, Link } from "@tanstack/react-router";
import { Brain, Activity, ShieldAlert, ArrowUpRight, Sparkles, Cpu, Network, Zap, Eye } from "lucide-react";
import { NeuralCanvas } from "@/components/NeuralCanvas";
import { AnimatedNumber } from "@/components/AnimatedNumber";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background layers */}
      <div className="pointer-events-none fixed inset-0 -z-10 neural-grid animate-grid-shift opacity-40" />
      <div className="pointer-events-none fixed -top-40 -left-40 -z-10 h-[480px] w-[480px] rounded-full bg-primary/30 blur-3xl animate-blob" />
      <div className="pointer-events-none fixed -bottom-40 -right-40 -z-10 h-[520px] w-[520px] rounded-full bg-accent/25 blur-3xl animate-blob" style={{ animationDelay: "-7s" }} />

      {/* Nav */}
      <header className="relative z-20 mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-2.5">
          <LogoMark />
          <span className="font-mono text-sm tracking-tight text-foreground">
            patternlab<span className="text-primary">.ml</span>
          </span>
        </div>
        <nav className="hidden items-center gap-1 rounded-full border border-border bg-surface/60 px-1.5 py-1.5 text-sm backdrop-blur md:flex">
          <NavPill to="/">Overview</NavPill>
          <NavPill to="/fraud">Fraud engine</NavPill>
          <NavPill to="/heartbeat">ECG classifier</NavPill>
          <NavPill to="/shapes">Shape vision</NavPill>
        </nav>
        <div className="hidden items-center gap-2 text-xs font-mono text-muted-foreground md:flex">
          <span className="relative flex h-2 w-2">
            <span className="absolute inset-0 rounded-full bg-success animate-pulse-ring" />
            <span className="relative rounded-full bg-success h-2 w-2" />
          </span>
          models online
        </div>
      </header>

      {/* Hero */}
      <section className="relative z-10 mx-auto grid max-w-7xl gap-12 px-6 pt-12 pb-24 lg:grid-cols-[1.1fr_1fr] lg:items-center">
        <div className="animate-fade-up">
          <div className="inline-flex items-center gap-2 rounded-full border border-border bg-surface/60 px-3 py-1 text-xs font-mono uppercase tracking-widest text-muted-foreground backdrop-blur">
            <Sparkles className="h-3 w-3 text-primary" />
            Live pattern recognition · trained in your browser
          </div>
          <h1 className="mt-6 font-display text-5xl font-bold leading-[1.05] tracking-tight text-foreground sm:text-6xl lg:text-7xl">
            Teach machines to <span className="text-gradient">see the patterns</span> humans miss.
          </h1>
          <p className="mt-6 max-w-xl text-lg text-muted-foreground">
            Two production-style ML pipelines — behavioral fraud detection and
            medical ECG classification — built from raw algorithms, trained
            live on synthetic data, and visualized as they think.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link
              to="/fraud"
              className="group relative inline-flex items-center gap-2 overflow-hidden rounded-full bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-[var(--shadow-glow-primary)] transition hover:scale-[1.02] active:scale-100"
            >
              <span className="absolute inset-0 bg-[linear-gradient(120deg,transparent_30%,oklch(1_0_0/0.3)_50%,transparent_70%)] bg-[length:200%_100%] animate-shimmer" />
              <ShieldAlert className="relative h-4 w-4" />
              <span className="relative">Run fraud engine</span>
              <ArrowUpRight className="relative h-4 w-4 transition group-hover:rotate-45" />
            </Link>
            <Link
              to="/heartbeat"
              className="inline-flex items-center gap-2 rounded-full border border-border-strong bg-surface/60 px-6 py-3 text-sm font-semibold text-foreground backdrop-blur transition hover:border-primary hover:bg-surface"
            >
              <Activity className="h-4 w-4 text-success" />
              Open ECG classifier
            </Link>
            <Link
              to="/shapes"
              className="inline-flex items-center gap-2 rounded-full border border-border-strong bg-surface/60 px-6 py-3 text-sm font-semibold text-foreground backdrop-blur transition hover:border-accent hover:bg-surface"
            >
              <Eye className="h-4 w-4 text-accent" />
              Launch shape vision
            </Link>
          </div>

          {/* Stats */}
          <div className="mt-12 grid max-w-lg grid-cols-3 gap-4">
            {[
              { v: 3, l: "ML pipelines", d: 0 },
              { v: 2400, l: "training samples", d: 0 },
              { v: 0, l: "external deps", d: 0 },
            ].map((s, i) => (
              <div
                key={i}
                className="rounded-2xl border border-border bg-surface/40 p-4 backdrop-blur animate-fade-up"
                style={{ animationDelay: `${0.2 + i * 0.08}s` }}
              >
                <div className="font-mono text-3xl font-bold text-foreground">
                  <AnimatedNumber value={s.v} decimals={s.d} />
                </div>
                <div className="mt-1 text-xs uppercase tracking-wider text-muted-foreground">{s.l}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: neural orb */}
        <div className="relative h-[420px] animate-fade-in lg:h-[560px]">
          <div className="absolute inset-0 rounded-3xl border border-border bg-surface/30 backdrop-blur-xl overflow-hidden shimmer-border">
            <NeuralCanvas className="absolute inset-0 h-full w-full" />
            {/* Central core */}
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
              <div className="relative h-32 w-32">
                <div className="absolute inset-0 rounded-full bg-primary/30 blur-2xl animate-pulse-glow" />
                <div className="absolute inset-4 rounded-full bg-gradient-to-br from-primary to-accent animate-float" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Brain className="h-10 w-10 text-primary-foreground drop-shadow" />
                </div>
                {/* Orbiting dots */}
                {[0, 1, 2].map((i) => (
                  <div
                    key={i}
                    className="absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary-glow shadow-[0_0_12px_oklch(0.85_0.18_195)]"
                    style={{
                      animation: "orbit 14s linear infinite",
                      animationDelay: `${-i * 4.6}s`,
                    }}
                  />
                ))}
              </div>
            </div>
            {/* corner labels */}
            <div className="absolute left-4 top-4 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              neural::topology
            </div>
            <div className="absolute right-4 top-4 font-mono text-[10px] uppercase tracking-widest text-primary">
              ● live
            </div>
            <div className="absolute bottom-4 left-4 right-4 flex justify-between font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              <span>nodes: dynamic</span>
              <span>fps: 60</span>
            </div>
          </div>
        </div>
      </section>

      {/* Models grid */}
      <section className="relative z-10 mx-auto max-w-7xl px-6 pb-24">
        <div className="mb-8 flex items-end justify-between">
          <div>
            <div className="font-mono text-xs uppercase tracking-widest text-primary">// 02 — models</div>
            <h2 className="mt-2 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
              Three minds. Three patterns.
            </h2>
          </div>
          <div className="hidden font-mono text-xs text-muted-foreground sm:block">
            all train on mount → ~80ms
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          <ModelCard
            to="/fraud"
            tag="behavioral"
            algo="LOGISTIC REGRESSION"
            icon={<ShieldAlert className="h-5 w-5" />}
            title="Transaction fraud engine"
            desc="Learns the boundary between ordinary spending and anomalous activity from amount, timing, location, and velocity signals."
            features={["binary classifier", "gradient descent", "live probability"]}
            color="primary"
            delay={0}
          />
          <ModelCard
            to="/heartbeat"
            tag="medical"
            algo="K-NEAREST NEIGHBORS"
            icon={<Activity className="h-5 w-5" />}
            title="ECG heartbeat classifier"
            desc="Compares an incoming beat against a library of labeled examples and votes on the most likely cardiac condition."
            features={["4-class classifier", "k = 7 neighbors", "live ECG visual"]}
            color="accent"
            delay={120}
          />
          <ModelCard
            to="/shapes"
            tag="image"
            algo="K-NN + FEATURE EXTRACTION"
            icon={<Eye className="h-5 w-5" />}
            title="Shape vision recognizer"
            desc="Reads pixels off a canvas, distills them into geometric features — circularity, symmetry, corner energy — and names what you drew."
            features={["draw to classify", "6 features", "live binary mask"]}
            color="primary"
            delay={240}
          />
        </div>
      </section>

      {/* How it works */}
      <section className="relative z-10 mx-auto max-w-7xl px-6 pb-32">
        <div className="font-mono text-xs uppercase tracking-widest text-primary">// 03 — pipeline</div>
        <h2 className="mt-2 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
          How a pattern becomes a prediction.
        </h2>
        <div className="mt-10 grid gap-6 md:grid-cols-4">
          {[
            { i: <Network className="h-5 w-5" />, t: "Generate", d: "Seeded synthetic data with realistic class distributions." },
            { i: <Cpu className="h-5 w-5" />, t: "Standardize", d: "Features rescaled to zero mean and unit variance." },
            { i: <Brain className="h-5 w-5" />, t: "Train", d: "Algorithms fit on 80% of samples, evaluated on the rest." },
            { i: <Zap className="h-5 w-5" />, t: "Predict", d: "Your inputs are scored instantly with confidence." },
          ].map((s, i) => (
            <div
              key={i}
              className="group relative rounded-2xl border border-border bg-surface/40 p-6 backdrop-blur transition hover:border-primary/60 animate-fade-up"
              style={{ animationDelay: `${i * 0.08}s` }}
            >
              <div className="absolute right-4 top-4 font-mono text-xs text-muted-foreground">
                0{i + 1}
              </div>
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/15 text-primary transition group-hover:bg-primary group-hover:text-primary-foreground">
                {s.i}
              </div>
              <div className="mt-4 font-semibold text-foreground">{s.t}</div>
              <div className="mt-1.5 text-sm text-muted-foreground">{s.d}</div>
            </div>
          ))}
        </div>
      </section>

      <footer className="relative z-10 border-t border-border">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6 font-mono text-xs text-muted-foreground">
          <span>built with vanilla TS · no ML libraries</span>
          <span>patternlab.ml — v1.0</span>
        </div>
      </footer>
    </div>
  );
}

function NavPill({ to, children }: { to: string; children: React.ReactNode }) {
  return (
    <Link
      to={to}
      className="rounded-full px-4 py-1.5 text-muted-foreground transition hover:bg-surface-2 hover:text-foreground"
      activeProps={{ className: "bg-primary text-primary-foreground hover:bg-primary hover:text-primary-foreground" }}
      activeOptions={{ exact: true }}
    >
      {children}
    </Link>
  );
}

function LogoMark() {
  return (
    <div className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent shadow-[var(--shadow-glow-primary)]">
      <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-primary to-accent animate-pulse-glow opacity-60" />
      <Brain className="relative h-4 w-4 text-primary-foreground" />
    </div>
  );
}

function ModelCard({
  to, tag, algo, icon, title, desc, features, color, delay,
}: {
  to: string; tag: string; algo: string; icon: React.ReactNode; title: string; desc: string; features: string[]; color: "primary" | "accent"; delay: number;
}) {
  const glow = color === "primary" ? "shadow-[var(--shadow-glow-primary)]" : "shadow-[var(--shadow-glow-accent)]";
  const text = color === "primary" ? "text-primary" : "text-accent";
  const bg = color === "primary" ? "bg-primary/10" : "bg-accent/10";

  return (
    <Link
      to={to}
      className="group relative overflow-hidden rounded-3xl border border-border bg-surface/50 p-8 backdrop-blur transition hover:border-border-strong hover:-translate-y-1 animate-fade-up"
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* hover gradient */}
      <div className={`pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full ${bg} blur-3xl opacity-60 transition group-hover:opacity-100`} />
      {/* scanline */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent opacity-0 transition group-hover:opacity-100" />

      <div className="relative flex items-start justify-between">
        <div className={`flex h-11 w-11 items-center justify-center rounded-xl ${bg} ${text} ${glow}`}>
          {icon}
        </div>
        <ArrowUpRight className="h-5 w-5 text-muted-foreground transition group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-foreground" />
      </div>

      <div className="relative mt-6 flex items-center gap-2 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        <span className="rounded-full bg-surface-2 px-2 py-0.5">{tag}</span>
        <span className={text}>{algo}</span>
      </div>
      <h3 className="relative mt-3 text-2xl font-bold text-foreground">{title}</h3>
      <p className="relative mt-2 text-sm text-muted-foreground">{desc}</p>

      <div className="relative mt-6 flex flex-wrap gap-2">
        {features.map((f) => (
          <span key={f} className="rounded-full border border-border bg-surface-2/60 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
            {f}
          </span>
        ))}
      </div>
    </Link>
  );
}
