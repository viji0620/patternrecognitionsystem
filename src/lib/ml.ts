// Lightweight ML algorithms implemented from scratch (no deps).
// Includes: Logistic Regression (binary classifier) and k-Nearest Neighbors.

export type Vec = number[];

// ---------------- Logistic Regression ----------------
export class LogisticRegression {
  weights: Vec = [];
  bias = 0;
  lr: number;
  epochs: number;
  losses: number[] = [];

  constructor(lr = 0.1, epochs = 300) {
    this.lr = lr;
    this.epochs = epochs;
  }

  private sigmoid(z: number) {
    return 1 / (1 + Math.exp(-z));
  }

  predictProba(x: Vec): number {
    let z = this.bias;
    for (let i = 0; i < x.length; i++) z += this.weights[i] * x[i];
    return this.sigmoid(z);
  }

  predict(x: Vec): 0 | 1 {
    return this.predictProba(x) >= 0.5 ? 1 : 0;
  }

  fit(X: Vec[], y: number[]) {
    const n = X.length;
    const d = X[0].length;
    this.weights = new Array(d).fill(0);
    this.bias = 0;
    this.losses = [];

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const gradW = new Array(d).fill(0);
      let gradB = 0;
      let loss = 0;
      for (let i = 0; i < n; i++) {
        const p = this.predictProba(X[i]);
        const err = p - y[i];
        for (let j = 0; j < d; j++) gradW[j] += err * X[i][j];
        gradB += err;
        const eps = 1e-9;
        loss += -(y[i] * Math.log(p + eps) + (1 - y[i]) * Math.log(1 - p + eps));
      }
      for (let j = 0; j < d; j++) this.weights[j] -= (this.lr * gradW[j]) / n;
      this.bias -= (this.lr * gradB) / n;
      this.losses.push(loss / n);
    }
  }

  accuracy(X: Vec[], y: number[]): number {
    let c = 0;
    for (let i = 0; i < X.length; i++) if (this.predict(X[i]) === y[i]) c++;
    return c / X.length;
  }
}

// ---------------- k-Nearest Neighbors ----------------
export class KNN {
  k: number;
  X: Vec[] = [];
  y: number[] = [];
  labels: string[] = [];

  constructor(k = 5) {
    this.k = k;
  }

  fit(X: Vec[], y: number[], labels: string[]) {
    this.X = X;
    this.y = y;
    this.labels = labels;
  }

  private dist(a: Vec, b: Vec) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return Math.sqrt(s);
  }

  predict(x: Vec): { label: string; confidence: number; votes: Record<string, number> } {
    const dists = this.X.map((xi, i) => ({ d: this.dist(xi, x), y: this.y[i] }));
    dists.sort((a, b) => a.d - b.d);
    const top = dists.slice(0, this.k);
    const votes: Record<string, number> = {};
    for (const t of top) {
      const lbl = this.labels[t.y];
      votes[lbl] = (votes[lbl] || 0) + 1;
    }
    let best = "";
    let bestN = -1;
    for (const [lbl, n] of Object.entries(votes)) {
      if (n > bestN) {
        best = lbl;
        bestN = n;
      }
    }
    return { label: best, confidence: bestN / this.k, votes };
  }

  accuracy(X: Vec[], y: number[]): number {
    let c = 0;
    for (let i = 0; i < X.length; i++) {
      const pred = this.predict(X[i]);
      if (pred.label === this.labels[y[i]]) c++;
    }
    return c / X.length;
  }
}

// ---------------- Feature scaling ----------------
export function standardize(X: Vec[]): { X: Vec[]; mean: Vec; std: Vec } {
  const d = X[0].length;
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  for (const row of X) for (let j = 0; j < d; j++) mean[j] += row[j];
  for (let j = 0; j < d; j++) mean[j] /= X.length;
  for (const row of X) for (let j = 0; j < d; j++) std[j] += (row[j] - mean[j]) ** 2;
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / X.length) || 1;
  const scaled = X.map((row) => row.map((v, j) => (v - mean[j]) / std[j]));
  return { X: scaled, mean, std };
}

export function applyStandardize(x: Vec, mean: Vec, std: Vec): Vec {
  return x.map((v, j) => (v - mean[j]) / std[j]);
}

// ---------------- Synthetic datasets ----------------
// Seeded RNG so accuracy is reproducible.
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng: () => number, mu: number, sigma: number) {
  const u = 1 - rng();
  const v = rng();
  return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Fraud detection dataset. Features:
// [amount ($), hour (0-23), distanceFromHome (km), txPerHour, foreignFlag (0/1)]
// Label: 1 = fraud, 0 = legit
export function makeFraudDataset(n = 600, seed = 42) {
  const rng = mulberry32(seed);
  const X: Vec[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const fraud = rng() < 0.3 ? 1 : 0;
    if (fraud) {
      X.push([
        Math.max(5, gauss(rng, 850, 400)),
        Math.floor(gauss(rng, 2, 3) + 24) % 24, // odd hours
        Math.max(0, gauss(rng, 1200, 600)),
        Math.max(1, gauss(rng, 6, 2)),
        rng() < 0.7 ? 1 : 0,
      ]);
      y.push(1);
    } else {
      X.push([
        Math.max(1, gauss(rng, 60, 35)),
        Math.floor(gauss(rng, 14, 4) + 24) % 24,
        Math.max(0, gauss(rng, 15, 12)),
        Math.max(1, gauss(rng, 1.5, 0.7)),
        rng() < 0.05 ? 1 : 0,
      ]);
      y.push(0);
    }
  }
  return { X, y };
}

// Heartbeat/medical dataset. Features describe an ECG-like beat:
// [heartRate (bpm), qrsWidth (ms), qtInterval (ms), stDeviation (mV), beatVariability]
// Classes: 0 normal, 1 bradycardia (slow), 2 tachycardia (fast), 3 arrhythmia
export const HEART_LABELS = ["Normal", "Bradycardia", "Tachycardia", "Arrhythmia"];

export function makeHeartbeatDataset(n = 600, seed = 7) {
  const rng = mulberry32(seed);
  const X: Vec[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = Math.floor(rng() * 4);
    let row: Vec;
    if (cls === 0) {
      row = [gauss(rng, 72, 6), gauss(rng, 90, 8), gauss(rng, 400, 15), gauss(rng, 0, 0.05), gauss(rng, 0.04, 0.01)];
    } else if (cls === 1) {
      row = [gauss(rng, 48, 5), gauss(rng, 95, 10), gauss(rng, 430, 20), gauss(rng, 0, 0.08), gauss(rng, 0.05, 0.015)];
    } else if (cls === 2) {
      row = [gauss(rng, 130, 10), gauss(rng, 85, 8), gauss(rng, 340, 18), gauss(rng, 0.05, 0.08), gauss(rng, 0.06, 0.02)];
    } else {
      row = [gauss(rng, 85, 25), gauss(rng, 130, 18), gauss(rng, 450, 35), gauss(rng, 0.25, 0.15), gauss(rng, 0.2, 0.06)];
    }
    X.push(row.map((v) => Math.max(0, v)));
    y.push(cls);
  }
  return { X, y };
}

export function trainTestSplit(X: Vec[], y: number[], testRatio = 0.2, seed = 1) {
  const rng = mulberry32(seed);
  const idx = X.map((_, i) => i).sort(() => rng() - 0.5);
  const cut = Math.floor(X.length * (1 - testRatio));
  const tr = idx.slice(0, cut);
  const te = idx.slice(cut);
  return {
    XTrain: tr.map((i) => X[i]),
    yTrain: tr.map((i) => y[i]),
    XTest: te.map((i) => X[i]),
    yTest: te.map((i) => y[i]),
  };
}

// ---------------- Shape / image patterns ----------------
export const SHAPE_LABELS = ["Star", "Square", "Triangle", "Line"];

export type ShapeFeatures = {
  fillRatio: number;       // filled pixels / bbox area
  aspect: number;          // min(w,h)/max(w,h)  (0..1)
  circularity: number;     // 4πA / P²
  vSym: number;            // 0..1
  hSym: number;            // 0..1
  cornerScore: number;     // approx corners via centroid distance variance
};

export function featuresToVec(f: ShapeFeatures): Vec {
  return [f.fillRatio, f.aspect, f.circularity, f.vSym, f.hSym, f.cornerScore];
}

// Extract features from a binary mask (1 = ink). w, h = mask dimensions.
export function extractShapeFeatures(mask: Uint8Array, w: number, h: number): ShapeFeatures | null {
  let minX = w, minY = h, maxX = -1, maxY = -1, count = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x]) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        count++;
      }
    }
  }
  if (count < 12 || maxX < 0) return null;
  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const bboxArea = bw * bh;
  const fillRatio = count / bboxArea;
  const aspect = Math.min(bw, bh) / Math.max(bw, bh);

  // Perimeter: count ink pixels that have at least one non-ink neighbor (4-connected).
  let perim = 0;
  let cx = 0, cy = 0;
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      cx += x; cy += y;
      const up = y > 0 ? mask[i - w] : 0;
      const dn = y < h - 1 ? mask[i + w] : 0;
      const lf = x > 0 ? mask[i - 1] : 0;
      const rt = x < w - 1 ? mask[i + 1] : 0;
      if (!up || !dn || !lf || !rt) perim++;
    }
  }
  cx /= count; cy /= count;
  const area = count;
  const circularity = Math.min(1, (4 * Math.PI * area) / (perim * perim || 1));

  // Symmetry: compare halves about bbox center.
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  let hMatch = 0, hTotal = 0, vMatch = 0, vTotal = 0;
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const mx = Math.round(2 * midX - x);
      const my = Math.round(2 * midY - y);
      if (mx >= 0 && mx < w) {
        const a = mask[y * w + x];
        const b = mask[y * w + mx];
        hTotal++;
        if (a === b) hMatch++;
      }
      if (my >= 0 && my < h) {
        const a = mask[y * w + x];
        const b = mask[my * w + x];
        vTotal++;
        if (a === b) vMatch++;
      }
    }
  }
  const hSym = hTotal ? hMatch / hTotal : 0;
  const vSym = vTotal ? vMatch / vTotal : 0;

  // Corner score: variance of distance from centroid to boundary pixels, normalized.
  let n = 0, sum = 0, sumSq = 0;
  const norm = Math.max(bw, bh) / 2 || 1;
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      const up = y > 0 ? mask[i - w] : 0;
      const dn = y < h - 1 ? mask[i + w] : 0;
      const lf = x > 0 ? mask[i - 1] : 0;
      const rt = x < w - 1 ? mask[i + 1] : 0;
      if (up && dn && lf && rt) continue;
      const d = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / norm;
      sum += d; sumSq += d * d; n++;
    }
  }
  const meanD = n ? sum / n : 0;
  const varD = n ? sumSq / n - meanD * meanD : 0;
  const cornerScore = Math.min(1, Math.sqrt(Math.max(0, varD)) * 3);

  return { fillRatio, aspect, circularity, vSym, hSym, cornerScore };
}

// Rasterize a canonical shape outline into a binary mask. cls: 0=Circle, 1=Square, 2=Triangle, 3=Line.
// Returns the same kind of mask the live canvas extractor produces, so the model trains
// on the actual feature distribution it will see at inference time.
export function rasterizeShape(cls: number, w: number, h: number, rng: () => number, thickness = 3): Uint8Array {
  const mask = new Uint8Array(w * h);
  const cx = w / 2 + (rng() - 0.5) * w * 0.1;
  const cy = h / 2 + (rng() - 0.5) * h * 0.1;
  const size = Math.min(w, h) * (0.45 + rng() * 0.35);
  const rot = (rng() - 0.5) * Math.PI * 0.5;
  const cos = Math.cos(rot), sin = Math.sin(rot);

  const plotLine = (x0: number, y0: number, x1: number, y1: number) => {
    const steps = Math.ceil(Math.hypot(x1 - x0, y1 - y0)) + 1;
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const px = x0 + (x1 - x0) * t;
      const py = y0 + (y1 - y0) * t;
      stamp(mask, w, h, px, py, thickness);
    }
  };

  if (cls === 0) {
    // Circle outline (slightly noisy radius).
    const r = size / 2;
    const segs = 64;
    let prev: [number, number] | null = null;
    for (let i = 0; i <= segs; i++) {
      const a = (i / segs) * Math.PI * 2;
      const rr = r * (1 + (rng() - 0.5) * 0.04);
      const x = cx + Math.cos(a) * rr;
      const y = cy + Math.sin(a) * rr;
      if (prev) plotLine(prev[0], prev[1], x, y);
      prev = [x, y];
    }
  } else if (cls === 1) {
    // Square outline (rotated, aspect jitter).
    const wH = size / 2;
    const hH = (size / 2) * (0.85 + rng() * 0.3);
    const pts: [number, number][] = [
      [-wH, -hH], [wH, -hH], [wH, hH], [-wH, hH], [-wH, -hH],
    ].map(([x, y]) => [cx + x * cos - y * sin, cy + x * sin + y * cos]);
    for (let i = 0; i < pts.length - 1; i++) plotLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]);
  } else if (cls === 2) {
    // Triangle outline (isoceles, rotated).
    const r = size / 2;
    const angles = [-Math.PI / 2, Math.PI / 6, (5 * Math.PI) / 6];
    const pts = angles.map((a) => {
      const x = Math.cos(a) * r;
      const y = Math.sin(a) * r;
      return [cx + x * cos - y * sin, cy + x * sin + y * cos] as [number, number];
    });
    plotLine(pts[0][0], pts[0][1], pts[1][0], pts[1][1]);
    plotLine(pts[1][0], pts[1][1], pts[2][0], pts[2][1]);
    plotLine(pts[2][0], pts[2][1], pts[0][0], pts[0][1]);
  } else {
    // Line segment.
    const half = size / 2;
    const x0 = cx - cos * half;
    const y0 = cy - sin * half;
    const x1 = cx + cos * half;
    const y1 = cy + sin * half;
    plotLine(x0, y0, x1, y1);
  }
  return mask;
}

function stamp(mask: Uint8Array, w: number, h: number, px: number, py: number, r: number) {
  const x0 = Math.max(0, Math.floor(px - r));
  const x1 = Math.min(w - 1, Math.ceil(px + r));
  const y0 = Math.max(0, Math.floor(py - r));
  const y1 = Math.min(h - 1, Math.ceil(py + r));
  const r2 = r * r;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - px, dy = y - py;
      if (dx * dx + dy * dy <= r2) mask[y * w + x] = 1;
    }
  }
}

// Build a feature-vector dataset by rasterizing real shapes and extracting features.
// Trains across a wide thickness range so the model handles both thin pencil
// strokes and chunky marker strokes that users actually draw with.
export function makeShapeDataset(n = 800, seed = 11, w = 280, h = 280) {
  const rng = mulberry32(seed);
  const X: Vec[] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = i % 4; // balanced classes
    // Sweep thickness proportional to canvas so small + large brushes both
    // appear in the training distribution.
    const base = Math.max(2, Math.min(w, h) * 0.012);
    const thickness = base + rng() * Math.max(4, Math.min(w, h) * 0.05);
    const mask = rasterizeShape(cls, w, h, rng, thickness);
    const f = extractShapeFeatures(mask, w, h);
    if (!f) continue;
    X.push(featuresToVec(f));
    y.push(cls);
  }
  return { X, y };
}

