// GTA VI trophy rarity forecaster.
//
// Reads every case file in the input directory (schema
// "makefun/gta6-trophies-case/v1") and writes, for each case, one forecast
// file with five quantiles of every trophy's earn rate at the case's target.
//
// Method, per trophy, in logit space (logit(p) = ln(p / (100 - p)) with the
// rate p in percent, clamped to [0.01, 99.99]):
//
//   * Fit a recency-weighted straight line of logit(rate) against
//     ln(1 + hours since the game's release) over the reads of the last
//     36 hours. Earn rates move as two things change: players finish
//     more of the game (rates up) and new players join the tracker
//     (rates down). The line in log-time captures the net drift of both
//     and bends as growth slows, so a short recent trend does not run
//     away when it is stretched over a week-long horizon.
//   * The median is the line read at the target, clamped so it can move
//     at most 4 logit units from the last read and never outside
//     [0.05%, 95%].
//   * The spread combines the fit's residual scale, the slope's
//     uncertainty stretched over the extrapolation, and a random-walk
//     term that grows with the horizon, floored by the median residual
//     scale across trophies so a trophy whose history happens to look
//     flat does not get a dishonestly narrow band.
//   * When a trophy's history is flat (no readable movement at all),
//     there is no trend to fit. Exophase refreshes its counts in jumps:
//     on launch night one refresh took the tracked base from 638 to
//     3,868 players and every rate but one fell. So a flat history gets
//     a dilution prior instead: the median shifts down by about 0.75
//     logit units with a wide spread, rather than claiming the rate
//     will sit exactly where it is.
//
// Everything is deterministic: no randomness, no clock reads, no network.
// Quantiles are the normal quantiles of the predictive logit distribution
// (-1.6449, -0.6745, 0, 0.6745, 1.6449), mapped back to percent.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const ZS = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

const logitOf = (pct: number): number => {
  const p = clamp(pct, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
};

const pctOf = (logit: number): number => {
  const p = 1 / (1 + Math.exp(-clamp(logit, -9.21, 9.21)));
  return clamp(p * 100, 0.01, 99.99);
};

const round4 = (v: number): number => Math.round(v * 10000) / 10000;

interface TrophyDef {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface HistoryRead {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: (number | null)[];
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: TrophyDef[];
  history: HistoryRead[];
}

// Weighted least squares of y on x. Returns slope, intercept, the residual
// standard deviation and the standard error of the slope, or null when the
// fit is degenerate (fewer than 3 points, or no spread in x).
function weightedLine(
  xs: number[],
  ys: number[],
  ws: number[],
): { slope: number; intercept: number; residSd: number; slopeSe: number } | null {
  const n = xs.length;
  if (n < 3) return null;
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sw += ws[i];
    sx += ws[i] * xs[i];
    sy += ws[i] * ys[i];
  }
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += ws[i] * dx * dx;
    sxy += ws[i] * dx * (ys[i] - my);
  }
  if (sxx <= 1e-12) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    rss += ws[i] * r * r;
  }
  // Weighted mean squared residual, corrected for the two fitted
  // parameters: this is the residual variance, not that divided by the
  // degrees of freedom a second time.
  const residVar = (rss / sw) * (n / Math.max(n - 2, 1));
  const residSd = Math.sqrt(Math.max(residVar, 0));
  const slopeSe = Math.sqrt(residVar / sxx);
  return { slope, intercept, residSd, slopeSe };
}

function forecastCase(c: CaseFile): { id: number; q: number[] }[] {
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const targetMs = Date.parse(c.target);
  const horizonDays = Math.max(c.horizon_minutes, 1) / 1440;
  const xTarget = Math.log(1 + (targetMs - releaseMs) / 3_600_000);
  const lastRead = c.history[c.history.length - 1];
  const lastAtMs = lastRead ? Date.parse(lastRead.at) : Date.parse(c.at);

  // First pass: per-trophy fits, collecting the residual scale for pooling.
  type Fit = {
    medianLogit: number;
    sigma: number;
    flat: boolean;
  };
  const fits: Fit[] = [];
  const residPool: number[] = [];

  for (let t = 0; t < c.trophies.length; t++) {
    const pts: { x: number; y: number; w: number }[] = [];
    for (const read of c.history) {
      const rate = read.rates[t];
      if (rate === null || rate === undefined) continue;
      const atMs = Date.parse(read.at);
      const hours = (atMs - releaseMs) / 3_600_000;
      if (hours < -1) continue;
      // Only the last 36 hours feed the trend; older reads are stale.
      const ageH = (lastAtMs - atMs) / 3_600_000;
      if (ageH > 36) continue;
      const w = Math.pow(0.5, ageH / 12);
      pts.push({ x: Math.log(1 + hours), y: logitOf(rate), w });
    }
    const lastRate = lastRead ? lastRead.rates[t] : null;
    const lastLogit = lastRate === null || lastRate === undefined ? logitOf(1) : logitOf(lastRate);
    if (pts.length === 0) {
      fits.push({ medianLogit: lastLogit, sigma: 1.2, flat: true });
      continue;
    }
    const ys = pts.map((p) => p.y);
    const spread = Math.max(...ys) - Math.min(...ys);
    let fit = weightedLine(
      pts.map((p) => p.x),
      ys,
      pts.map((p) => p.w),
    );
    // One round of outlier trimming: a single stale or partial read
    // (Exophase refreshes in jumps, and a read can catch the page
    // mid-refresh) should not tilt the slope. Refit without points
    // more than 3 residual scales away, when enough points remain.
    if (fit !== null && pts.length >= 6 && fit.residSd > 1e-9) {
      const kept = pts.filter(
        (p) => Math.abs(p.y - (fit!.intercept + fit!.slope * p.x)) <= 3 * fit!.residSd,
      );
      if (kept.length >= 3 && kept.length < pts.length) {
        const refit = weightedLine(
          kept.map((p) => p.x),
          kept.map((p) => p.y),
          kept.map((p) => p.w),
        );
        if (refit !== null) fit = refit;
      }
    }
    if (fit === null || spread < 0.05) {
      fits.push({ medianLogit: lastLogit, sigma: 0, flat: true });
      continue;
    }
    residPool.push(fit.residSd);
    const rawMedian = fit.intercept + fit.slope * xTarget;
    const medianLogit = clamp(rawMedian, lastLogit - 4, lastLogit + 4);
    const xMean = pts.reduce((s, p) => s + p.w * p.x, 0) / pts.reduce((s, p) => s + p.w, 0);
    const leverage = (xTarget - xMean) * (xTarget - xMean);
    const sigma = Math.sqrt(
      fit.residSd * fit.residSd +
        fit.slopeSe * fit.slopeSe * leverage +
        0.25 * horizonDays,
    );
    fits.push({ medianLogit, sigma, flat: false });
  }

  // Pooled residual scale: the median across trophies that had a fit.
  residPool.sort((a, b) => a - b);
  const pooledResid =
    residPool.length > 0 ? residPool[Math.floor(residPool.length / 2)] : 0.35;
  const residFloor = Math.max(pooledResid, 0.12);

  const rows: { id: number; q: number[] }[] = [];
  for (let t = 0; t < c.trophies.length; t++) {
    const trophy = c.trophies[t];
    const fit = fits[t];
    let medianLogit: number;
    let sigma: number;
    if (fit.flat) {
      // Dilution prior: a refresh is coming sooner or later, and it
      // pushes rates down as the tracked base grows. Shift the median
      // down and keep the band honest and wide.
      medianLogit = clamp(fit.medianLogit - 0.75, logitOf(0.05), logitOf(95));
      sigma = 0.9 + 0.25 * Math.sqrt(horizonDays);
    } else {
      medianLogit = clamp(fit.medianLogit, logitOf(0.05), logitOf(95));
      sigma = clamp(Math.max(fit.sigma, residFloor), 0.3, 2.0);
    }
    const q: number[] = [];
    let prev = -Infinity;
    for (const z of ZS) {
      let v = round4(pctOf(medianLogit + z * sigma));
      if (v < prev) v = prev;
      if (v > 99.99) v = 99.99;
      q.push(v);
      prev = v;
    }
    rows.push({ id: trophy.id, q });
  }
  rows.sort((a, b) => a.id - b.id);
  return rows;
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  mkdirSync(outputDir, { recursive: true });
  const names = readdirSync(inputDir)
    .filter((n) => n.endsWith(".in"))
    .sort();
  for (const name of names) {
    const id = name.slice(0, -3);
    const c = JSON.parse(readFileSync(join(inputDir, name), "utf8")) as CaseFile;
    const rows = forecastCase(c);
    const out = JSON.stringify({ trophies: rows });
    writeFileSync(join(outputDir, `${id}.out`), out, "utf8");
  }
}

main();
