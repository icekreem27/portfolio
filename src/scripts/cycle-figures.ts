// Client side for the cycle hire figures: segmented toggles, three line charts drawn at the container width, and the cost explorer.
const $ = <T extends Element = HTMLElement>(s: string, r: ParentNode = document) => r.querySelector(s) as T | null;
const $$ = <T extends Element = HTMLElement>(s: string, r: ParentNode = document) => [...r.querySelectorAll(s)] as T[];
const json = (id: string) => JSON.parse($(`#${id}`)!.textContent!);
const NS = 'http://www.w3.org/2000/svg';
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

type Pt = [number, number];
const M = { l: 40, r: 14, t: 22, b: 30 };

function frame(host: HTMLElement, h = 260) {
  const w = Math.max(host.clientWidth, 280);
  return { w, h: w < 520 ? Math.round(h * 0.88) : h };
}
const line = (pts: Pt[]) => pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('');
const stepPts = (xs: (t: number) => number, ys: (v: number) => number, arr: number[]) =>
  arr.flatMap((v, t) => [[xs(t), ys(v)], [xs(t + 1), ys(v)]] as Pt[]);

function axes(w: number, h: number, ys: (v: number) => number, ticks: number[], fmt: (v: number) => string) {
  return ticks
    .map((v) => `<line class="grid" x1="${M.l}" x2="${w - M.r}" y1="${ys(v)}" y2="${ys(v)}"/><text x="${M.l - 8}" y="${ys(v) + 4}" text-anchor="end">${fmt(v)}</text>`)
    .join('');
}
const hourAxis = (w: number, h: number, xs: (v: number) => number) =>
  [0, 4, 8, 12, 16, 20, 23].map((v) => `<text x="${xs(v)}" y="${h - 8}" text-anchor="middle">${String(v).padStart(2, '0')}:00</text>`).join('');

/* ---------- toggles ---------- */
document.addEventListener('click', (e) => {
  const b = (e.target as Element).closest('.seg button') as HTMLElement | null;
  if (!b) return;
  const seg = b.parentElement!;
  const fig = seg.closest('.fig') as HTMLElement;
  $$('button', seg).forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  fig.dataset[seg.dataset.key!] = b.dataset.v!;
  fig.dispatchEvent(new CustomEvent('fig:change', { detail: { key: seg.dataset.key } }));
});

function onSize(host: HTMLElement, draw: () => void) {
  let w = 0;
  const ro = new ResizeObserver(() => { if (host.clientWidth !== w) { w = host.clientWidth; draw(); } });
  ro.observe(host);
  host.closest('.fig')!.addEventListener('fig:change', draw);
}

/* ---------- demand profile ---------- */
function demandChart() {
  const host = $('[data-chart="demand"]');
  if (!host) return;
  const d = json('data-demand');
  const S: { key: string; label: string; cls: string }[] = [
    { key: 'morning docks, weekday', label: 'Morning docks on weekdays', cls: 'c-acc' },
    { key: 'weekend docks, weekend', label: 'Weekend docks on weekends', cls: 'c-acc2' },
    { key: 'morning docks, weekend', label: 'Morning docks on weekends', cls: 'c-grey' },
    { key: 'weekend docks, weekday', label: 'Weekend docks on weekdays', cls: 'c-grey2' },
  ];
  const draw = () => {
    const { w, h } = frame(host);
    const max = Math.ceil(Math.max(...S.flatMap((s) => d.series[s.key] as number[])) / 2) * 2;
    const xs = (v: number) => M.l + (v / 23) * (w - M.l - M.r);
    const ys = (v: number) => h - M.b - (v / max) * (h - M.b - M.t);
    const ticks = Array.from({ length: max / 4 + 1 }, (_, i) => i * 4);
    const paths = [...S].reverse().map((s) => `<path class="${s.cls}" d="${line((d.series[s.key] as number[]).map((v, i) => [xs(d.hours[i]), ys(v)]))}"/>`).join('');
    const peak = d.series[S[0].key][8];
    host.innerHTML = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="Average departures per dock-hour by hour of day. Morning docks on weekdays peak at 8am at about ${peak.toFixed(0)} an hour.">
      ${axes(w, h, ys, ticks, (v) => String(v))}${hourAxis(w, h, xs)}${paths}
      <circle class="pk" cx="${xs(8)}" cy="${ys(peak)}" r="4"/>
      <text class="note" x="${xs(8) + 10}" y="${ys(peak) + 4}">8am peak, about ${peak.toFixed(0)} an hour</text>
      <text class="axis-title" x="${M.l}" y="${M.t - 3}">Departures per dock per hour</text></svg>
      <div class="legend">${S.map((s) => `<span><i class="lg ${s.cls}"></i>${s.label}</span>`).join('')}</div>`;
  };
  onSize(host, draw);
}

/* ---------- coverage by hour ---------- */
function coverageChart() {
  const host = $('[data-chart="coverage"]');
  if (!host) return;
  const d = json('data-coverage');
  const fig = host.closest('.fig') as HTMLElement;
  const draw = () => {
    const { w, h } = frame(host, 240);
    const lo = 0.6, hi = 1;
    const xs = (v: number) => M.l + (v / 23) * (w - M.l - M.r);
    const ys = (v: number) => h - M.b - ((v - lo) / (hi - lo)) * (h - M.b - M.t);
    const k = fig.dataset.period!;
    const pts = d[k].hours.map((r: any) => [xs(r.hour), ys(r.inside_band)] as Pt);
    host.innerHTML = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="Share of hours where the real count landed inside the forecast range, by hour of day, against an 80% target">
      ${axes(w, h, ys, [0.6, 0.7, 0.8, 0.9, 1], (v) => `${Math.round(v * 100)}%`)}${hourAxis(w, h, xs)}
      <line class="target" x1="${M.l}" x2="${w - M.r}" y1="${ys(0.8)}" y2="${ys(0.8)}"/>
      <text class="note" x="${w - M.r}" y="${ys(0.8) - 6}" text-anchor="end">80% target</text>
      <path class="c-acc" d="${line(pts)}"/>${pts.map((p: Pt) => `<circle class="pk" cx="${p[0]}" cy="${p[1]}" r="2.5"/>`).join('')}</svg>`;
  };
  onSize(host, draw);
}

/* ---------- forecast fan ---------- */
// Busy dock with the lowest error relative to its volume in the strike week, chosen to show the model at its best.
const DEFAULT_STATION = '001011';

function fanChart() {
  const root = $('[data-fan]');
  if (!root) return;
  const fig = root.closest('.fig') as HTMLElement;
  const host = $('[data-chart="fan"]', root)!;
  const sel = $<HTMLSelectElement>('[data-fan-station]', root)!;
  const note = $('[data-fan-note]', root)!;
  let data: any = null;
  let week = 'strike-week';
  let station = '';

  const draw = () => {
    if (!data) return;
    const wk = data.weeks[week];
    const s = wk.stations[station];
    const st = data.stations.find((x: any) => x.id === station);
    const { w, h } = frame(host, 300);
    const n = 168;
    const maxV = Math.max(4, ...s.q90, ...s.actual.filter((v: number | null) => v != null));
    const yMax = Math.ceil(maxV / 2) * 2;
    const xs = (t: number) => M.l + (t / n) * (w - M.l - M.r);
    const ys = (v: number) => h - M.b - (v / yMax) * (h - M.b - M.t);
    const step = Math.max(2, Math.ceil(yMax / 5 / 2) * 2);
    const ticks = Array.from({ length: Math.floor(yMax / step) + 1 }, (_, i) => i * step);
    const top = stepPts(xs, ys, s.q90);
    const bot = stepPts(xs, ys, s.q10).reverse();
    const strike = (wk.strike_hours as boolean[]).map((v, t) => (v ? `<rect class="strike" x="${xs(t)}" y="${M.t}" width="${xs(t + 1) - xs(t) + 0.5}" height="${h - M.b - M.t}"/>` : '')).join('');
    const day0 = new Date(wk.first_hour_london.slice(0, 10) + 'T12:00:00Z');
    const days = Array.from({ length: 7 }, (_, i) => {
      const dt = new Date(day0.getTime() + i * 864e5);
      const lab = dt.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', timeZone: 'UTC' });
      return `<line class="grid v" x1="${xs(i * 24)}" x2="${xs(i * 24)}" y1="${M.t}" y2="${h - M.b}"/><text x="${xs(i * 24 + 12)}" y="${h - 8}" text-anchor="middle">${w < 460 ? lab.slice(0, 1) + ' ' + dt.getUTCDate() : lab}</text>`;
    }).join('');
    const dots = (s.actual as (number | null)[]).map((v, t) => (v == null ? '' : `<circle class="dot" cx="${xs(t + 0.5)}" cy="${ys(v)}" r="${w < 460 ? 1.6 : 2.1}"/>`)).join('');
    host.innerHTML = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">
      ${axes(w, h, ys, ticks, (v) => String(v))}${strike}${days}
      <path class="band" d="${line([...top, ...bot])}Z"/><path class="c-acc" d="${line(stepPts(xs, ys, s.q50))}"/>${dots}</svg>`;
    host.setAttribute('aria-label', `${st.name}, ${wk.label}. Hourly departures with the forecast range.`);
    const missing = (s.actual as (number | null)[]).filter((v) => v == null).length;
    note.textContent = `${st.name} · ${st.role} dock · ${st.docks} docks · ${st.departures_2y.toLocaleString('en-GB')} departures in two years. ${wk.label}.${missing ? ` ${missing} hours are missing because the dock was closed.` : ''} Y axis: departures per hour.`;
  };

  fig.addEventListener('fig:change', (e) => {
    if ((e as CustomEvent).detail.key === 'week') { week = fig.dataset.week!; draw(); }
  });
  sel.addEventListener('change', () => { station = sel.value; draw(); });
  new ResizeObserver(() => draw()).observe(host);

  const load = () => fetch('/data/cycle/fan_charts.json').then((r) => r.json()).then((j) => {
    data = j;
    station = DEFAULT_STATION;
    sel.innerHTML = j.stations.map((x: any) => `<option value="${x.id}">${x.name} (${x.volume})</option>`).join('');
    sel.value = station;
    draw();
  });
  const io = new IntersectionObserver((es) => { if (es[0].isIntersecting) { io.disconnect(); load(); } }, { rootMargin: '300px' });
  io.observe(root);
}

/* ---------- cost explorer ---------- */
function explorer() {
  const root = $('[data-ex]');
  if (!root) return;
  const fig = root.closest('.fig') as HTMLElement;
  const { data, names } = json('data-explorer');
  const stops = [3, 10, 15, 25, 50];
  const losts = [1.1, 1.65, 3.3];
  const inStop = $<HTMLInputElement>('[data-in="stop"]', root)!;
  const inLost = $<HTMLInputElement>('[data-in="lost"]', root)!;
  const bars = $('[data-ex-bars]', root)!;
  const empty = $('[data-ex-empty]', root)!;
  const note = $('[data-ex-note]', root)!;
  const find = (k: string, s: number, l: number) =>
    data[k].find((x: any) => x.stop === s && x.lost === l && x.load === 18 && x.dock === 1 && Math.abs(x.blocked - l * (l === 1.65 ? 0.5 : 0.5)) < 1e-6);

  const draw = (changed?: 'stop' | 'lost') => {
    const k = fig.dataset.period!;
    let s = stops[+inStop.value], l = losts[+inLost.value];
    let msg = '';
    if (!find(k, s, l)) {
      // scenarios move one input at a time from base: keep the slider just moved, return the other to base
      if (changed === 'lost') { inStop.value = '1'; s = 10; } else { inLost.value = '1'; l = 1.65; }
      msg = `Scenarios change one input at a time, so the ${changed === 'lost' ? 'van-stop cost' : 'lost-trip value'} went back to its base value.`;
    }
    const sc = find(k, s, l);
    $('[data-out="stop"]', root)!.textContent = String(s);
    $('[data-out="lost"]', root)!.textContent = l.toFixed(2);
    note.textContent = msg || `Showing the scenario with a van stop at ${s} and a lost trip at ${l.toFixed(2)}.`;
    const order = Object.keys(names);
    const anyVan = order.some((p) => sc.p[p][2] > 0);
    bars.hidden = !anyVan;
    empty.hidden = anyVan;
    bars.innerHTML = order.map((p) => {
      const [cost, share, vans] = sc.p[p];
      return `<div class="bar-row ${p === 'quantile (full forecast)' ? 'on' : ''}"><span class="bar-n">${names[p]}</span><span class="bar-t"><i style="width:${Math.max(share, 0) * 100}%"></i></span><span class="bar-v">${pct(share)}</span><span class="bar-s">cost ${cost.toFixed(2)} per window, ${vans.toFixed(1)} van stops per 100 windows</span></div>`;
    }).join('');
  };
  inStop.addEventListener('input', () => draw('stop'));
  inLost.addEventListener('input', () => draw('lost'));
  fig.addEventListener('fig:change', () => draw());
  draw();
}

/* ---------- limited van ---------- */
function budgetChart() {
  const host = $('[data-chart="budget"]');
  if (!host) return;
  const fig = host.closest('.fig') as HTMLElement;
  const readout = $('[data-bud-readout]', fig)!;
  const all: { f: string; l: number | null; p: Record<string, number>; g: any }[] = json('data-budget');
  const S = [
    { key: 'last week as certain', label: 'Plan from last week’s counts', cls: 'c-grey2', dot: 'pk-g2' },
    { key: 'model mean as certain', label: 'Plan from the model’s average', cls: 'c-grey', dot: 'pk-g' },
    { key: 'quantile (full forecast)', label: 'Plan against the whole forecast', cls: 'c-acc', dot: 'pk' },
    { key: 'oracle', label: 'Oracle that sees the future', cls: 'c-oracle', dot: '' },
  ];
  const start: Record<string, number> = { stop10: 3, paid: 16 };
  let sel: Record<string, number | null> = { ...start };
  const fmt = (x: number) => `${x < 0 ? '−' : '+'}${Math.abs(x).toFixed(1)}`;
  const draw = () => {
    const fam = fig.dataset.family!;
    const rows = all.filter((r) => r.f === fam).sort((a, b) => (a.l ?? 1e9) - (b.l ?? 1e9));
    const { w, h } = frame(host, 280);
    const max = Math.ceil(Math.max(...rows.map((r) => r.p['oracle'])) / 20) * 20;
    const xs = (i: number) => M.l + 8 + (i / (rows.length - 1)) * (w - M.l - M.r - 16);
    const ys = (v: number) => h - M.b - (v / max) * (h - M.b - M.t);
    const step = max > 80 ? 40 : 10;
    const ticks = Array.from({ length: Math.floor(max / step) + 1 }, (_, i) => i * step);
    const si = Math.max(0, rows.findIndex((r) => r.l === sel[fam]));
    const lines = [...S].reverse().map((s) => {
      const pts = rows.map((r, i) => [xs(i), ys(r.p[s.key])] as Pt);
      return `<path class="${s.cls}" d="${line(pts)}"/>${s.dot ? pts.map((q) => `<circle class="${s.dot}" cx="${q[0]}" cy="${q[1]}" r="${s.dot === 'pk' ? 3.5 : 2.5}"/>`).join('') : ''}`;
    }).join('');
    const labels = rows.map((r, i) => `<text x="${xs(i)}" y="${h - 8}" text-anchor="middle">${r.l ?? 'No limit'}</text>`).join('');
    const bw = (w - M.l - M.r) / rows.length;
    const hits = rows.map((r, i) => `<rect class="hit" data-i="${i}" x="${xs(i) - bw / 2}" y="${M.t}" width="${bw}" height="${h - M.t - M.b + 22}"/>`).join('');
    host.innerHTML = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">
      ${axes(w, h, ys, ticks, (v) => String(v))}<line class="sel" x1="${xs(si)}" x2="${xs(si)}" y1="${M.t}" y2="${h - M.b}"/>${lines}${labels}
      <text class="axis-title" x="${M.l}" y="${M.t - 8}">Pounds saved per morning against doing nothing</text>
      <text class="axis-title" x="${w - M.r}" y="${h - M.b + 24}" text-anchor="end" style="display:none">Stops allowed per morning</text>${hits}</svg>
      <div class="legend">${S.map((s) => `<span><i class="lg ${s.cls}"></i>${s.label}</span>`).join('')}</div>
      <div class="legend"><span>Stops allowed per morning, left to right. Click a column for the numbers.</span></div>`;
    const r = rows[si];
    const g = r.g['model mean as certain'], gl = r.g['last week as certain'];
    readout.innerHTML = `With <b>${r.l ?? 'no limit on'} stops</b> a morning, the whole forecast saves <b>${r.p['quantile (full forecast)'].toFixed(1)}</b>, the model’s average <b>${r.p['model mean as certain'].toFixed(1)}</b>, last week’s counts <b>${r.p['last week as certain'].toFixed(1)}</b> and the oracle <b>${r.p['oracle'].toFixed(1)}</b>.<br>Whole forecast against the model’s average: ${fmt(g.pounds_per_morning)} a morning (${fmt(g.lo)} to ${fmt(g.hi)}). Against last week’s counts: ${fmt(gl.pounds_per_morning)} (${fmt(gl.lo)} to ${fmt(gl.hi)}). Negative means the whole forecast costs less.`;
    $$('.hit', host).forEach((el) => {
      const pick = () => { sel[fam] = rows[+el.dataset.i!].l; draw(); };
      el.addEventListener('click', pick);
      el.addEventListener('mouseenter', pick);
    });
  };
  onSize(host, draw);
}

demandChart();
budgetChart();
coverageChart();
fanChart();
explorer();
