"use client";

import { useMemo, useState } from "react";
import { Activity, ChevronDown, Volume2 } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { formatPercent, formatSignedPercent } from "@/lib/format";
import type { ResearchArtifact } from "@/lib/types";

const number = (value: number | undefined) => value == null ? "Unavailable" : new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(value);

export function MarketActivityPanel({ data }: { data: ResearchArtifact }) {
  const activity = data.market_activity;
  const events = data.large_moves;
  const [lane, setLane] = useState("Hybrid + Volume");
  const [source, setSource] = useState("walk-forward");
  const [horizon, setHorizon] = useState("1");
  const [month, setMonth] = useState(data.meta.latest_closed_utc.slice(0, 7));
  const [alertsOnly, setAlertsOnly] = useState(false);
  const [page, setPage] = useState(0);
  const sourceRows = source === "official" ? events?.official : events?.historical;
  const rows = useMemo(() => (sourceRows ?? []).filter((row) => row.lane === lane && row.horizon === Number(horizon) && row.date.startsWith(month)), [sourceRows, lane, horizon, month]);
  const visible = rows.filter((row) => !alertsOnly || row.alert);
  const matured = rows.filter((row) => row.actual_event != null);
  const hits = matured.filter((row) => row.status === "hit").length;
  const falseAlarms = matured.filter((row) => row.status === "false-alarm").length;
  const missed = matured.filter((row) => row.status === "missed").length;
  const lastFlow = activity?.history.at(-1);
  const volumeChart = activity?.history.slice(-45).map((row) => ({ ...row, turnover: row.quote_volume / 1e9 })) ?? [];
  const next = events?.next.filter((row) => row.lane === lane) ?? [];
  const months = useMemo(() => [...new Set((sourceRows ?? []).map((row) => row.date.slice(0, 7)))].sort().reverse(), [sourceRows]);
  const scores = events?.metrics.filter((row) => row.source === source && row.horizon === Number(horizon)) ?? [];

  if (!events) return <div className="border border-border p-5 text-sm text-muted-foreground">Market activity research is not available in this artifact. The directional forecasts remain available.</div>;
  return (
    <section className="min-w-0 space-y-4" aria-label="Volume and large-move research">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div><p className="eyebrow">MARKET ACTIVITY / INDEPENDENT EVENT AUDIT</p><h2 className="mt-2 text-2xl font-semibold tracking-tight">Volume &amp; large moves</h2><p className="mt-2 max-w-3xl text-xs leading-5 text-muted-foreground">Measure magnitude separately from direction. A wick, a nearby pivot and a correct UP call are not interchangeable outcomes.</p></div>
        <Badge variant={activity?.health.status === "healthy" ? "outline" : "destructive"}>FLOW {activity?.health.status ?? "unavailable"}</Badge>
      </header>

      <div className="grid gap-px border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
        {[
          ["BTC volume", `${number(lastFlow?.base_volume)} BTC`, activity?.health.source ?? "No source"],
          ["Quote turnover", `${number(lastFlow?.quote_volume)} USDT`, `${number(lastFlow?.trades)} trades`],
          ["Relative volume", lastFlow?.spot_rvol20 == null ? "Unavailable" : `${lastFlow.spot_rvol20.toFixed(2)}x`, "versus prior 20-day average"],
          ["Taker imbalance", formatSignedPercent(lastFlow?.spot_taker_imbalance), "buy minus sell / total BTC volume"],
        ].map(([title, value, note]) => <div key={title} className="min-w-0 bg-card p-4"><p className="eyebrow">{title}</p><p className="mt-3 font-mono text-2xl font-semibold">{value}</p><p className="mt-2 text-[11px] text-muted-foreground">{note}</p></div>)}
      </div>

      <details className="group border border-border bg-card">
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3 p-4"><span className="flex items-center gap-2 text-sm"><Volume2 className="size-4 text-primary" />Exchange turnover <span className="text-xs text-muted-foreground">through {activity?.health.latest_closed_utc ?? "unavailable"}</span></span><ChevronDown className="size-4 transition group-open:rotate-180" /></summary>
        <div className="border-t border-border p-4">
          <div className="h-52 w-full min-w-0"><ResponsiveContainer width="100%" height="100%"><BarChart data={volumeChart}><CartesianGrid stroke="#272727" vertical={false} /><XAxis dataKey="date" tickFormatter={(date: string) => date.slice(5)} tick={{ fontSize: 10, fill: "#aaa" }} minTickGap={40} /><YAxis width={40} tick={{ fontSize: 10, fill: "#aaa" }} unit="B" /><Tooltip contentStyle={{ background: "#151515", border: "1px solid #555", color: "#fff" }} labelStyle={{ color: "#fff" }} itemStyle={{ color: "#f7931a" }} /><Bar name="USDT billions" dataKey="turnover" fill="#f7931a" /></BarChart></ResponsiveContainer></div>
          <p className="mt-2 text-[11px] text-muted-foreground">One venue, spot BTC/USDT only. Not global market volume. No imputed volume is displayed. Missing days: {activity?.health.missing_days ?? "unknown"}.</p>
        </div>
      </details>

      <p className="border border-amber-500/25 bg-amber-500/5 p-3 text-xs leading-5 text-amber-200">Experimental magnitude forecasts, not trading signals. Review precision and missed events below; a low predicted risk does not rule out a large move.</p>
      <div className="grid gap-3 sm:grid-cols-3">
        {next.map((row) => <article key={row.horizon} className={`border p-4 ${row.alert ? "border-primary bg-primary/5" : "border-border bg-card"}`}>
          <div className="flex items-center justify-between gap-2"><p className="eyebrow">{row.horizon}D / |MOVE| &gt; {formatPercent(row.threshold_move, 0)}</p><Activity className="size-4 text-primary" /></div>
          <div className="mt-3 flex items-end justify-between gap-2"><strong className="font-mono text-3xl">{formatPercent(row.probability)}</strong><Badge variant={row.alert ? "default" : "outline"}>{row.alert ? "ALERT" : "Below alert"}</Badge></div>
          <div className="mt-3 h-1 bg-white/10"><div className="h-full bg-primary" style={{ width: `${row.probability * 100}%` }} /></div>
          <p className="mt-3 text-xs text-muted-foreground">Target {row.date} · {row.model}</p><p className="mt-1 text-[11px] text-muted-foreground">{lane} · alert at {formatPercent(row.alert_threshold)}</p>
        </article>)}
      </div>

      <div className="flex flex-col gap-3 border border-border bg-card p-3">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          <Select value={source} onValueChange={(value) => { setSource(value); setPage(0); }}><SelectTrigger aria-label="Event evidence source"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="walk-forward">Historical walk-forward</SelectItem><SelectItem value="official">Published live record</SelectItem></SelectContent></Select>
          <Select value={lane} onValueChange={(value) => { setLane(value); setPage(0); }}><SelectTrigger aria-label="Event feature set"><SelectValue /></SelectTrigger><SelectContent>{["Index + Astro", "Hybrid", "Hybrid + Volume"].map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}</SelectContent></Select>
          <Select value={horizon} onValueChange={(value) => { setHorizon(value); setPage(0); }}><SelectTrigger aria-label="Event horizon"><SelectValue /></SelectTrigger><SelectContent>{["1", "3", "5"].map((value) => <SelectItem key={value} value={value}>{value}D horizon</SelectItem>)}</SelectContent></Select>
          <Select value={month} onValueChange={(value) => { setMonth(value); setPage(0); }}><SelectTrigger aria-label="Event month"><SelectValue placeholder={month} /></SelectTrigger><SelectContent>{[...new Set([month, ...months])].sort().reverse().map((value) => <SelectItem key={value} value={value}>{value}</SelectItem>)}</SelectContent></Select>
        </div>
        <p className="text-[11px] text-muted-foreground">Month metrics use all matured sessions in {month}, including missed events. Alerts-only changes the list, not its denominator.</p>
      </div>
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        {[["Alert precision", formatPercent(hits + falseAlarms ? hits / (hits + falseAlarms) : null), `${hits} hits / ${hits + falseAlarms} alerts`],
          ["Event recall", formatPercent(hits + missed ? hits / (hits + missed) : null), `${missed} missed events`],
          ["False alarms", String(falseAlarms), "warnings without a large move"],
          ["Evaluated", `${matured.length} / ${rows.length}`, "complete UTC horizons only"]].map(([label, value, note]) => <div key={label} className="border border-border bg-card p-4"><p className="eyebrow">{label}</p><strong className="mt-2 block font-mono text-2xl">{value}</strong><p className="mt-1 text-[11px] text-muted-foreground">{note}</p></div>)}
      </div>
      <details className="group border border-border bg-card">
        <summary className="flex cursor-pointer list-none items-center justify-between p-4 text-sm">Model comparison · full selected evidence period<ChevronDown className="size-4 transition group-open:rotate-180" /></summary>
        <div className="grid gap-2 border-t border-border p-3 md:grid-cols-2 xl:grid-cols-3">
          {scores.map((score) => <article key={`${score.lane}-${score.model}`} className="border border-border p-3 text-xs"><p className="eyebrow">{score.lane}</p><h3 className="mt-1 font-semibold">{score.model}</h3><div className="mt-3 grid grid-cols-2 gap-2"><p>Precision <b>{formatPercent(score.precision)}</b></p><p>Recall <b>{formatPercent(score.recall)}</b></p><p>Brier skill <b className={(score.brier_skill ?? 0) > 0 ? "text-emerald-400" : "text-red-400"}>{formatSignedPercent(score.brier_skill)}</b></p><p>Precision lower bound <b>{formatPercent(score.precision_lcb)}</b></p></div><p className="mt-3 text-[10px] text-muted-foreground">{score.samples} sessions · {score.alerts} alerts · {score.start} to {score.end}</p></article>)}
          {!scores.length && <p className="p-3 text-xs text-muted-foreground">No matured results yet. Live accuracy is not backfilled with historical results.</p>}
        </div>
      </details>

      <div className="flex items-center justify-between gap-2 text-xs"><h3 className="font-medium">Session audit · {month}</h3><label className="flex items-center gap-2"><input type="checkbox" checked={alertsOnly} onChange={(event) => { setAlertsOnly(event.target.checked); setPage(0); }} className="accent-orange-500" />Alerts only</label></div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {visible.slice(page * 12, (page + 1) * 12).map((row) => <article key={row.date} className="border border-border bg-card p-4">
          <div className="flex items-center justify-between gap-2"><time className="font-mono text-xs">{row.date}</time><Badge variant={row.status === "false-alarm" || row.status === "missed" ? "destructive" : "outline"}>{row.status}</Badge></div>
          <div className="mt-3 grid grid-cols-2 gap-2"><div><p className="eyebrow">Predicted risk</p><strong className="font-mono text-xl">{formatPercent(row.probability)}</strong></div><div><p className="eyebrow">Actual {horizon}D move</p><strong className="font-mono text-xl">{formatSignedPercent(row.actual_move)}</strong></div></div>
          <p className="mt-3 text-[11px] text-muted-foreground">Range {formatPercent(row.actual_range)} · {row.model}</p>
        </article>)}
        {!visible.length && <p className="col-span-full border border-dashed border-border p-6 text-center text-sm text-muted-foreground">No records for this selection. Published live records begin with this upgrade; history remains separate.</p>}
      </div>
      {visible.length > 12 && <div className="flex justify-end gap-4 text-xs"><button disabled={!page} onClick={() => setPage(page - 1)} className="p-2 disabled:opacity-30">Previous</button><span className="p-2">{page + 1} / {Math.ceil(visible.length / 12)}</span><button disabled={(page + 1) * 12 >= visible.length} onClick={() => setPage(page + 1)} className="p-2 disabled:opacity-30">Next</button></div>}
      <details className="group border border-border p-4 text-xs text-muted-foreground"><summary className="cursor-pointer text-foreground">Methodology &amp; limitations</summary><div className="mt-3 space-y-2">{Object.entries(events.definitions).map(([key, value]) => <p key={key}><b className="capitalize text-foreground">{key}: </b>{value}</p>)}<p>Overlapping 3D/5D outcomes are dependent. Comparisons are exploratory; positive Brier skill means improvement over the historical base-rate probability, not guaranteed profit.</p></div></details>
    </section>
  );
}
