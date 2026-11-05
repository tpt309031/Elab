"use client";

import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ResearchArtifact } from "@/lib/types";

const money = (value: number | null | undefined) => value == null ? "Unavailable" : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 2 }).format(value);
const tooltip = { background: "#151515", border: "1px solid #555", color: "#fff" };

export function EtfFlowPanel({ data, startDate, endDate }: { data: ResearchArtifact; startDate?: string; endDate?: string }) {
  const etf = data.etf_flow;
  const archive = etf?.display_archive;
  const displayOnly = !etf?.history.length && !!archive?.history.length;
  const history = displayOnly ? archive?.history : etf?.history;
  const [mode, setMode] = useState<"daily" | "cumulative">("daily");
  const rows = useMemo(() => (history ?? []).filter((row) => (!startDate || row.date >= startDate.slice(0, 10)) && (!endDate || row.date <= endDate.slice(0, 10)))
    .map((row) => ({ ...row, value: (mode === "daily" ? row.etf_net_flow : row.cumulative_flow) / 1e6 })), [history, startDate, endDate, mode]);
  const latest = history?.at(-1);
  const evidence = etf?.contribution;
  const timestamp = displayOnly ? archive?.retrieved_at : latest?.available_at;
  const utcLabel = timestamp ? new Date(timestamp).toISOString().slice(0, 19).replace("T", " ") : "Unknown";
  return <section aria-label="Spot Bitcoin ETF flow" className="min-w-0 space-y-4 border border-border bg-card p-4 sm:p-5">
    <header className="flex flex-wrap items-start justify-between gap-3"><div><p className="eyebrow">ETF FLOW MONITOR</p><h2 className="mt-2 text-xl font-semibold">Spot Bitcoin ETF flow</h2><p className="mt-1 text-xs text-muted-foreground">US spot ETFs · USD · forecast lead: 2 days</p></div><Badge variant={etf?.health.status === "healthy" ? "outline" : "secondary"}>{displayOnly ? archive?.status : etf?.health.status ?? "Awaiting ETF pipeline artifact"}</Badge></header>
    {displayOnly && <p role="status" className="border-l-2 border-primary bg-primary/5 p-3 text-xs leading-5 text-muted-foreground">Historical display only. Original publication times are unknown, so these values are excluded from model features, backtests and forecast explanations. Capture time is not publication time.</p>}
    {!latest ? <div role="status" className="border border-dashed border-border p-5 text-sm leading-6 text-muted-foreground">ETF features are inactive until verified publication-time records are available in macro.csv. No synthetic data or assumed publication times are used. Existing forecasts remain available.</div> : <>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">{[["Latest reported flow", money(latest.etf_net_flow)], ["Cumulative since feed start", money(latest.cumulative_flow)], ["Observation date", latest.date], [displayOnly ? "Captured (UTC)" : "Published (UTC)", utcLabel]].map(([label, value]) => <div key={label} className="min-w-0 border border-border p-3"><p className="eyebrow">{label}</p><p className="mt-2 break-words font-mono text-sm sm:text-lg">{value}</p></div>)}</div>
      <div className="flex flex-wrap items-center justify-between gap-3"><p className="text-xs text-muted-foreground">Uses the price chart date filter · {displayOnly ? "latest downloaded history, may be revised" : "latest known vintage"}</p><div className="flex gap-2">{(["daily", "cumulative"] as const).map((value) => <Button key={value} size="sm" variant={mode === value ? "default" : "outline"} aria-pressed={mode === value} onClick={() => setMode(value)}>{value === "daily" ? "Daily" : "Cumulative"}</Button>)}</div></div>
      {rows.length ? <div role="img" aria-label={`${mode} ETF flow in USD millions`} className="h-64 min-w-0 overflow-hidden"><ResponsiveContainer width="100%" height="100%">{mode === "daily" ? <BarChart data={rows}><CartesianGrid stroke="#272727" vertical={false} /><XAxis dataKey="date" tick={{ fill: "#aaa", fontSize: 10 }} minTickGap={40} tickFormatter={(value: string) => value.slice(5)} /><YAxis width={56} tick={{ fill: "#aaa", fontSize: 10 }} unit="M" /><Tooltip formatter={(value) => typeof value === "number" ? value.toFixed(2) : value} contentStyle={tooltip} labelStyle={{ color: "#fff" }} itemStyle={{ color: "#fff" }} /><ReferenceLine y={0} stroke="#666" /><Bar dataKey="value" name="Net flow (USD M)" isAnimationActive={false}>{rows.map((row) => <Cell key={row.date} fill={row.value >= 0 ? "#34d399" : "#ef4444"} />)}</Bar></BarChart> : <LineChart data={rows}><CartesianGrid stroke="#272727" vertical={false} /><XAxis dataKey="date" tick={{ fill: "#aaa", fontSize: 10 }} minTickGap={40} tickFormatter={(value: string) => value.slice(5)} /><YAxis width={56} tick={{ fill: "#aaa", fontSize: 10 }} unit="M" /><Tooltip formatter={(value) => typeof value === "number" ? value.toFixed(2) : value} contentStyle={tooltip} labelStyle={{ color: "#fff" }} itemStyle={{ color: "#f7931a" }} /><Line dataKey="value" name="Cumulative (USD M)" stroke="#f7931a" dot={false} strokeWidth={2} isAnimationActive={false} /></LineChart>}</ResponsiveContainer></div> : <p className="p-4 text-sm text-muted-foreground">No ETF observations in this date range.</p>}
    </>}
    {displayOnly && <p className="text-[11px] leading-5 text-muted-foreground">Source: <a href="https://www.tftc.io/bitcoin-etf-flows" target="_blank" rel="noreferrer" className="underline">The Commoner Bitcoin ETF Tracker by TFTC</a>, via SoSoValue / Farside. <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer" className="underline">CC BY 4.0</a>. ELAB removes holiday placeholders and recalculates cumulative totals. <a href="https://www.tftc.io/bitcoin-etf-flows/methodology" target="_blank" rel="noreferrer" className="underline">Source methodology</a>.</p>}
    <details className="border-t border-border pt-3"><summary className="cursor-pointer text-sm">Latest call: ETF contribution &amp; correlation</summary><p className="mt-3 text-xs leading-5 text-muted-foreground">{evidence ? `${evidence.model} reference for ${evidence.target_date}. ${evidence.method}. These are logit terms, not changes in the calibrated ensemble probability.` : etf?.contribution_status ?? "No publication-time ETF evidence yet."}</p>
      <div className="mt-3 grid gap-2 md:grid-cols-2">{(etf?.correlations ?? []).filter((row) => row.samples > 0).map((row) => {
        const term = evidence?.features.find((feature) => feature.feature === row.feature);
        return <article key={row.feature} className="min-w-0 border border-border p-3 text-xs"><h3 className="break-all font-mono text-primary">{row.feature}</h3><p className="mt-2">Return correlation: {row.correlation?.toFixed(3) ?? "Insufficient variation / samples"} · n={row.samples}</p>{term && <p className="mt-2 font-mono">DOWN {term.down.toFixed(3)} · SIDEWAY {term.sideway.toFixed(3)} · UP {term.up.toFixed(3)}</p>}</article>;
      })}</div><p className="mt-3 text-[11px] text-muted-foreground">Correlation is descriptive, not a causal effect or proof of out-of-sample improvement. Published calls without an ETF snapshot are never assigned a new explanation retroactively.</p>
    </details>
    <details className="border-t border-border pt-3 text-xs text-muted-foreground"><summary className="cursor-pointer text-foreground">Availability &amp; interpretation</summary><div className="mt-3 space-y-2">{Object.entries(etf?.definitions ?? {}).map(([key, value]) => <p key={key}><b className="capitalize text-foreground">{key}: </b>{value}</p>)}</div></details>
  </section>;
}
