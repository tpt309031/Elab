"use client";

import { useMemo, useState } from "react";
import { Activity, AlertTriangle, BarChart3, BrainCircuit, CalendarDays, ChartCandlestick, DatabaseZap, Menu, Radar, RefreshCw, ScanSearch, ShieldCheck, SlidersHorizontal, Target } from "lucide-react";
import dynamic from "next/dynamic";

import { CorrelationHeatmap } from "@/components/dashboard/correlation-heatmap";
import { DashboardSkeleton } from "@/components/dashboard/dashboard-skeleton";
import { DecisionStrip } from "@/components/dashboard/decision-strip";
import { MarketChart } from "@/components/dashboard/market-chart";
import { ProbabilityGauge } from "@/components/dashboard/probability-gauge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { HeroOdyssey } from "@/components/ui/hero-odyssey";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Sheet, SheetClose, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { formatDate, formatPercent, formatSignedPercent, formatUsd } from "@/lib/format";
import type { ForecastRow, MarketRow } from "@/lib/types";
import { useResearchData } from "@/lib/use-research-data";

const panelLoader = () => <div className="min-h-64 animate-pulse border border-border bg-card p-5 text-xs uppercase tracking-[0.18em] text-muted-foreground">Loading research module…</div>;
const BacktestPanel = dynamic(() => import("@/components/dashboard/backtest-panel").then((module) => module.BacktestPanel), { loading: panelLoader });
const DiagnosticsPanel = dynamic(() => import("@/components/dashboard/diagnostics-panel").then((module) => module.DiagnosticsPanel), { loading: panelLoader });
const EventLab = dynamic(() => import("@/components/dashboard/event-lab").then((module) => module.EventLab), { loading: panelLoader });
const ForecastPanel = dynamic(() => import("@/components/dashboard/forecast-panel").then((module) => module.ForecastPanel), { loading: panelLoader });
const ModelPanel = dynamic(() => import("@/components/dashboard/model-panel").then((module) => module.ModelPanel), { loading: panelLoader });
const PatternPanel = dynamic(() => import("@/components/dashboard/pattern-panel").then((module) => module.PatternPanel), { loading: panelLoader });
const SystemPanel = dynamic(() => import("@/components/dashboard/system-panel").then((module) => module.SystemPanel), { loading: panelLoader });

const navigation = [
  { value: "decision", label: "Decision", icon: ChartCandlestick },
  { value: "forecast", label: "Forecast", icon: CalendarDays },
  { value: "patterns", label: "Patterns", icon: ScanSearch },
  { value: "backtest", label: "Backtest", icon: BarChart3 },
  { value: "models", label: "Models", icon: BrainCircuit },
  { value: "diagnostics", label: "Diagnostics", icon: SlidersHorizontal },
  { value: "events", label: "Event Lab", icon: Radar },
  { value: "system", label: "System", icon: ShieldCheck },
] as const;

function aggregateLiveDay(rows: MarketRow[], latestClosed: string): MarketRow[] {
  const byDate = new Map<string, MarketRow[]>();
  for (const row of rows) {
    const date = row.timestamp.slice(0, 10);
    if (date <= latestClosed) continue;
    const bucket = byDate.get(date) ?? [];
    bucket.push(row);
    byDate.set(date, bucket);
  }
  return [...byDate.entries()].map(([timestamp, bucket]) => ({
    timestamp,
    open: bucket[0].open,
    high: Math.max(...bucket.map((row) => row.high)),
    low: Math.min(...bucket.map((row) => row.low)),
    close: bucket.at(-1)?.close ?? bucket[0].close,
    volume: bucket.reduce((sum, row) => sum + row.volume, 0),
  }));
}

export function DashboardShell() {
  const [section, setSection] = useState("decision");
  const [windowMonths, setWindowMonths] = useState("3");
  const [showForecasts, setShowForecasts] = useState(true);
  const [showIndices, setShowIndices] = useState(true);
  const loadResearchDetails = section !== "decision";
  const { research, live, deep, health } = useResearchData(loadResearchDetails);
  const data = research.data;
  const latestClosed = data?.meta.latest_closed_utc ?? new Date().toISOString().slice(0, 10);
  const [defaultYear, defaultMonth] = latestClosed.slice(0, 7).split("-");
  const [anchorYear, setAnchorYear] = useState(defaultYear);
  const [anchorMonth, setAnchorMonth] = useState(defaultMonth);

  const marketWithLive = useMemo(() => {
    if (!data) return [];
    const partial = aggregateLiveDay(live.data?.rows ?? [], data.meta.latest_closed_utc);
    return [...data.market, ...partial].sort((left, right) => left.timestamp.localeCompare(right.timestamp));
  }, [data, live.data?.rows]);
  const latestMarket = live.data?.rows.at(-1) ?? marketWithLive.at(-1);
  const priorDaily = data?.market.at(-1);
  const liveMove = latestMarket && priorDaily ? latestMarket.close / priorDaily.close - 1 : null;
  const firstPublishableTarget = data?.meta.first_publishable_target_utc?.slice(0, 10) ?? latestClosed;
  const officialNextFull = (data?.learning?.official_forecast_ledger ?? [])
    .filter((row) => (
      row.lane === "Full Hybrid"
      && (row.contract_version ?? 0) >= 2
      && row.date >= firstPublishableTarget
    ))
    .sort((left, right) => left.date.localeCompare(right.date))[0];
  const generatedNextFull = data?.forecast.full_hybrid_next_session.find((row) => row.forecast !== "no-call")
    ?? data?.forecast.full_hybrid_next_session[0];
  const generatedOfficialMatch = officialNextFull
    ? data?.forecast.full_hybrid_next_session.find((row) => row.date === officialNextFull.date)
    : undefined;
  const nextFull = officialNextFull
    ? { ...generatedOfficialMatch, ...officialNextFull }
    : generatedNextFull;
  const fullModelRankings = data?.performance.model_rankings.filter((row) => row.lane === "Full Hybrid") ?? [];
  const activeModel = fullModelRankings.find((row) => row.status === "active");
  const topOosModel = activeModel ?? fullModelRankings[0];
  const years = useMemo(() => data ? [...new Set(data.indices.map((row) => row.date.slice(0, 4)))].sort() : [], [data]);
  const chartWindow = useMemo(() => {
    if (!data) return { market: [], indices: [], forecasts: [] as ForecastRow[] };
    const end = new Date(Date.UTC(Number(anchorYear), Number(anchorMonth), 0));
    const latest = new Date(`${marketWithLive.at(-1)?.timestamp ?? latestClosed}T00:00:00Z`);
    const boundedEnd = end > latest ? latest : end;
    const start = new Date(boundedEnd);
    start.setUTCMonth(start.getUTCMonth() - Number(windowMonths));
    start.setUTCDate(start.getUTCDate() + 1);
    const startText = start.toISOString().slice(0, 10);
    const endText = boundedEnd.toISOString().slice(0, 10);
    const historical = data.forecast.historical_full_hybrid_oos;
    const future = data.forecast.full_hybrid_next_session;
    const official = (data.learning?.official_forecast_ledger ?? [])
      .filter((row) => row.lane === "Full Hybrid")
      .map((row) => ({ ...row, daily_return: row.actual_return }));
    const forecastMap = new Map([...historical, ...future, ...official].map((row) => [row.date, row]));
    return {
      market: marketWithLive.filter((row) => row.timestamp >= startText && row.timestamp <= endText),
      indices: showIndices ? data.indices.filter((row) => row.date >= startText && row.date <= endText) : [],
      forecasts: showForecasts ? [...forecastMap.values()].filter((row) => row.date >= startText && row.date <= endText) : [],
    };
  }, [anchorMonth, anchorYear, data, latestClosed, marketWithLive, showForecasts, showIndices, windowMonths]);

  if (research.error) {
    return <main className="grid min-h-screen place-items-center p-6"><Card className="max-w-md"><CardHeader><CardTitle>Research artifact unavailable</CardTitle></CardHeader><CardContent><p className="text-sm text-muted-foreground">{research.error.message}</p><Button className="mt-4" onClick={() => research.mutate()}><RefreshCw />Retry</Button></CardContent></Card></main>;
  }
  if (research.isLoading || !data) return <DashboardSkeleton />;
  const probabilities = [
    { label: "UP", value: nextFull?.prob_up ?? 1 / 3, className: "[&>div]:bg-emerald-400" },
    { label: "SIDEWAY", value: nextFull?.prob_sideway ?? 1 / 3, className: "[&>div]:bg-amber-300" },
    { label: "DOWN", value: nextFull?.prob_down ?? 1 / 3, className: "[&>div]:bg-red-400" },
  ];
  const activeSection = navigation.find((item) => item.value === section)?.label ?? "Decision";
  const heroMetrics = [
    { label: "BTC live", value: formatUsd(latestMarket?.close), detail: `${live.data?.provider ?? data.meta.market_provider} · ${live.data ? "5-minute refresh" : "closed daily"}`, tone: (liveMove ?? 0) >= 0 ? "positive" as const : "negative" as const },
    { label: "Current move", value: formatSignedPercent(liveMove), detail: "versus latest closed UTC candle", tone: (liveMove ?? 0) >= 0 ? "positive" as const : "negative" as const },
    { label: "Top OOS model", value: formatPercent(topOosModel?.directional_accuracy), detail: `${topOosModel?.model ?? "Awaiting model"} · ${activeModel ? "trade eligible" : "standby"}`, tone: activeModel ? "positive" as const : "warning" as const },
    { label: "System target", value: formatPercent(data.meta.target_directional_accuracy), detail: `current best ${formatPercent(data.meta.achieved_directional_accuracy)}`, tone: data.meta.target_reached ? "positive" as const : "warning" as const },
    { label: "Next session", value: (nextFull?.forecast ?? "no-call").toUpperCase(), detail: nextFull ? `${formatDate(nextFull.date)} · ${nextFull.trade_eligible ? `TRADE ${nextFull.trade_action?.toUpperCase()}` : "FLAT"}` : "awaiting research refresh" },
  ];
  return (
    <main className="mx-auto min-h-screen w-full max-w-[1720px] px-2 py-2 sm:px-5 sm:py-4 lg:px-7">
      <HeroOdyssey
        eyebrow="ELAB / HYBRID QUANT RESEARCH"
        title="BTC Decision Console"
        description="Private energy indices, Astro timing, technical patterns and calibrated machine learning. Every production claim is measured on purged walk-forward data."
        sectionLabel={activeSection}
        live={Boolean(live.data)}
        liveLabel={live.data ? `${live.data.provider} · 5 MIN` : "STATIC DAILY"}
        latestClosed={data.meta.latest_closed_utc}
        forecast={(nextFull?.forecast ?? "no-call").toUpperCase()}
        forecastDate={nextFull ? formatDate(nextFull.date) : undefined}
        forecastConfidence={nextFull?.expected_score}
        metrics={heroMetrics}
        nodes={[
          { label: "Model", value: topOosModel?.model ?? "Awaiting selection" },
          { label: "Validation", value: "Purged walk-forward" },
          { label: "Market feed", value: live.data?.provider ?? data.meta.market_provider },
          { label: "Artifact", value: data.meta.generated_at.slice(0, 16).replace("T", " ") + " UTC" },
        ]}
        actions={(
          <Sheet>
            <SheetTrigger asChild><Button variant="outline" aria-label="Open section menu"><Menu /><span className="hidden sm:inline">{activeSection}</span></Button></SheetTrigger>
            <SheetContent side="right"><SheetHeader><SheetTitle>Research sections</SheetTitle></SheetHeader><nav className="mt-6 grid gap-2">{navigation.map((item) => <SheetClose asChild key={item.value}><Button variant={section === item.value ? "default" : "ghost"} className="justify-start" onClick={() => setSection(item.value)}><item.icon />{item.label}</Button></SheetClose>)}</nav></SheetContent>
          </Sheet>
        )}
      />

      <DecisionStrip
        forecast={nextFull}
        meta={data.meta}
        marketStale={Boolean(health.data?.artifact.stale || data.health?.market.stale)}
      />

      <Tabs value={section} onValueChange={setSection}>
        {(health.data?.status === "unhealthy" || data.health?.market.stale) && <div className="mb-4 flex gap-2 border border-red-500/30 bg-red-500/5 p-3 text-xs text-red-100"><AlertTriangle className="size-4 shrink-0 text-red-400" /><span>Research freshness check failed. Forecasts remain visible for audit, but no stale artifact should be treated as a new decision.</span></div>}

        <TabsContent value="decision" className="space-y-4">
          <div className="flex flex-col gap-3 border border-border bg-card p-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex flex-wrap gap-2">{[1, 3, 6, 12].map((months) => <Button key={months} size="sm" variant={windowMonths === String(months) ? "default" : "outline"} onClick={() => setWindowMonths(String(months))}>{months}M</Button>)}</div>
            <div className="grid grid-cols-2 gap-2 sm:flex">
              <Select value={anchorYear} onValueChange={setAnchorYear}><SelectTrigger className="w-full sm:w-28"><SelectValue /></SelectTrigger><SelectContent>{years.map((year) => <SelectItem key={year} value={year}>{year}</SelectItem>)}</SelectContent></Select>
              <Select value={anchorMonth} onValueChange={setAnchorMonth}><SelectTrigger className="w-full sm:w-32"><SelectValue /></SelectTrigger><SelectContent>{Array.from({ length: 12 }, (_, index) => String(index + 1).padStart(2, "0")).map((month) => <SelectItem value={month} key={month}>{new Intl.DateTimeFormat("en-US", { month: "long", timeZone: "UTC" }).format(new Date(Date.UTC(2024, Number(month) - 1, 1)))}</SelectItem>)}</SelectContent></Select>
              <Button size="sm" variant={showIndices ? "secondary" : "outline"} onClick={() => setShowIndices((value) => !value)}><Activity />Indices</Button>
              <Button size="sm" variant={showForecasts ? "secondary" : "outline"} onClick={() => setShowForecasts((value) => !value)}><Target />Calls</Button>
            </div>
          </div>
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
            <MarketChart market={chartWindow.market} indices={chartWindow.indices} forecasts={chartWindow.forecasts} />
            <aside className="space-y-4">
              <Card className="panel-grid"><CardHeader><CardTitle className="flex items-center justify-between text-sm"><span className="flex items-center gap-2"><Target className="size-4 text-primary" />Probability stack</span><Badge variant="outline">{nextFull?.lane ?? "Full Hybrid"}</Badge></CardTitle></CardHeader><CardContent><ProbabilityGauge forecast={nextFull} /><div className="space-y-3">{probabilities.map((item) => <div key={item.label}><div className="mb-1 flex justify-between font-mono text-[10px]"><span>{item.label}</span><span>{formatPercent(item.value)}</span></div><Progress value={item.value * 100} className={item.className} /></div>)}</div></CardContent></Card>
              <Card><CardHeader><CardTitle className="text-sm">Decision context</CardTitle></CardHeader><CardContent className="space-y-3 text-xs text-muted-foreground"><p><b className="text-foreground">Session:</b> {nextFull ? formatDate(nextFull.date) : "—"}</p><p><b className="text-foreground">Policy:</b> {nextFull?.policy_mode ?? "calibrated utility"} · SIDEWAY weight {nextFull?.sideway_penalty?.toFixed(2) ?? "—"}</p><p><b className="text-foreground">Execution:</b> {nextFull?.trade_eligible ? `TRADE ${nextFull.trade_action?.toUpperCase()}` : "FLAT"} · lower-bound edge {formatSignedPercent(nextFull?.expectancy_lcb)}</p><p><b className="text-foreground">Members:</b> {nextFull?.model_members?.join(", ") ?? "awaiting selection"}</p><p><b className="text-foreground">Pattern:</b> {nextFull?.top_pattern?.name ?? "No active pattern match"}</p><p><b className="text-foreground">Timing:</b> information through {nextFull?.information_cutoff_utc?.slice(0, 10) ?? data.meta.latest_closed_utc}; target opens {nextFull?.target_start_utc?.slice(0, 10) ?? "—"} UTC.</p></CardContent></Card>
              <div className="flex gap-2 border border-border bg-card p-3 text-xs text-muted-foreground"><DatabaseZap className="size-4 shrink-0 text-primary" /><span>Research artifact {data.meta.generated_at.slice(0, 16).replace("T", " ")} UTC. Live price refreshes independently every five minutes.</span></div>
            </aside>
          </div>
          <CorrelationHeatmap rows={data.research.correlation_heatmap} />
        </TabsContent>
        <TabsContent value="forecast"><ForecastPanel data={data} /></TabsContent>
        <TabsContent value="patterns"><PatternPanel data={data} /></TabsContent>
        <TabsContent value="backtest"><BacktestPanel data={data} /></TabsContent>
        <TabsContent value="models"><ModelPanel data={data} /></TabsContent>
        <TabsContent value="diagnostics"><DiagnosticsPanel data={data} /></TabsContent>
        <TabsContent value="events"><EventLab data={data} /></TabsContent>
        <TabsContent value="system"><SystemPanel data={data} deep={deep.data} health={health.data} /></TabsContent>
      </Tabs>
      <footer className="mt-8 flex flex-col gap-2 border-t border-border py-5 text-[11px] text-muted-foreground sm:flex-row sm:items-center sm:justify-between"><span>Research decision support only. No performance target is guaranteed.</span><span className="font-mono">schema v{data.meta.schema_version} · {data.meta.oos_start} → {data.meta.oos_end}</span></footer>
    </main>
  );
}
