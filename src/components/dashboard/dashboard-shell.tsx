"use client";

import { useMemo, useState } from "react";
import { Activity, BarChart3, BrainCircuit, CalendarDays, ChartCandlestick, DatabaseZap, Menu, Radio, RefreshCw, ScanSearch, Target } from "lucide-react";

import { BacktestPanel } from "@/components/dashboard/backtest-panel";
import { CorrelationHeatmap } from "@/components/dashboard/correlation-heatmap";
import { DashboardSkeleton } from "@/components/dashboard/dashboard-skeleton";
import { ForecastPanel } from "@/components/dashboard/forecast-panel";
import { MarketChart } from "@/components/dashboard/market-chart";
import { ModelPanel } from "@/components/dashboard/model-panel";
import { PatternPanel } from "@/components/dashboard/pattern-panel";
import { ProbabilityGauge } from "@/components/dashboard/probability-gauge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Sheet, SheetClose, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatDate, formatPercent, formatSignedPercent, formatUsd } from "@/lib/format";
import type { ForecastRow, MarketRow } from "@/lib/types";
import { useResearchData } from "@/lib/use-research-data";

const navigation = [
  { value: "decision", label: "Decision", icon: ChartCandlestick },
  { value: "forecast", label: "Forecast", icon: CalendarDays },
  { value: "patterns", label: "Patterns", icon: ScanSearch },
  { value: "backtest", label: "Backtest", icon: BarChart3 },
  { value: "models", label: "Models", icon: BrainCircuit },
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

function MetricCard({ label, value, detail, tone = "neutral" }: { label: string; value: string; detail: string; tone?: "positive" | "negative" | "neutral" }) {
  return <Card className="py-0"><CardContent className="p-4"><p className="eyebrow">{label}</p><strong className={`metric-value mt-2 block ${tone === "positive" ? "text-emerald-400" : tone === "negative" ? "text-red-400" : ""}`}>{value}</strong><p className="mt-2 truncate text-[11px] text-muted-foreground">{detail}</p></CardContent></Card>;
}

export function DashboardShell() {
  const { research, live } = useResearchData();
  const [section, setSection] = useState("decision");
  const [windowMonths, setWindowMonths] = useState("3");
  const [showForecasts, setShowForecasts] = useState(true);
  const [showIndices, setShowIndices] = useState(true);
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
  const nextFull = data?.forecast.full_hybrid_next_session.find((row) => row.forecast !== "no-call") ?? data?.forecast.full_hybrid_next_session[0];
  const bestModel = data?.performance.model_rankings.find((row) => row.lane === "Full Hybrid" && row.status === "active");
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
    return {
      market: marketWithLive.filter((row) => row.timestamp >= startText && row.timestamp <= endText),
      indices: showIndices ? data.indices.filter((row) => row.date >= startText && row.date <= endText) : [],
      forecasts: showForecasts ? [...historical, ...future].filter((row) => row.date >= startText && row.date <= endText) : [],
    };
  }, [anchorMonth, anchorYear, data, latestClosed, marketWithLive, showForecasts, showIndices, windowMonths]);

  if (research.isLoading || !data) return <DashboardSkeleton />;
  if (research.error) {
    return <main className="grid min-h-screen place-items-center p-6"><Card className="max-w-md"><CardHeader><CardTitle>Research artifact unavailable</CardTitle></CardHeader><CardContent><p className="text-sm text-muted-foreground">{research.error.message}</p><Button className="mt-4" onClick={() => research.mutate()}><RefreshCw />Retry</Button></CardContent></Card></main>;
  }
  const probabilities = [
    { label: "UP", value: nextFull?.prob_up ?? 1 / 3, className: "[&>div]:bg-emerald-400" },
    { label: "SIDEWAY", value: nextFull?.prob_sideway ?? 1 / 3, className: "[&>div]:bg-amber-300" },
    { label: "DOWN", value: nextFull?.prob_down ?? 1 / 3, className: "[&>div]:bg-red-400" },
  ];
  return (
    <main className="mx-auto min-h-screen w-full max-w-[1680px] px-3 py-4 sm:px-6 lg:px-8">
      <header className="mb-5 border-b border-border pb-5">
        <div className="flex items-start justify-between gap-4">
          <div><p className="eyebrow text-primary">ELAB / HYBRID QUANT RESEARCH</p><h1 className="mt-2 text-2xl font-semibold tracking-[-0.04em] sm:text-4xl">BTC Decision Console</h1><p className="mt-1 max-w-2xl text-sm text-muted-foreground">Private energy indices + Astro + patterns + calibrated machine learning, evaluated only through purged walk-forward data.</p></div>
          <div className="hidden items-center gap-2 lg:flex"><Badge variant="outline" className="h-9 gap-2"><Radio className={`size-3 ${live.data ? "text-emerald-400" : "text-amber-400"}`} />{live.data ? `${live.data.provider} · 5m` : "static daily"}</Badge><Badge variant="secondary" className="h-9 font-mono">closed UTC {data.meta.latest_closed_utc}</Badge></div>
          <Sheet>
            <SheetTrigger asChild><Button variant="outline" size="icon" className="md:hidden" aria-label="Open section menu"><Menu /></Button></SheetTrigger>
            <SheetContent side="right"><SheetHeader><SheetTitle>Research sections</SheetTitle></SheetHeader><nav className="mt-6 grid gap-2">{navigation.map((item) => <SheetClose asChild key={item.value}><Button variant={section === item.value ? "default" : "ghost"} className="justify-start" onClick={() => setSection(item.value)}><item.icon />{item.label}</Button></SheetClose>)}</nav></SheetContent>
          </Sheet>
        </div>
      </header>

      <Tabs value={section} onValueChange={setSection}>
        <TabsList className="mb-4 hidden h-11 w-full justify-start border border-border bg-card p-1 md:flex">{navigation.map((item) => <TabsTrigger key={item.value} value={item.value} className="gap-2 px-4"><item.icon />{item.label}</TabsTrigger>)}</TabsList>
        <div className="mb-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
          <MetricCard label="BTC live" value={formatUsd(latestMarket?.close)} detail={`${live.data?.provider ?? data.meta.market_provider} · ${live.data ? "5-minute refresh" : "closed daily"}`} tone={(liveMove ?? 0) >= 0 ? "positive" : "negative"} />
          <MetricCard label="Current move" value={formatSignedPercent(liveMove)} detail="versus latest closed UTC candle" tone={(liveMove ?? 0) >= 0 ? "positive" : "negative"} />
          <MetricCard label="Best OOS model" value={formatPercent(bestModel?.directional_accuracy)} detail={`${bestModel?.model ?? "No active model"} · PF ${bestModel?.profit_factor ? Number(bestModel.profit_factor).toFixed(2) : "—"}`} tone={(bestModel?.expectancy ?? 0) > 0 ? "positive" : "negative"} />
          <MetricCard label="System target" value={formatPercent(data.meta.target_directional_accuracy)} detail={`current best ${formatPercent(data.meta.achieved_directional_accuracy)}`} tone={data.meta.target_reached ? "positive" : "negative"} />
          <MetricCard label="Next session" value={(nextFull?.forecast ?? "no-call").toUpperCase()} detail={nextFull ? `${formatDate(nextFull.date)} · expected ${formatPercent(nextFull.expected_score)}` : "awaiting research refresh"} />
        </div>

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
              <Card><CardHeader><CardTitle className="text-sm">Decision context</CardTitle></CardHeader><CardContent className="space-y-3 text-xs text-muted-foreground"><p><b className="text-foreground">Session:</b> {nextFull ? formatDate(nextFull.date) : "—"}</p><p><b className="text-foreground">Policy:</b> {nextFull?.policy_mode ?? "calibrated utility"} · sideway weight {nextFull?.sideway_penalty?.toFixed(2) ?? "—"}</p><p><b className="text-foreground">Monthly limits:</b> max {data.meta.validation.maximum_sideway_calls_per_month ?? 8} SIDEWAY · max {data.meta.validation.maximum_no_calls_per_month ?? 6} NO CALL</p><p><b className="text-foreground">Members:</b> {nextFull?.model_members?.join(", ") ?? "awaiting selection"}</p><p><b className="text-foreground">Pattern:</b> {nextFull?.top_pattern?.name ?? "No active pattern match"}</p><p><b className="text-foreground">Data:</b> {data.meta.availability_assumption}</p></CardContent></Card>
              <div className="flex gap-2 border border-border bg-card p-3 text-xs text-muted-foreground"><DatabaseZap className="size-4 shrink-0 text-primary" /><span>Research artifact {data.meta.generated_at.slice(0, 16).replace("T", " ")} UTC. Live price refreshes independently every five minutes.</span></div>
            </aside>
          </div>
          <CorrelationHeatmap rows={data.research.correlation_heatmap} />
        </TabsContent>
        <TabsContent value="forecast"><ForecastPanel data={data} /></TabsContent>
        <TabsContent value="patterns"><PatternPanel data={data} /></TabsContent>
        <TabsContent value="backtest"><BacktestPanel data={data} /></TabsContent>
        <TabsContent value="models"><ModelPanel data={data} /></TabsContent>
      </Tabs>
      <footer className="mt-8 flex flex-col gap-2 border-t border-border py-5 text-[11px] text-muted-foreground sm:flex-row sm:items-center sm:justify-between"><span>Research decision support only. No performance target is guaranteed.</span><span className="font-mono">schema v{data.meta.schema_version} · {data.meta.oos_start} → {data.meta.oos_end}</span></footer>
    </main>
  );
}
