"use client";

import { useMemo, useState } from "react";
import { CalendarDays, ChevronDown, Target } from "lucide-react";

import { AccuracyBar } from "@/components/dashboard/accuracy-bar";
import { AnalogMiniCard } from "@/components/dashboard/analog-mini-card";
import { ForecastCalendar } from "@/components/dashboard/forecast-calendar";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatDate, formatPercent, formatSignedPercent } from "@/lib/format";
import type { ForecastDirection, ForecastRow, ResearchArtifact } from "@/lib/types";

interface ForecastPanelProps {
  data: ResearchArtifact;
}

function mergeForecastRows(...sources: ForecastRow[][]): ForecastRow[] {
  const rows = new Map<string, ForecastRow>();
  for (const source of sources) for (const row of source) rows.set(row.date, row);
  return [...rows.values()].sort((left, right) => left.date.localeCompare(right.date));
}

function buildFusionFutureRows(calendar: ForecastRow[], fullHybrid: ForecastRow[]): ForecastRow[] {
  const rows = new Map<string, ForecastRow>(
    calendar.map((row) => [row.date, { ...row, lane: "Fusion ex-ante" }]),
  );
  for (const row of fullHybrid) rows.set(row.date, row);
  return [...rows.values()].sort((left, right) => left.date.localeCompare(right.date));
}

export function ForecastPanel({ data }: ForecastPanelProps) {
  const initialMonth = data.meta.latest_closed_utc.slice(0, 7);
  const [lane, setLane] = useState<"calendar" | "full">("full");
  const [month, setMonth] = useState(initialMonth);
  const [direction, setDirection] = useState<ForecastDirection | "all">("all");
  const [selected, setSelected] = useState<ForecastRow>();
  const laneRows = useMemo(() => {
    const official = (data.learning?.official_forecast_ledger ?? [])
      .filter((row) => row.lane === (lane === "calendar" ? "Calendar" : "Full Hybrid"))
      .map((row) => ({ ...row, daily_return: row.actual_return, lane: `${row.lane} · official` }));
    return lane === "calendar"
      ? mergeForecastRows(data.forecast.historical_calendar_oos, data.forecast.calendar, official)
      : mergeForecastRows(
        data.forecast.historical_full_hybrid_oos,
        buildFusionFutureRows(data.forecast.calendar, data.forecast.full_hybrid_next_session),
        official,
      );
  }, [data, lane]);
  const filtered = useMemo(() => laneRows.filter((row) => row.date.startsWith(month) && (direction === "all" || row.forecast === direction)), [direction, laneRows, month]);
  const years = useMemo(() => [...new Set(laneRows.map((row) => row.date.slice(0, 4)))].sort(), [laneRows]);
  const selectedYear = month.slice(0, 4);
  const selectedMonth = month.slice(5, 7);
  const setYear = (year: string) => setMonth(`${year}-${selectedMonth}`);
  const setMonthNumber = (monthNumber: string) => setMonth(`${selectedYear}-${monthNumber}`);
  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 border border-border bg-card p-3 lg:flex-row lg:items-center lg:justify-between">
        <Tabs value={lane} onValueChange={(value) => setLane(value as "calendar" | "full")}>
          <TabsList>
            <TabsTrigger value="full">Fusion Forecast</TabsTrigger>
            <TabsTrigger value="calendar">Index + Astro</TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="grid grid-cols-3 gap-2 sm:flex">
          <Select value={selectedYear} onValueChange={setYear}>
            <SelectTrigger className="w-full sm:w-28"><SelectValue /></SelectTrigger>
            <SelectContent>{years.map((year) => <SelectItem key={year} value={year}>{year}</SelectItem>)}</SelectContent>
          </Select>
          <Select value={selectedMonth} onValueChange={setMonthNumber}>
            <SelectTrigger className="w-full sm:w-32"><SelectValue /></SelectTrigger>
            <SelectContent>{Array.from({ length: 12 }, (_, index) => String(index + 1).padStart(2, "0")).map((value) => <SelectItem key={value} value={value}>{new Intl.DateTimeFormat("en-US", { month: "short", timeZone: "UTC" }).format(new Date(Date.UTC(2024, Number(value) - 1, 1)))}</SelectItem>)}</SelectContent>
          </Select>
          <Select value={direction} onValueChange={(value) => setDirection(value as ForecastDirection | "all")}>
            <SelectTrigger className="w-full sm:w-32"><SelectValue /></SelectTrigger>
            <SelectContent>{["all", "up", "down", "sideway", "no-call"].map((value) => <SelectItem key={value} value={value}>{value.toUpperCase()}</SelectItem>)}</SelectContent>
          </Select>
        </div>
      </div>
      {lane === "full" && <p className="border-l-2 border-primary bg-primary/5 px-3 py-2 text-xs text-muted-foreground">The next UTC session uses Full Hybrid. Later sessions use leakage-safe Fusion ex-ante signals from Index + Astro because future OHLCV is not yet observable.</p>}
      <AccuracyBar rows={laneRows} />
      <ForecastCalendar month={month} rows={filtered} selectedDate={selected?.date} onMonthChange={setMonth} onSelect={setSelected} />
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
        <details className="group border border-border bg-card" open={Boolean(selected)}>
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 p-4">
            <span className="flex items-center gap-2 text-sm font-medium"><CalendarDays className="size-4 text-primary" />Daily decision evidence</span>
            <ChevronDown className="size-4 text-muted-foreground transition group-open:rotate-180" />
          </summary>
          <div className="border-t border-border p-4">
            {selected ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="outline">{formatDate(selected.date)}</Badge>
                  <Badge variant="outline">{selected.lane ?? (lane === "full" ? "Full Hybrid" : "Index + Astro")}</Badge>
                  <Badge className="uppercase">{selected.forecast}</Badge>
                  <Badge variant={selected.status === "wrong" ? "destructive" : "secondary"}>{selected.status}</Badge>
                  {selected.sideway_cap_override && <Badge variant="outline">SIDEWAY cap override</Badge>}
                  <span className="font-mono text-xs text-muted-foreground">actual {formatSignedPercent(selected.daily_return)}</span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="border border-border p-3"><p className="eyebrow">UP</p><strong className="font-mono text-emerald-400">{formatPercent(selected.prob_up)}</strong></div>
                  <div className="border border-border p-3"><p className="eyebrow">SIDEWAY</p><strong className="font-mono text-amber-300">{formatPercent(selected.prob_sideway)}</strong></div>
                  <div className="border border-border p-3"><p className="eyebrow">DOWN</p><strong className="font-mono text-red-400">{formatPercent(selected.prob_down)}</strong></div>
                </div>
                {selected.top_pattern && <p className="text-sm text-muted-foreground"><Target className="mr-2 inline size-4 text-primary" />Pattern #{selected.top_pattern.rank}: <span className="text-foreground">{selected.top_pattern.name}</span> · {selected.top_pattern.occurrences} occurrences · {formatPercent(selected.top_pattern.weighted_accuracy)}</p>}
              </div>
            ) : <p className="text-sm text-muted-foreground">Select a calendar session to inspect its probability stack, pattern evidence, and analogs.</p>}
          </div>
        </details>
        <Card>
          <CardHeader><CardTitle className="text-sm">Nearest historical analog paths</CardTitle></CardHeader>
          <CardContent className="grid gap-2 sm:grid-cols-2 lg:grid-cols-1">
            {(selected?.similar_cases ?? []).slice(0, 6).map((item) => <AnalogMiniCard key={item.date} item={item} market={data.market} />)}
            {!selected?.similar_cases?.length && <p className="text-sm text-muted-foreground">Analog evidence is shown for future calendar calls.</p>}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
