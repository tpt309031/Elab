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
    calendar.map((row) => [row.date, { ...row, lane: "Index + Astro outlook" }]),
  );
  for (const row of fullHybrid) rows.set(row.date, row);
  return [...rows.values()].sort((left, right) => left.date.localeCompare(right.date));
}

export function ForecastPanel({ data }: ForecastPanelProps) {
  const firstFutureDate = [
    ...data.forecast.full_hybrid_next_session,
    ...data.forecast.calendar,
  ].filter((row) => row.date > data.meta.latest_closed_utc).sort((left, right) => left.date.localeCompare(right.date))[0]?.date;
  const initialMonth = (firstFutureDate ?? data.meta.latest_closed_utc).slice(0, 7);
  const [lane, setLane] = useState<"calendar" | "full">("full");
  const [month, setMonth] = useState(initialMonth);
  const [direction, setDirection] = useState<ForecastDirection | "all">("all");
  const [selectedDate, setSelectedDate] = useState<string>();
  const [evidence, setEvidence] = useState("all");
  const officialRows = useMemo(() => (data.learning?.official_forecast_ledger ?? [])
    .filter((row) => row.lane === (lane === "calendar" ? "Calendar" : "Full Hybrid"))
    .map((row) => ({ ...row, daily_return: row.actual_return, lane: `${row.lane} · official` })), [data, lane]);
  const laneRows = useMemo(() => {
    const official = officialRows;
    if (evidence === "official") return official;
    if (evidence === "oos") return lane === "calendar" ? data.forecast.historical_calendar_oos : data.forecast.historical_full_hybrid_oos;
    return lane === "calendar"
      ? mergeForecastRows(data.forecast.historical_calendar_oos, data.forecast.calendar, official)
      : mergeForecastRows(
        data.forecast.historical_full_hybrid_oos,
        buildFusionFutureRows(data.forecast.calendar, data.forecast.full_hybrid_next_session),
        official,
      );
  }, [data, lane, evidence, officialRows]);
  const filtered = useMemo(() => laneRows.filter((row) => row.date.startsWith(month) && (direction === "all" || row.forecast === direction)), [direction, laneRows, month]);
  const selected = filtered.find((row) => row.date === selectedDate);
  const selectedEvents = data.large_moves?.official.filter((row) => row.date === selectedDate && row.lane === (lane === "calendar" ? "Index + Astro" : "Hybrid + Volume")) ?? [];
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
      {lane === "full" && <p className="border-l-2 border-primary bg-primary/5 px-3 py-2 text-xs text-muted-foreground">The first publishable UTC session uses Full Hybrid. Later dates are Index + Astro outlooks because future market bars do not exist yet. Monthly policy allows at most 8 SIDEWAY calls and 4 NO CALL sessions; only rows marked TRADE pass the after-cost execution gate.</p>}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-muted-foreground">Accuracy for {month} and the selected direction filter. Historical research and live results are different evidence.</p>
        <Select value={evidence} onValueChange={setEvidence}><SelectTrigger className="w-full sm:w-56" aria-label="Forecast evidence"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">Calendar overview</SelectItem><SelectItem value="official">Published forecasts only</SelectItem><SelectItem value="oos">Walk-forward only</SelectItem></SelectContent></Select>
      </div>
      <AccuracyBar rows={filtered} />
      <ForecastCalendar month={month} rows={filtered} selectedDate={selected?.date} onMonthChange={setMonth} onSelect={(row) => setSelectedDate(row.date)} />
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
                  <Badge variant={selected.trade_eligible ? "default" : "outline"}>{selected.trade_eligible ? `TRADE ${selected.trade_action?.toUpperCase()}` : "FLAT"}</Badge>
                  {selected.contract_version && <Badge variant="outline">CONTRACT V{selected.contract_version}</Badge>}
                  <span className="font-mono text-xs text-muted-foreground">actual {formatSignedPercent(selected.daily_return)}</span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="border border-border p-3"><p className="eyebrow">UP</p><strong className="font-mono text-emerald-400">{formatPercent(selected.prob_up)}</strong></div>
                  <div className="border border-border p-3"><p className="eyebrow">SIDEWAY</p><strong className="font-mono text-amber-300">{formatPercent(selected.prob_sideway)}</strong></div>
                  <div className="border border-border p-3"><p className="eyebrow">DOWN</p><strong className="font-mono text-red-400">{formatPercent(selected.prob_down)}</strong></div>
                </div>
                <p className="text-[11px] leading-5 text-muted-foreground">These class probabilities use UP &gt;+1%, DOWN &lt;-1%, SIDEWAY within +/-1%. They are not the probability of an exact +/-3% daily grade. Daily outcomes remain Correct, Partial or Wrong.</p>
                {selected.sideway_cap_override && <p className="border border-amber-500/30 p-2 text-xs text-amber-200">Monthly quota changed the preferred SIDEWAY decision. This forced directional outlook is not additional model evidence.</p>}
                {selectedEvents.length > 0 && <div className="border border-border p-3 text-xs"><p className="eyebrow mb-2">Separately published magnitude forecast</p>{selectedEvents.map((row) => <p key={row.horizon} className="mt-1">{row.horizon}D |move| &gt; {formatPercent(row.threshold_move, 0)}: <b>{formatPercent(row.probability)}</b> · {row.status} · actual {formatSignedPercent(row.actual_move)}</p>)}</div>}
                {selected.top_pattern && <p className="text-sm text-muted-foreground"><Target className="mr-2 inline size-4 text-primary" />Pattern #{selected.top_pattern.rank}: <span className="text-foreground">{selected.top_pattern.name}</span> · {selected.top_pattern.occurrences} occurrences · weighted grade {formatPercent(selected.top_pattern.weighted_accuracy)} · {selected.top_pattern.duration_days ?? 1}d shape / +{selected.top_pattern.signal_lag_days ?? 0}d lead</p>}
                <div className="grid gap-2 border-t border-border pt-3 text-xs text-muted-foreground sm:grid-cols-2"><p><b className="text-foreground">Execution:</b> {selected.trade_gate_reason ?? "Legacy forecast without execution gate"}</p><p><b className="text-foreground">Timing:</b> information cutoff {selected.information_cutoff_utc?.slice(0, 10) ?? "legacy"}; target opens {selected.target_start_utc?.slice(0, 10) ?? selected.date} UTC.</p></div>
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
