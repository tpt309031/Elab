"use client";

import { useMemo, useState } from "react";
import { Clock3, Radar, Target } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { formatDate, formatPercent } from "@/lib/format";
import type { EventEvaluation, ResearchArtifact } from "@/lib/types";

interface EventLabProps {
  data: ResearchArtifact;
}

function EventCard({ event }: { event: EventEvaluation }) {
  const statusVariant = event.status === "matched" ? "default" : event.status === "not-matched" ? "destructive" : "secondary";
  return (
    <article className="border border-border bg-card p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div><p className="eyebrow">{event.source_type} / {event.lane}</p><h3 className="mt-1 text-sm font-medium">{event.source_name}</h3></div>
        <Badge variant={statusVariant}>{event.status}</Badge>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
        <div><p className="eyebrow">Target</p><p className="mt-1 font-mono">{formatDate(event.target_date)}</p></div>
        <div><p className="eyebrow">Call</p><p className="mt-1 font-mono uppercase">{event.direction}</p></div>
        <div><p className="eyebrow">Window</p><p className="mt-1 font-mono">{event.window_start.slice(5)} to {event.window_end.slice(5)}</p></div>
        <div><p className="eyebrow">Lead / lag</p><p className="mt-1 font-mono">{event.lead_lag_days == null ? "pending" : `${event.lead_lag_days > 0 ? "+" : ""}${event.lead_lag_days}d`}</p></div>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        {event.status === "pending"
          ? `Matures after ${formatDate(event.matures_after)}; this research score cannot change the daily grade.`
          : event.matched_event_types.length
            ? `${event.matched_event_types.join(", ")} matched on ${event.event_date ? formatDate(event.event_date) : "the event window"}.`
            : "No compatible pivot, wick, sideway cluster, or large-move event was found in the research window."}
      </p>
    </article>
  );
}

export function EventLab({ data }: EventLabProps) {
  const events = useMemo(() => data.learning?.event_evaluation_ledger ?? [], [data.learning?.event_evaluation_ledger]);
  const [lane, setLane] = useState("all");
  const [status, setStatus] = useState("all");
  const [source, setSource] = useState("all");
  const filtered = useMemo(() => events
    .filter((event) => lane === "all" || event.lane === lane)
    .filter((event) => status === "all" || event.status === status)
    .filter((event) => source === "all" || event.source_type === source)
    .sort((left, right) => right.target_date.localeCompare(left.target_date)), [events, lane, source, status]);
  const evaluated = events.filter((event) => event.status !== "pending");
  const matched = events.filter((event) => event.status === "matched");
  const definitions = data.research.event_definitions ?? {};

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 border border-border bg-card p-3 lg:flex-row lg:items-center lg:justify-between">
        <div><p className="eyebrow">separate research ledger</p><h2 className="mt-1 text-xl font-semibold">Pivot and large-move Event Lab</h2></div>
        <div className="grid grid-cols-3 gap-2">
          <Select value={lane} onValueChange={setLane}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All lanes</SelectItem><SelectItem value="Calendar">Calendar</SelectItem><SelectItem value="Full Hybrid">Full Hybrid</SelectItem></SelectContent></Select>
          <Select value={source} onValueChange={setSource}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All sources</SelectItem><SelectItem value="official">Official</SelectItem><SelectItem value="model">Model</SelectItem><SelectItem value="pattern">Pattern</SelectItem></SelectContent></Select>
          <Select value={status} onValueChange={setStatus}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All states</SelectItem><SelectItem value="pending">Pending</SelectItem><SelectItem value="matched">Matched</SelectItem><SelectItem value="not-matched">Not matched</SelectItem></SelectContent></Select>
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-3">
        <Card><CardContent className="p-4"><p className="eyebrow">Event records</p><strong className="metric-value mt-2 block">{events.length}</strong><small className="text-muted-foreground">daily grades remain immutable</small></CardContent></Card>
        <Card><CardContent className="p-4"><p className="eyebrow">Matured</p><strong className="metric-value mt-2 block">{evaluated.length}</strong><small className="text-muted-foreground">outside the full ±3d window</small></CardContent></Card>
        <Card><CardContent className="p-4"><p className="eyebrow">Match rate</p><strong className="metric-value mt-2 block">{evaluated.length ? formatPercent(matched.length / evaluated.length) : "—"}</strong><small className="text-muted-foreground">event match, not daily correctness</small></CardContent></Card>
      </div>

      <details className="group border border-border bg-card">
        <summary className="flex cursor-pointer list-none items-center justify-between p-4"><span className="flex items-center gap-2 text-sm font-medium"><Radar className="size-4 text-primary" />Research definitions</span><Badge variant="outline">±{definitions.window_days ?? 3} days</Badge></summary>
        <div className="grid gap-3 border-t border-border p-4 text-xs text-muted-foreground sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(definitions).map(([key, value]) => <p key={key}><b className="block text-foreground">{key.replaceAll("_", " ")}</b>{String(value)}</p>)}
        </div>
      </details>

      {!events.length ? (
        <Card><CardContent className="p-6 text-sm text-muted-foreground">No event evaluations have been issued yet.</CardContent></Card>
      ) : !filtered.length ? (
        <Card><CardContent className="p-6 text-sm text-muted-foreground">No event records match these filters.</CardContent></Card>
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">{filtered.slice(0, 60).map((event) => <EventCard key={event.event_id} event={event} />)}</div>
      )}
      {filtered.length > 60 && <div className="flex items-center gap-2 border border-border bg-card p-3 text-xs text-muted-foreground"><Clock3 className="size-4 text-primary" />Showing the 60 most recent records of {filtered.length}. Filters remain applied.</div>}
      <div className="flex gap-2 border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-100/80"><Target className="size-4 shrink-0 text-amber-400" /><p>Event matching is delayed by design until the complete research window matures. The official one-day forecast is still graded only as correct, partial, or wrong immediately after the UTC candle closes.</p></div>
    </div>
  );
}
