"use client";

import { useMemo, useState } from "react";
import { Activity, ArrowDown, ArrowUp, CircleDot, History } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatPercent, formatSignedPercent } from "@/lib/format";
import type { PatternMetric, ResearchArtifact } from "@/lib/types";

interface PatternPanelProps {
  data: ResearchArtifact;
}

function DirectionIcon({ direction }: { direction: PatternMetric["direction"] }) {
  if (direction === "up") return <ArrowUp className="size-4 text-emerald-400" />;
  if (direction === "down") return <ArrowDown className="size-4 text-red-400" />;
  return <CircleDot className="size-4 text-amber-300" />;
}

export function PatternPanel({ data }: PatternPanelProps) {
  const [lane, setLane] = useState<"full_hybrid" | "calendar">("full_hybrid");
  const patterns = data.patterns[lane];
  const patternKey = (pattern: PatternMetric) => `${pattern.pattern_id}:${pattern.direction}`;
  const [selectedId, setSelectedId] = useState(patterns[0] ? patternKey(patterns[0]) : undefined);
  const selected = patterns.find((pattern) => patternKey(pattern) === selectedId) ?? patterns[0];
  const chartData = useMemo(() => patterns.slice(0, 16).map((pattern) => ({
    name: `#${pattern.rank}`,
    accuracy: pattern.weighted_accuracy * 100,
    direction: pattern.direction,
  })), [patterns]);
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
      <div className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border border-border bg-card p-3">
          <div><p className="eyebrow">walk-forward registry</p><h2 className="mt-1 text-xl font-semibold">Pattern ranking</h2></div>
          <Tabs value={lane} onValueChange={(value) => { const nextLane = value as typeof lane; setLane(nextLane); const first = data.patterns[nextLane][0]; setSelectedId(first ? patternKey(first) : undefined); }}>
            <TabsList><TabsTrigger value="full_hybrid">Fusion</TabsTrigger><TabsTrigger value="calendar">Calendar</TabsTrigger></TabsList>
          </Tabs>
        </div>
        <div className="grid gap-2 sm:grid-cols-2">
          {patterns.slice(0, 24).map((pattern) => (
            <button
              type="button"
              key={`${pattern.pattern_id}-${pattern.direction}`}
              onClick={() => setSelectedId(patternKey(pattern))}
              className={`grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3 border p-3 text-left transition hover:border-primary/70 ${selected && patternKey(selected) === patternKey(pattern) ? "border-primary bg-primary/5" : "border-border bg-card"}`}
            >
              <span className="grid size-8 place-content-center border border-border bg-background font-mono text-xs">{pattern.rank}</span>
              <span className="min-w-0"><b className="block truncate text-sm font-medium">{pattern.pattern}</b><small className="mt-1 block font-mono text-[10px] text-muted-foreground">n={pattern.occurrences} · {formatPercent(pattern.weighted_accuracy)} weighted</small></span>
              <DirectionIcon direction={pattern.direction} />
            </button>
          ))}
        </div>
        <Card>
          <CardHeader><CardTitle className="text-sm">Top-16 probability profile</CardTitle></CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ left: -20, right: 6 }}>
                <CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#888", fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#888", fontSize: 10 }} axisLine={false} tickLine={false} domain={[0, 100]} />
                <Tooltip contentStyle={{ background: "#111", border: "1px solid #3a3a3a", color: "#f5f5f5" }} formatter={(value) => `${Number(value).toFixed(1)}%`} />
                <Bar dataKey="accuracy" radius={0}>{chartData.map((item) => <Cell key={item.name} fill={item.direction === "up" ? "#34d399" : item.direction === "down" ? "#ef4444" : "#fbbf24"} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
      <aside className="space-y-4 xl:sticky xl:top-4 xl:self-start">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between gap-2"><Badge variant="outline">Rank {selected?.rank ?? "—"}</Badge><Badge>{selected?.status ?? "standby"}</Badge></div>
            <CardTitle className="pt-2 text-lg">{selected?.pattern ?? "No eligible pattern"}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {selected && <>
              <div className="grid grid-cols-2 gap-2">
                <div className="border border-border p-3"><p className="eyebrow">occurrences</p><strong className="metric-value">{selected.occurrences}</strong></div>
                <div className="border border-border p-3"><p className="eyebrow">weighted hit</p><strong className="metric-value">{formatPercent(selected.weighted_accuracy)}</strong></div>
                <div className="border border-border p-3"><p className="eyebrow">exact hit</p><strong className="metric-value">{formatPercent(selected.exact_accuracy)}</strong></div>
                <div className="border border-border p-3"><p className="eyebrow">expectancy</p><strong className={selected.expectancy >= 0 ? "metric-value text-emerald-400" : "metric-value text-red-400"}>{formatSignedPercent(selected.expectancy)}</strong></div>
              </div>
              <div className="border border-border p-3"><p className="eyebrow">rule expression</p><code className="mt-2 block break-words font-mono text-[11px] leading-5 text-muted-foreground">{selected.expression}</code></div>
              <div><p className="mb-2 flex items-center gap-2 text-xs font-medium"><History className="size-4 text-primary" />Recent historical occurrences</p><div className="flex flex-wrap gap-2">{selected.examples.map((date) => <Badge variant="secondary" key={date}>{date}</Badge>)}</div></div>
            </>}
          </CardContent>
        </Card>
        <div className="border border-border bg-card p-4 text-xs text-muted-foreground"><Activity className="mb-2 size-4 text-primary" />Patterns are scored only on rows available before each forecast. A strong in-sample shape cannot become active unless its walk-forward expectancy and hit score survive replacement checks.</div>
      </aside>
    </div>
  );
}
