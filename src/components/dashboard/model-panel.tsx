"use client";

import { useState } from "react";
import { BrainCircuit, CheckCircle2, CircleOff, RefreshCw } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { humanizeFeature } from "@/lib/format";
import type { ResearchArtifact } from "@/lib/types";

interface ModelPanelProps {
  data: ResearchArtifact;
}

export function ModelPanel({ data }: ModelPanelProps) {
  const [lane, setLane] = useState<"full" | "calendar">("full");
  const importance = lane === "full" ? data.explainability.full_hybrid : data.explainability.calendar;
  const selection = lane === "full" ? data.models.full_hybrid_latest_selection : data.models.calendar_latest_selection;
  const learning = data.learning?.summary;
  const chart = importance.slice(0, 16).map((row) => ({ feature: humanizeFeature(row.feature), importance: Math.max(0, row.importance) }));
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border border-border bg-card p-3">
        <div><p className="eyebrow">model governance</p><h2 className="mt-1 text-xl font-semibold">Ranking, replacement, and explainability</h2></div>
        <Tabs value={lane} onValueChange={(value) => setLane(value as typeof lane)}><TabsList><TabsTrigger value="full">Full Hybrid</TabsTrigger><TabsTrigger value="calendar">Calendar</TabsTrigger></TabsList></Tabs>
      </div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <Card><CardContent className="p-3"><p className="eyebrow">Official ledger</p><strong className="metric-value mt-2 block">{learning?.official_forecasts ?? 0}</strong><small className="text-muted-foreground">immutable forecasts</small></CardContent></Card>
        <Card><CardContent className="p-3"><p className="eyebrow">Evaluated</p><strong className="metric-value mt-2 block">{learning?.evaluated_forecasts ?? 0}</strong><small className="text-muted-foreground">last {learning?.last_evaluated_date ?? "awaiting close"}</small></CardContent></Card>
        <Card><CardContent className="p-3"><p className="eyebrow">Live weighted hit</p><strong className="metric-value mt-2 block">{learning?.live_weighted_accuracy == null ? "—" : `${(learning.live_weighted_accuracy * 100).toFixed(1)}%`}</strong><small className="text-muted-foreground">correct + partial scoring</small></CardContent></Card>
        <Card><CardContent className="p-3"><p className="eyebrow">Last rerank</p><strong className="mt-2 block font-mono text-lg">{learning?.last_selection_date ?? "—"}</strong><small className="text-muted-foreground">after closed UTC candle</small></CardContent></Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><BrainCircuit className="size-4 text-primary" />Global feature importance</CardTitle></CardHeader>
          <CardContent className="h-[460px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chart} layout="vertical" margin={{ left: 88, right: 12 }}>
                <CartesianGrid stroke="rgba(255,255,255,.06)" horizontal={false} />
                <XAxis type="number" tick={{ fill: "#888", fontSize: 9 }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="feature" width={138} tick={{ fill: "#aaa", fontSize: 9 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "#111", border: "1px solid #3a3a3a", color: "#f5f5f5" }} formatter={(value) => Number(value).toFixed(4)} />
                <Bar dataKey="importance" fill="#f7931a" radius={0} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <div className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="text-sm">Latest calibration selection</CardTitle></CardHeader>
            <CardContent className="space-y-3">{selection.map((row) => {
              const active = row.status === "active";
              return <div key={String(row.model)} className="space-y-2 border-b border-border pb-3 last:border-0"><div className="flex items-center justify-between gap-3"><span className="flex items-center gap-2 text-sm">{active ? <CheckCircle2 className="size-4 text-emerald-400" /> : <CircleOff className="size-4 text-muted-foreground" />}{String(row.model)}</span><span className="flex gap-1"><Badge variant="outline">#{Number(row.rank) || "—"}</Badge><Badge variant={active ? "default" : "secondary"}>{String(row.status)}</Badge></span></div><Progress value={Number(row.weight) * 100} /><div className="flex justify-between font-mono text-[10px] text-muted-foreground"><span>cal {(Number(row.calibration_score) * 100).toFixed(1)}% · live n={Number(row.live_samples) || 0}</span><span>{String(row.selection_change ?? "standby")}</span><span>weight {(Number(row.weight) * 100).toFixed(1)}%</span></div></div>;
            })}</CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><RefreshCw className="size-4 text-primary" />Candidate availability</CardTitle></CardHeader>
            <CardContent className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1">{data.models.availability.map((model) => <div key={model.model} className="flex items-center justify-between border border-border p-2.5 text-xs"><span><b className="font-medium">{model.model}</b><small className="ml-2 text-muted-foreground">{model.family}</small></span><Badge variant={model.available ? "outline" : "secondary"}>{model.available ? model.cadence : "gated"}</Badge></div>)}</CardContent>
          </Card>
        </div>
      </div>
      <p className="border border-border bg-card p-3 text-xs text-muted-foreground">{data.explainability.method}</p>
    </div>
  );
}
