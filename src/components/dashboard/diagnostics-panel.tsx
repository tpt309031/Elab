"use client";

import { useMemo, useState } from "react";
import { Activity, Grid3X3, SlidersHorizontal } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatPercent, formatSignedPercent } from "@/lib/format";
import type { ResearchArtifact } from "@/lib/types";

interface DiagnosticsPanelProps {
  data: ResearchArtifact;
}

const classes = ["down", "sideway", "up"] as const;
const tooltipStyle = { background: "#111", border: "1px solid #353535", color: "#f5f5f5" };

export function DiagnosticsPanel({ data }: DiagnosticsPanelProps) {
  const [lane, setLane] = useState<"Calendar" | "Full Hybrid">("Full Hybrid");
  const dimensions = useMemo(
    () => [...new Set((data.performance.grouped ?? []).map((row) => row.dimension))].sort(),
    [data.performance.grouped],
  );
  const [dimension, setDimension] = useState(dimensions[0] ?? "year");
  const classMetrics = (data.performance.class_metrics ?? []).filter((row) => row.lane === lane);
  const confusion = (data.performance.confusion_matrix ?? []).filter((row) => row.lane === lane);
  const confidenceRisk = (data.performance.confidence_risk ?? []).filter((row) => row.lane === lane);
  const grouped = (data.performance.grouped ?? []).filter((row) => row.lane === lane && row.dimension === dimension);
  const forecasts = lane === "Full Hybrid"
    ? data.forecast.historical_full_hybrid_oos
    : data.forecast.historical_calendar_oos;
  const distribution = classes.map((direction) => ({
    direction: direction.toUpperCase(),
    count: forecasts.filter((row) => row.forecast === direction).length,
  }));
  const cell = (actual: string, predicted: string) => confusion.find(
    (row) => row.actual === actual && row.predicted === predicted,
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 border border-border bg-card p-3 sm:flex-row sm:items-center sm:justify-between">
        <div><p className="eyebrow">OOS diagnostics</p><h2 className="mt-1 text-xl font-semibold">Calibration, errors, and regime stability</h2></div>
        <Tabs value={lane} onValueChange={(value) => setLane(value as typeof lane)}>
          <TabsList><TabsTrigger value="Full Hybrid">Full Hybrid</TabsTrigger><TabsTrigger value="Calendar">Calendar</TabsTrigger></TabsList>
        </Tabs>
      </div>

      {!classMetrics.length ? (
        <Card><CardContent className="p-6 text-sm text-muted-foreground">Diagnostics are unavailable in this artifact version.</CardContent></Card>
      ) : (
        <>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(320px,.7fr)]">
            <Card>
              <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><Grid3X3 className="size-4 text-primary" />Confusion matrix</CardTitle></CardHeader>
              <CardContent>
                <div className="grid grid-cols-4 gap-1 text-center text-xs">
                  <span />
                  {classes.map((item) => <span key={item} className="eyebrow py-2">Pred {item}</span>)}
                  {classes.map((actual) => (
                    <div key={actual} className="contents">
                      <span className="eyebrow flex items-center justify-end pr-2">Actual {actual}</span>
                      {classes.map((predicted) => {
                        const value = cell(actual, predicted);
                        const rate = value?.row_rate ?? 0;
                        return (
                          <div
                            key={`${actual}-${predicted}`}
                            className="border border-border p-4"
                            style={{ backgroundColor: `rgb(247 147 26 / ${Math.max(0.04, rate * 0.75)})` }}
                          >
                            <strong className="block font-mono text-lg">{value?.count ?? 0}</strong>
                            <small className="font-mono text-muted-foreground">{formatPercent(rate)}</small>
                          </div>
                        );
                      })}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle className="text-sm">Precision and recall</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                {classMetrics.map((metric) => (
                  <div key={metric.class} className="border-b border-border pb-3 last:border-0">
                    <div className="mb-2 flex items-center justify-between"><Badge variant="outline" className="uppercase">{metric.class}</Badge><span className="font-mono text-xs text-muted-foreground">n={metric.support}</span></div>
                    <div className="grid grid-cols-3 gap-2 text-center"><div><p className="eyebrow">Precision</p><strong className="font-mono">{formatPercent(metric.precision)}</strong></div><div><p className="eyebrow">Recall</p><strong className="font-mono">{formatPercent(metric.recall)}</strong></div><div><p className="eyebrow">F1</p><strong className="font-mono">{formatPercent(metric.f1)}</strong></div></div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            <Card>
              <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><SlidersHorizontal className="size-4 text-primary" />Confidence-risk curve</CardTitle></CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={confidenceRisk} margin={{ left: -12, right: 12 }}>
                    <CartesianGrid stroke="rgba(255,255,255,.06)" />
                    <XAxis dataKey="coverage" tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} tick={{ fill: "#888", fontSize: 9 }} />
                    <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} tick={{ fill: "#888", fontSize: 9 }} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(value) => formatPercent(Number(value))} />
                    <Legend wrapperStyle={{ fontSize: 10 }} />
                    <Line dataKey="exact_accuracy" name="Exact" stroke="#34d399" strokeWidth={2} dot={false} />
                    <Line dataKey="weighted_accuracy" name="Weighted" stroke="#f7931a" strokeWidth={2} dot={false} />
                    <Line dataKey="minimum_confidence" name="Min confidence" stroke="#6b7280" strokeDasharray="4 4" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><Activity className="size-4 text-primary" />Forecast distribution</CardTitle></CardHeader>
              <CardContent className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={distribution} margin={{ left: -18, right: 12 }}>
                    <CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} />
                    <XAxis dataKey="direction" tick={{ fill: "#aaa", fontSize: 10 }} />
                    <YAxis tick={{ fill: "#888", fontSize: 9 }} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="count" name="OOS sessions" fill="#f7931a" radius={0} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader className="flex-row items-center justify-between gap-3"><CardTitle className="text-sm">Performance by independent segment</CardTitle><Select value={dimension} onValueChange={setDimension}><SelectTrigger className="w-44"><SelectValue /></SelectTrigger><SelectContent>{dimensions.map((item) => <SelectItem value={item} key={item}>{item.replaceAll("_", " ")}</SelectItem>)}</SelectContent></Select></CardHeader>
            <CardContent className="overflow-x-auto p-0">
              <Table>
                <TableHeader><TableRow><TableHead>Segment</TableHead><TableHead>Calls</TableHead><TableHead>Exact</TableHead><TableHead>Weighted</TableHead><TableHead>Directional</TableHead><TableHead>Expectancy</TableHead></TableRow></TableHeader>
                <TableBody>{grouped.map((row) => <TableRow key={`${row.dimension}-${row.value}`}><TableCell className="font-medium">{row.value}</TableCell><TableCell className="font-mono">{row.calls}</TableCell><TableCell className="font-mono">{formatPercent(row.exact_accuracy)}</TableCell><TableCell className="font-mono">{formatPercent(row.weighted_accuracy)}</TableCell><TableCell className="font-mono">{formatPercent(row.directional_accuracy)}</TableCell><TableCell className={row.expectancy >= 0 ? "font-mono text-emerald-400" : "font-mono text-red-400"}>{formatSignedPercent(row.expectancy, 3)}</TableCell></TableRow>)}</TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
