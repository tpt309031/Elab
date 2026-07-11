"use client";

import { useMemo, useState } from "react";
import { Activity, ShieldCheck, TrendingDown, TrendingUp } from "lucide-react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { formatPercent, formatSignedPercent } from "@/lib/format";
import type { ModelMetric, ResearchArtifact } from "@/lib/types";

interface BacktestPanelProps {
  data: ResearchArtifact;
}
const tooltipStyle = { background: "#111", border: "1px solid #3a3a3a", color: "#f5f5f5", fontSize: 11 };

function MetricTile({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "positive" | "negative" | "neutral" }) {
  return <div className="border border-border bg-card p-3"><p className="eyebrow">{label}</p><strong className={`metric-value mt-2 block ${tone === "positive" ? "text-emerald-400" : tone === "negative" ? "text-red-400" : ""}`}>{value}</strong></div>;
}

export function BacktestPanel({ data }: BacktestPanelProps) {
  const [lane, setLane] = useState<"Full Hybrid" | "Calendar">("Full Hybrid");
  const metrics = data.performance.model_rankings.filter((metric) => metric.lane === lane);
  const ensembleName = lane === "Full Hybrid" ? "Full Hybrid Ensemble" : "Calendar Ensemble";
  const ensemble = metrics.find((metric) => metric.model === ensembleName) ?? metrics[0];
  const equity = lane === "Full Hybrid" ? data.performance.full_hybrid_equity : data.performance.calendar_equity;
  const reliability = lane === "Full Hybrid" ? data.performance.full_hybrid_reliability : data.performance.calendar_reliability;
  const monthly = data.performance.monthly.filter((row) => row.lane === lane).slice(-18).map((row) => ({ ...row, exact: row.exact_accuracy * 100, weighted: row.weighted_accuracy * 100 }));
  const chartEquity = useMemo(() => equity.map((point, index) => index % 3 === 0 || index === equity.length - 1 ? point : null).filter(Boolean), [equity]);
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 border border-border bg-card p-3">
        <div><p className="eyebrow">out-of-sample only</p><h2 className="mt-1 text-xl font-semibold">Walk-forward performance</h2></div>
        <Tabs value={lane} onValueChange={(value) => setLane(value as typeof lane)}><TabsList><TabsTrigger value="Full Hybrid">Full Hybrid</TabsTrigger><TabsTrigger value="Calendar">Calendar</TabsTrigger></TabsList></Tabs>
      </div>
      {ensemble && <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8">
        <MetricTile label="Directional" value={formatPercent(ensemble.directional_accuracy)} />
        <MetricTile label="Exact" value={formatPercent(ensemble.exact_accuracy)} />
        <MetricTile label="Weighted" value={formatPercent(ensemble.weighted_accuracy)} />
        <MetricTile label="Coverage" value={formatPercent(ensemble.coverage)} />
        <MetricTile label="Sharpe" value={ensemble.sharpe.toFixed(2)} tone={ensemble.sharpe > 0 ? "positive" : "negative"} />
        <MetricTile label="Profit factor" value={Number.isFinite(ensemble.profit_factor) ? Number(ensemble.profit_factor).toFixed(2) : "∞"} tone={Number(ensemble.profit_factor) > 1 ? "positive" : "negative"} />
        <MetricTile label="Max drawdown" value={formatPercent(ensemble.max_drawdown)} tone="negative" />
        <MetricTile label="Expectancy" value={formatSignedPercent(ensemble.expectancy, 3)} tone={ensemble.expectancy > 0 ? "positive" : "negative"} />
      </div>}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(360px,1fr)]">
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><TrendingUp className="size-4 text-primary" />Equity and benchmark</CardTitle></CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartEquity} margin={{ left: -12, right: 8 }}>
                <defs><linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#f7931a" stopOpacity={0.32} /><stop offset="1" stopColor="#f7931a" stopOpacity={0} /></linearGradient></defs>
                <CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} />
                <XAxis dataKey="date" minTickGap={48} tick={{ fill: "#888", fontSize: 9 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#888", fontSize: 9 }} axisLine={false} tickLine={false} tickFormatter={(value) => `${Number(value).toFixed(1)}x`} />
                <Tooltip contentStyle={tooltipStyle} formatter={(value) => `${Number(value).toFixed(3)}x`} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Area type="monotone" dataKey="equity" name="Strategy" stroke="#f7931a" fill="url(#equityFill)" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="benchmark" name="BTC buy & hold" stroke="#6b7280" strokeWidth={1} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><ShieldCheck className="size-4 text-primary" />Probability calibration</CardTitle></CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={reliability} margin={{ left: -18, right: 8 }}>
                <CartesianGrid stroke="rgba(255,255,255,.06)" />
                <XAxis dataKey="confidence" tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} tick={{ fill: "#888", fontSize: 9 }} />
                <YAxis domain={[0, 1]} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} tick={{ fill: "#888", fontSize: 9 }} />
                <Tooltip contentStyle={tooltipStyle} formatter={(value) => formatPercent(Number(value))} />
                <Line type="monotone" dataKey="observed_accuracy" name="Observed" stroke="#f7931a" strokeWidth={2} />
                <Line type="linear" dataKey="confidence" name="Perfect calibration" stroke="#6b7280" dot={false} strokeDasharray="4 4" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><Activity className="size-4 text-primary" />Monthly stability</CardTitle></CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={monthly} margin={{ left: -20, right: 8 }}>
              <CartesianGrid stroke="rgba(255,255,255,.06)" vertical={false} />
              <XAxis dataKey="month" minTickGap={24} tick={{ fill: "#888", fontSize: 9 }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 100]} tick={{ fill: "#888", fontSize: 9 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} formatter={(value) => `${Number(value).toFixed(1)}%`} />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Bar dataKey="exact" name="Exact" fill="#34d399" radius={0} />
              <Bar dataKey="weighted" name="Weighted" fill="#f7931a" radius={0} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle className="text-sm">Model evaluation and replacement state</CardTitle></CardHeader>
        <CardContent className="overflow-x-auto p-0">
          <Table>
            <TableHeader><TableRow><TableHead>Rank</TableHead><TableHead>Model</TableHead><TableHead>Status</TableHead><TableHead>Directional</TableHead><TableHead>Weighted</TableHead><TableHead>Sharpe</TableHead><TableHead>PF</TableHead><TableHead>Expectancy</TableHead><TableHead>DD</TableHead><TableHead>Coverage</TableHead></TableRow></TableHeader>
            <TableBody>{metrics.map((metric: ModelMetric) => <TableRow key={metric.model}><TableCell className="font-mono">#{metric.rank}</TableCell><TableCell className="font-medium">{metric.model}</TableCell><TableCell><Badge variant={metric.status === "active" ? "default" : "secondary"}>{metric.status}</Badge></TableCell><TableCell className="font-mono">{formatPercent(metric.directional_accuracy)}</TableCell><TableCell className="font-mono">{formatPercent(metric.weighted_accuracy)}</TableCell><TableCell className={metric.sharpe >= 0 ? "font-mono text-emerald-400" : "font-mono text-red-400"}>{metric.sharpe.toFixed(2)}</TableCell><TableCell className="font-mono">{Number.isFinite(metric.profit_factor) ? Number(metric.profit_factor).toFixed(2) : "∞"}</TableCell><TableCell className={metric.expectancy >= 0 ? "font-mono text-emerald-400" : "font-mono text-red-400"}>{formatSignedPercent(metric.expectancy, 3)}</TableCell><TableCell className="font-mono text-red-300">{formatPercent(metric.max_drawdown)}</TableCell><TableCell className="font-mono">{formatPercent(metric.coverage)}</TableCell></TableRow>)}</TableBody>
          </Table>
        </CardContent>
      </Card>
      <div className="flex gap-2 border border-amber-500/25 bg-amber-500/5 p-3 text-xs text-amber-100/80"><TrendingDown className="size-4 shrink-0 text-amber-400" /><p>The 70% target is a research gate, not a guaranteed display value. Any model with negative expectancy or unstable monthly performance is moved to standby even if one accuracy metric looks strong.</p></div>
    </div>
  );
}
