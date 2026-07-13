"use client";

import { AlertTriangle, CheckCircle2, Cpu, DatabaseZap, ShieldCheck } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { formatPercent, formatSignedPercent } from "@/lib/format";
import type { DeepResearchArtifact, ResearchArtifact, SystemHealthResponse } from "@/lib/types";

interface SystemPanelProps {
  data: ResearchArtifact;
  deep?: DeepResearchArtifact;
  health?: SystemHealthResponse;
}

const tooltipStyle = { background: "#111", border: "1px solid #353535", color: "#f5f5f5" };

export function SystemPanel({ data, deep, health }: SystemPanelProps) {
  const market = data.health?.market;
  const intraday = data.health?.intraday ?? [];
  const external = data.health?.external ?? [];
  const drift = data.research.drift;
  const featureDrift = (drift?.features ?? []).slice().sort((left, right) => right.psi - left.psi);
  const alerts = featureDrift.filter((row) => row.status === "alert");
  const performanceAlarms = Object.entries(drift?.performance ?? {}).filter(([, value]) => value.alarm);
  const status = health?.status ?? (market?.stale ? "unhealthy" : "degraded");
  const healthy = status === "healthy";

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 border border-border bg-card p-3 sm:flex-row sm:items-center sm:justify-between">
        <div><p className="eyebrow">operations and governance</p><h2 className="mt-1 text-xl font-semibold">System health, drift, and challengers</h2></div>
        <Badge variant={healthy ? "default" : status === "unhealthy" ? "destructive" : "secondary"} className="gap-2 uppercase">{healthy ? <CheckCircle2 className="size-3" /> : <AlertTriangle className="size-3" />}{status}</Badge>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <Card><CardContent className="p-4"><p className="eyebrow">Expected candle</p><strong className="mt-2 block font-mono text-lg">{health?.expectedClosedUtc ?? market?.expected_closed_utc ?? "—"}</strong><small className="text-muted-foreground">UTC daily close</small></CardContent></Card>
        <Card><CardContent className="p-4"><p className="eyebrow">Actual candle</p><strong className={market?.stale ? "mt-2 block font-mono text-lg text-red-400" : "mt-2 block font-mono text-lg text-emerald-400"}>{market?.actual_closed_utc ?? data.meta.latest_closed_utc}</strong><small className="text-muted-foreground">{market?.selected_provider ?? data.meta.market_provider}</small></CardContent></Card>
        <Card><CardContent className="p-4"><p className="eyebrow">Feature drift</p><strong className={alerts.length ? "metric-value mt-2 block text-amber-400" : "metric-value mt-2 block text-emerald-400"}>{alerts.length}</strong><small className="text-muted-foreground">PSI alerts / {featureDrift.length} monitored</small></CardContent></Card>
        <Card><CardContent className="p-4"><p className="eyebrow">Deep challengers</p><strong className="metric-value mt-2 block">{deep?.models.rankings.length ?? 0}</strong><small className="text-muted-foreground">{deep ? `${deep.meta.epochs} epochs · ${deep.meta.folds} fold` : "optional artifact unavailable"}</small></CardContent></Card>
      </div>

      {(status !== "healthy" || performanceAlarms.length > 0) && (
        <div className="flex gap-3 border border-amber-500/30 bg-amber-500/5 p-4 text-sm text-amber-100/85">
          <AlertTriangle className="size-5 shrink-0 text-amber-400" />
          <div><b className="text-amber-200">Research caution.</b> {status !== "healthy" ? "One or more artifacts are stale or unavailable. " : ""}{performanceAlarms.length ? `Outcome drift alarms are active for ${performanceAlarms.map(([lane]) => lane.replaceAll("_", " ")).join(", ")}. Promotion remains gated.` : ""}</div>
        </div>
      )}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><DatabaseZap className="size-4 text-primary" />Data source health</CardTitle></CardHeader>
          <CardContent className="overflow-x-auto p-0">
            <Table>
              <TableHeader><TableRow><TableHead>Source</TableHead><TableHead>Status</TableHead><TableHead>Latest / rows</TableHead><TableHead>Role</TableHead></TableRow></TableHeader>
              <TableBody>
                <TableRow><TableCell className="font-medium">BTC daily</TableCell><TableCell><Badge variant={market?.stale ? "destructive" : "default"}>{market?.status ?? "unknown"}</Badge></TableCell><TableCell className="font-mono">{market?.actual_closed_utc ?? "—"}</TableCell><TableCell className="text-xs text-muted-foreground">required</TableCell></TableRow>
                {intraday.map((row) => <TableRow key={row.timeframe}><TableCell className="font-medium">BTC {row.timeframe}</TableCell><TableCell><Badge variant={row.stale ? "destructive" : "secondary"}>{row.status}</Badge></TableCell><TableCell className="font-mono">{row.rows.toLocaleString()}</TableCell><TableCell className="text-xs text-muted-foreground">deep + realized features</TableCell></TableRow>)}
                {external.map((row) => <TableRow key={row.source}><TableCell className="font-medium capitalize">{row.source}</TableCell><TableCell><Badge variant={row.available ? "default" : "outline"}>{row.status}</Badge></TableCell><TableCell className="font-mono">{row.usable_rows}</TableCell><TableCell className="text-xs text-muted-foreground">optional, never zero-filled</TableCell></TableRow>)}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><ShieldCheck className="size-4 text-primary" />Point-in-time contract</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-xs text-muted-foreground">
            <p><b className="text-foreground">Forecast cutoff:</b> {String(data.meta.validation.daily_evaluation_utc ?? "03:20")} UTC evaluation after the closed candle.</p>
            <p><b className="text-foreground">Validation:</b> {String(data.meta.validation.outer ?? "purged walk-forward")} with separate calibration and policy windows.</p>
            <p><b className="text-foreground">Stacking:</b> {String(data.meta.validation.stacking ?? "pre-test OOS selection")}</p>
            <p><b className="text-foreground">Promotion:</b> minimum {String(data.meta.validation.minimum_live_grades_for_promotion ?? 20)} immutable live grades; production changes monthly.</p>
            <p><b className="text-foreground">Availability:</b> {data.meta.availability_assumption}</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><AlertTriangle className="size-4 text-primary" />Feature drift PSI</CardTitle></CardHeader>
          <CardContent className="h-96">
            {featureDrift.length ? <ResponsiveContainer width="100%" height="100%"><BarChart data={featureDrift.slice(0, 16)} layout="vertical" margin={{ left: 92, right: 12 }}><CartesianGrid stroke="rgba(255,255,255,.06)" horizontal={false} /><XAxis type="number" tick={{ fill: "#888", fontSize: 9 }} /><YAxis dataKey="feature" type="category" width={150} tick={{ fill: "#aaa", fontSize: 9 }} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => Number(value).toFixed(3)} /><Bar dataKey="psi" fill="#f7931a" radius={0} /></BarChart></ResponsiveContainer> : <p className="text-sm text-muted-foreground">Feature drift is unavailable.</p>}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2 text-sm"><Cpu className="size-4 text-primary" />Deep challenger ranking</CardTitle></CardHeader>
          <CardContent className="overflow-x-auto p-0">
            {deep?.models.rankings.length ? <Table><TableHeader><TableRow><TableHead>Rank</TableHead><TableHead>Model</TableHead><TableHead>OOS</TableHead><TableHead>Dir / LCB</TableHead><TableHead>ECE</TableHead><TableHead>Expectancy</TableHead></TableRow></TableHeader><TableBody>{deep.models.rankings.map((row) => <TableRow key={row.model}><TableCell className="font-mono">#{row.rank}</TableCell><TableCell><span className="font-medium">{row.model}</span><small className="block text-muted-foreground">challenger</small></TableCell><TableCell className="font-mono">{row.observations}</TableCell><TableCell className="font-mono">{formatPercent(row.directional_accuracy)} / {formatPercent(row.directional_lcb)}</TableCell><TableCell className="font-mono">{formatPercent(row.ece)}</TableCell><TableCell className={row.expectancy >= 0 ? "font-mono text-emerald-400" : "font-mono text-red-400"}>{formatSignedPercent(row.expectancy, 3)}</TableCell></TableRow>)}</TableBody></Table> : <div className="p-6 text-sm text-muted-foreground">Deep artifact is optional. The daily production pipeline remains operational without it.</div>}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
