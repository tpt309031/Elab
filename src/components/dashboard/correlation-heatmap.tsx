"use client";

import { useMemo, useState } from "react";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { humanizeFeature } from "@/lib/format";

interface CorrelationHeatmapProps {
  rows: Array<Record<string, string | number | null>>;
}
function correlationColor(value: number): string {
  const magnitude = Math.min(1, Math.abs(value));
  if (value >= 0) return `rgba(52, 211, 153, ${0.08 + magnitude * 0.72})`;
  return `rgba(239, 68, 68, ${0.08 + magnitude * 0.72})`;
}

export function CorrelationHeatmap({ rows }: CorrelationHeatmapProps) {
  const available = useMemo(() => rows.map((row) => String(row.feature)), [rows]);
  const [focus, setFocus] = useState("daily_return");
  const focusRow = rows.find((row) => row.feature === focus) ?? rows[0];
  const values = Object.entries(focusRow ?? {})
    .filter(([key, value]) => key !== "feature" && typeof value === "number")
    .sort((left, right) => Math.abs(Number(right[1])) - Math.abs(Number(left[1])));
  return (
    <section className="border border-border bg-card p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div><p className="eyebrow">spearman correlation</p><h3 className="mt-1 text-lg font-semibold">Feature relationship heatmap</h3></div>
        <Select value={focus} onValueChange={setFocus}>
          <SelectTrigger className="w-48"><SelectValue /></SelectTrigger>
          <SelectContent>{available.map((feature) => <SelectItem key={feature} value={feature}>{humanizeFeature(feature)}</SelectItem>)}</SelectContent>
        </Select>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-4">
        {values.map(([feature, value]) => {
          const numeric = Number(value);
          return (
            <div key={feature} className="min-w-0 border border-white/10 p-3" style={{ backgroundColor: correlationColor(numeric) }}>
              <p className="truncate text-[10px] uppercase tracking-wide text-white/70">{humanizeFeature(feature)}</p>
              <strong className="mt-2 block font-mono text-lg text-white">{numeric.toFixed(2)}</strong>
            </div>
          );
        })}
      </div>
      <p className="mt-3 text-[11px] text-muted-foreground">Descriptive correlation only. Feature admission still depends on purged walk-forward performance.</p>
    </section>
  );
}
