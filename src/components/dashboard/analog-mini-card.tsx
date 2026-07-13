"use client";

import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { formatPercent, formatSignedPercent } from "@/lib/format";
import type { MarketRow, SimilarCase } from "@/lib/types";

interface AnalogMiniCardProps {
  item: SimilarCase;
  market: MarketRow[];
}

const tooltipStyle = { background: "#111", border: "1px solid #353535", color: "#f5f5f5" };

export function AnalogMiniCard({ item, market }: AnalogMiniCardProps) {
  const center = market.findIndex((row) => row.timestamp === item.date);
  const source = center >= 0 ? market.slice(Math.max(0, center - 3), center + 5) : [];
  const base = source[0]?.close ?? 1;
  const chart = source.map((row) => ({
    date: row.timestamp.slice(5),
    normalized: row.close / base - 1,
    event: row.timestamp === item.date ? 1 : 0,
  }));
  return (
    <article className="border border-border bg-background/40 p-3">
      <div className="flex items-center justify-between gap-2 text-xs"><span className="font-mono text-muted-foreground">{item.date}</span><span className="font-mono text-muted-foreground">sim {formatPercent(item.similarity)}</span></div>
      <div className="mt-2 flex items-baseline justify-between"><strong className="text-sm uppercase">{item.outcome}</strong><span className={item.move >= 0 ? "font-mono text-emerald-400" : "font-mono text-red-400"}>{formatSignedPercent(item.move)}</span></div>
      <div className="mt-2 h-24">
        {chart.length ? <ResponsiveContainer width="100%" height="100%"><LineChart data={chart}><XAxis dataKey="date" hide /><YAxis hide domain={["dataMin", "dataMax"]} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => formatSignedPercent(Number(value))} /><ReferenceLine x={item.date.slice(5)} stroke="#f7931a" strokeDasharray="3 3" /><Line dataKey="normalized" stroke={item.move >= 0 ? "#34d399" : "#ef4444"} strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer> : <div className="grid h-full place-items-center text-xs text-muted-foreground">Price window unavailable</div>}
      </div>
    </article>
  );
}
