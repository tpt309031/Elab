"use client";

import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";

import type { ForecastRow } from "@/lib/types";

const probabilityConfig = [
  { key: "prob_up", label: "UP", color: "#34d399" },
  { key: "prob_sideway", label: "SIDEWAY", color: "#fbbf24" },
  { key: "prob_down", label: "DOWN", color: "#ef4444" },
] as const;

interface ProbabilityGaugeProps {
  forecast?: ForecastRow;
}

export function ProbabilityGauge({ forecast }: ProbabilityGaugeProps) {
  const data = probabilityConfig.map((item) => ({
    ...item,
    value: forecast?.[item.key] ?? 1 / 3,
  }));
  const call = forecast?.forecast ?? "no-call";
  return (
    <div className="relative h-52 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie data={data} dataKey="value" innerRadius="68%" outerRadius="90%" startAngle={210} endAngle={-30} stroke="none" paddingAngle={2}>
            {data.map((entry) => <Cell key={entry.key} fill={entry.color} fillOpacity={0.9} />)}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute inset-0 grid place-content-center pt-6 text-center">
        <span className="eyebrow">next call</span>
        <strong className="mt-1 text-2xl font-semibold uppercase tracking-tight">{call}</strong>
        <span className="mt-1 font-mono text-xs text-muted-foreground">expected score {Math.round((forecast?.expected_score ?? 0) * 100)}%</span>
      </div>
    </div>
  );
}
