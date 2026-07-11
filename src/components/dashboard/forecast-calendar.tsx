"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ForecastRow } from "@/lib/types";

const weekdays = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];

interface ForecastCalendarProps {
  month: string;
  rows: ForecastRow[];
  selectedDate?: string;
  onMonthChange: (month: string) => void;
  onSelect: (row: ForecastRow) => void;
}
function shiftMonth(month: string, offset: number): string {
  const [year, monthIndex] = month.split("-").map(Number);
  const shifted = new Date(Date.UTC(year, monthIndex - 1 + offset, 1));
  return `${shifted.getUTCFullYear()}-${String(shifted.getUTCMonth() + 1).padStart(2, "0")}`;
}

function cellBackground(row?: ForecastRow): string {
  if (!row || row.forecast === "no-call") return "bg-[#242424]";
  if (row.forecast === "up") {
    if (row.status === "correct") return "bg-emerald-500/75";
    if (row.status === "wrong") return "bg-[linear-gradient(to_bottom,rgba(52,211,153,.32)_0_50%,rgba(239,68,68,.72)_50%_100%)]";
    return "bg-emerald-400/25";
  }
  if (row.forecast === "down") {
    if (row.status === "correct") return "bg-red-600/80";
    if (row.status === "wrong") return "bg-[linear-gradient(to_bottom,rgba(239,68,68,.28)_0_50%,rgba(153,27,27,.85)_50%_100%)]";
    return "bg-red-400/25";
  }
  if (row.status === "correct") return "bg-amber-500/80";
  if (row.status === "wrong") return "bg-[linear-gradient(to_bottom,rgba(251,191,36,.28)_0_50%,rgba(239,68,68,.75)_50%_100%)]";
  return "bg-amber-300/20";
}

export function ForecastCalendar({ month, rows, selectedDate, onMonthChange, onSelect }: ForecastCalendarProps) {
  const [year, monthNumber] = month.split("-").map(Number);
  const firstDay = new Date(Date.UTC(year, monthNumber - 1, 1)).getUTCDay();
  const daysInMonth = new Date(Date.UTC(year, monthNumber, 0)).getUTCDate();
  const byDate = new Map(rows.map((row) => [row.date, row]));
  const today = new Date().toISOString().slice(0, 10);
  const monthLabel = new Intl.DateTimeFormat("en-US", { month: "long", year: "numeric", timeZone: "UTC" }).format(new Date(Date.UTC(year, monthNumber - 1, 1)));
  const cells = Array.from({ length: firstDay + daysInMonth }, (_, index) => {
    if (index < firstDay) return <div key={`empty-${index}`} className="min-h-20 border border-transparent sm:min-h-28" aria-hidden="true" />;
    const day = index - firstDay + 1;
    const date = `${month}-${String(day).padStart(2, "0")}`;
    const row = byDate.get(date);
    const move = row?.daily_return;
    return (
      <button
        type="button"
        key={date}
        onClick={() => row && onSelect(row)}
        disabled={!row}
        className={cn(
          "group flex min-h-20 min-w-0 flex-col justify-between border p-1.5 text-left shadow-[0_8px_20px_rgba(0,0,0,.18)] transition hover:-translate-y-0.5 hover:border-white/40 sm:min-h-28 sm:p-2.5",
          row ? cellBackground(row) : "border-border/50 bg-[#151515] text-muted-foreground",
          date === today && "border-primary shadow-[inset_0_0_0_1px_#f7931a]",
          date === selectedDate ? "border-white" : "border-white/10",
        )}
        aria-label={`${date}${row ? ` ${row.forecast} ${row.status}` : " no forecast"}`}
      >
        <span className="font-mono text-[10px] text-white/80 sm:text-xs">{day}</span>
        <strong className={cn("truncate font-mono text-[11px] tracking-tight sm:text-lg", move == null ? "text-white/55" : move >= 0 ? "text-emerald-50" : "text-red-50")}>
          {move == null ? "—" : `${move > 0 ? "+" : ""}${(move * 100).toFixed(1)}%`}
        </strong>
        <span className="flex min-w-0 flex-col font-mono text-[7px] uppercase leading-tight text-white/80 sm:text-[9px]">
          <b className="truncate font-medium">{row?.forecast ?? "NO CALL"}</b>
          <span className="truncate">{row?.status ?? "—"}</span>
        </span>
      </button>
    );
  });
  return (
    <section className="border border-border bg-[#0f0f0f] p-2.5 sm:p-4">
      <header className="mb-4 flex items-center justify-between gap-3">
        <Button variant="outline" size="icon" onClick={() => onMonthChange(shiftMonth(month, -1))} aria-label="Previous month"><ChevronLeft /></Button>
        <div className="text-center">
          <p className="eyebrow">forecast calendar</p>
          <h3 className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">{monthLabel}</h3>
        </div>
        <Button variant="outline" size="icon" onClick={() => onMonthChange(shiftMonth(month, 1))} aria-label="Next month"><ChevronRight /></Button>
      </header>
      <div className="grid grid-cols-7 gap-1 sm:gap-2">
        {weekdays.map((weekday) => <div key={weekday} className="py-1 text-center font-mono text-[7px] tracking-wider text-muted-foreground sm:text-[10px]">{weekday}</div>)}
        {cells}
      </div>
    </section>
  );
}
