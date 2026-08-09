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

interface DayCardProps {
  date: string;
  day: number;
  row?: ForecastRow;
  today: string;
  selectedDate?: string;
  mobile?: boolean;
  onSelect: (row: ForecastRow) => void;
}

function DayCard({ date, day, row, today, selectedDate, mobile = false, onSelect }: DayCardProps) {
  const move = row?.daily_return;
  const weekday = weekdays[new Date(`${date}T00:00:00Z`).getUTCDay()];
  return (
    <button
      type="button"
      onClick={() => row && onSelect(row)}
      disabled={!row}
      className={cn(
        "group min-w-0 rounded-xl border text-left shadow-[0_8px_20px_rgba(0,0,0,.18)] transition hover:-translate-y-0.5 hover:border-white/45 hover:shadow-[0_12px_28px_rgba(0,0,0,.35)]",
        mobile ? "grid min-h-20 grid-cols-[52px_minmax(0,1fr)_auto] items-center gap-3 p-3" : "flex min-h-28 flex-col justify-between p-2.5",
        row ? cellBackground(row) : "border-border/50 bg-[#151515] text-muted-foreground",
        date === today && "border-primary shadow-[inset_0_0_0_1px_#f7931a]",
        date === selectedDate ? "border-white" : "border-white/10",
      )}
      aria-label={`${date}${row ? ` ${row.forecast} ${row.status}` : " no forecast"}`}
    >
      {mobile ? (
        <>
          <span className="border-r border-white/15 pr-3 text-center">
            <b className="block font-mono text-xl text-white">{day}</b>
            <small className="font-mono text-[9px] text-white/55">{weekday}</small>
          </span>
          <span className="min-w-0">
            <strong className={cn("block font-mono text-lg", move == null ? "text-white/55" : move >= 0 ? "text-emerald-50" : "text-red-50")}>
              {move == null ? "Move pending" : `${move > 0 ? "+" : ""}${(move * 100).toFixed(1)}%`}
            </strong>
            <small className="block truncate text-[10px] text-white/65">{row?.trade_gate_reason ?? "No published forecast"}</small>
          </span>
          <span className="text-right font-mono text-[9px] uppercase text-white/85">
            <b className="block text-[11px]">{row?.forecast ?? "NO DATA"}</b>
            <span className="block">{row?.status ?? "UNAVAILABLE"}</span>
            {row && <span className="mt-1 block text-white/55">{row.trade_eligible ? `TRADE ${row.trade_action}` : "FLAT"}</span>}
          </span>
        </>
      ) : (
        <>
          <span className="font-mono text-xs text-white/80">{day}</span>
          <strong className={cn("truncate font-mono text-lg tracking-tight", move == null ? "text-white/55" : move >= 0 ? "text-emerald-50" : "text-red-50")}>
            {move == null ? "—" : `${move > 0 ? "+" : ""}${(move * 100).toFixed(1)}%`}
          </strong>
          <span className="flex min-w-0 flex-col font-mono text-[9px] uppercase leading-tight text-white/80">
            <b className="truncate font-medium">{row?.forecast ?? "NO FORECAST"}</b>
            <span className="truncate">{row?.status ?? "NOT AVAILABLE"}</span>
          </span>
        </>
      )}
    </button>
  );
}

export function ForecastCalendar({ month, rows, selectedDate, onMonthChange, onSelect }: ForecastCalendarProps) {
  const [year, monthNumber] = month.split("-").map(Number);
  const firstDay = new Date(Date.UTC(year, monthNumber - 1, 1)).getUTCDay();
  const daysInMonth = new Date(Date.UTC(year, monthNumber, 0)).getUTCDate();
  const byDate = new Map(rows.map((row) => [row.date, row]));
  const today = new Date().toISOString().slice(0, 10);
  const monthLabel = new Intl.DateTimeFormat("en-US", { month: "long", year: "numeric", timeZone: "UTC" }).format(new Date(Date.UTC(year, monthNumber - 1, 1)));
  const days = Array.from({ length: daysInMonth }, (_, index) => {
    const day = index + 1;
    const date = `${month}-${String(day).padStart(2, "0")}`;
    return { day, date, row: byDate.get(date) };
  });
  return (
    <section className="border border-border bg-[#0f0f0f] p-2.5 sm:p-4">
      <header className="mb-4 grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3">
        <Button variant="outline" size="icon" onClick={() => onMonthChange(shiftMonth(month, -1))} aria-label="Previous month"><ChevronLeft /></Button>
        <div className="text-center">
          <p className="eyebrow">forecast calendar</p>
          <h3 className="mt-1 text-xl font-semibold tracking-tight sm:text-2xl">{monthLabel}</h3>
        </div>
        <Button variant="outline" size="icon" onClick={() => onMonthChange(shiftMonth(month, 1))} aria-label="Next month"><ChevronRight /></Button>
      </header>
      <div className="mb-3 hidden justify-end gap-3 font-mono text-[9px] text-muted-foreground sm:flex">
        <span><i className="mr-1 inline-block size-2 rounded-sm bg-emerald-400/60" />UP</span>
        <span><i className="mr-1 inline-block size-2 rounded-sm bg-amber-400/60" />SIDEWAY</span>
        <span><i className="mr-1 inline-block size-2 rounded-sm bg-red-400/60" />DOWN</span>
        <span>split card = WRONG</span>
      </div>
      <div className="grid gap-2 sm:hidden">
        {days.map(({ day, date, row }) => <DayCard key={date} date={date} day={day} row={row} today={today} selectedDate={selectedDate} mobile onSelect={onSelect} />)}
      </div>
      <div className="hidden grid-cols-7 gap-2 sm:grid">
        {weekdays.map((weekday) => <div key={weekday} className="py-1 text-center font-mono text-[10px] tracking-wider text-muted-foreground">{weekday}</div>)}
        {Array.from({ length: firstDay }, (_, index) => <div key={`empty-${index}`} className="min-h-28" aria-hidden="true" />)}
        {days.map(({ day, date, row }) => <DayCard key={date} date={date} day={day} row={row} today={today} selectedDate={selectedDate} onSelect={onSelect} />)}
      </div>
    </section>
  );
}
