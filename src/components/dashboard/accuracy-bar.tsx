import type { ForecastRow } from "@/lib/types";

interface AccuracyBarProps {
  rows: ForecastRow[];
}
export function AccuracyBar({ rows }: AccuracyBarProps) {
  const evaluated = rows.filter((row) => ["correct", "partial", "wrong"].includes(row.status));
  const count = (status: ForecastRow["status"]) => evaluated.filter((row) => row.status === status).length;
  const correct = count("correct");
  const partial = count("partial");
  const wrong = count("wrong");
  const total = Math.max(1, evaluated.length);
  return (
    <div className="space-y-2 border border-border bg-card p-3">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
        <span className="font-medium">Evaluated calls</span>
        <span className="font-mono text-muted-foreground">{evaluated.length} sessions</span>
      </div>
      <div className="flex h-2 overflow-hidden bg-muted" aria-label="Correct, partial, and wrong forecast ratio">
        <div className="bg-emerald-500" style={{ width: `${(correct / total) * 100}%` }} />
        <div className="bg-amber-400" style={{ width: `${(partial / total) * 100}%` }} />
        <div className="bg-red-500" style={{ width: `${(wrong / total) * 100}%` }} />
      </div>
      <div className="grid grid-cols-3 gap-2 font-mono text-[10px] text-muted-foreground">
        <span><i className="mr-1 inline-block size-1.5 bg-emerald-500" />Correct {Math.round((correct / total) * 100)}%</span>
        <span className="text-center"><i className="mr-1 inline-block size-1.5 bg-amber-400" />Partial {Math.round((partial / total) * 100)}%</span>
        <span className="text-right"><i className="mr-1 inline-block size-1.5 bg-red-500" />Wrong {Math.round((wrong / total) * 100)}%</span>
      </div>
    </div>
  );
}
