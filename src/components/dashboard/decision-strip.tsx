import { AlertTriangle, Clock3, ShieldCheck, Target, TrendingUp } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { formatDate, formatPercent, formatSignedPercent } from "@/lib/format";
import type { ForecastRow, ResearchArtifact } from "@/lib/types";
import { cn } from "@/lib/utils";

interface DecisionStripProps {
  forecast?: ForecastRow;
  meta: ResearchArtifact["meta"];
  marketStale?: boolean;
}

const directionTone = {
  up: "text-emerald-300",
  down: "text-red-300",
  sideway: "text-amber-300",
  "no-call": "text-muted-foreground",
} as const;

export function DecisionStrip({ forecast, meta, marketStale = false }: DecisionStripProps) {
  const tradeAction = forecast?.trade_action ?? "flat";
  const tradeEligible = Boolean(forecast?.trade_eligible && tradeAction !== "flat");
  const provenanceWarning = meta.provenance?.status === "research-only";
  const confidence = forecast?.confidence
    ?? (forecast ? Math.max(forecast.prob_down, forecast.prob_sideway, forecast.prob_up) : undefined);

  return (
    <section className="mb-4 border border-border bg-[#0c0c0c]" aria-label="Current forecast and execution decision">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-3 py-2.5 sm:px-4">
        <div className="flex items-center gap-2">
          <Target className="size-4 text-primary" />
          <span className="eyebrow text-foreground">Decision contract</span>
          <Badge variant="outline">UTC DAILY</Badge>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant={marketStale ? "destructive" : "outline"}>{marketStale ? "STALE MARKET" : "MARKET CURRENT"}</Badge>
          <Badge variant={provenanceWarning ? "secondary" : "outline"}>{provenanceWarning ? "RESEARCH-ONLY PROVENANCE" : "PIT VERIFIED"}</Badge>
        </div>
      </header>

      <div className="grid grid-cols-2 divide-x divide-y divide-border sm:grid-cols-3 lg:grid-cols-5 lg:divide-y-0">
        <div className="p-3 sm:p-4">
          <p className="eyebrow">Market outlook</p>
          <strong className={cn("mt-2 block font-mono text-2xl uppercase", directionTone[forecast?.forecast ?? "no-call"])}>
            {forecast?.forecast ?? "pending"}
          </strong>
          <small className="text-muted-foreground">{forecast ? formatDate(forecast.date) : "Awaiting artifact"}</small>
        </div>
        <div className="p-3 sm:p-4">
          <p className="eyebrow">Execution</p>
          <strong className={cn("mt-2 block font-mono text-2xl uppercase", tradeEligible ? directionTone[tradeAction as "up" | "down"] : "text-white/70")}>
            {tradeEligible ? `TRADE ${tradeAction}` : "FLAT"}
          </strong>
          <small className="line-clamp-2 text-muted-foreground">{forecast?.trade_gate_reason ?? "Awaiting schema v5 execution gate"}</small>
        </div>
        <div className="p-3 sm:p-4">
          <p className="eyebrow">Forecast confidence</p>
          <strong className="mt-2 block font-mono text-2xl">{formatPercent(confidence)}</strong>
          <small className="text-muted-foreground">expected grade {formatPercent(forecast?.expected_score)}</small>
        </div>
        <div className="p-3 sm:p-4">
          <p className="eyebrow">After-cost edge</p>
          <strong className={cn("mt-2 block font-mono text-2xl", (forecast?.expectancy_lcb ?? 0) > 0 ? "text-emerald-300" : "text-white/55")}>
            {formatSignedPercent(forecast?.expectancy_lcb)}
          </strong>
          <small className="text-muted-foreground">OOS expectancy lower bound</small>
        </div>
        <div className="col-span-2 p-3 sm:col-span-1 sm:p-4">
          <p className="eyebrow">Information timing</p>
          <strong className="mt-2 flex items-center gap-2 font-mono text-sm"><Clock3 className="size-4 text-primary" />EX-ANTE</strong>
          <small className="text-muted-foreground">cutoff {forecast?.information_cutoff_utc?.slice(0, 10) ?? meta.latest_closed_utc} · target opens {forecast?.target_start_utc?.slice(0, 10) ?? meta.first_publishable_target_utc?.slice(0, 10) ?? "—"}</small>
        </div>
      </div>

      {(provenanceWarning || marketStale) && (
        <div className="flex gap-2 border-t border-amber-500/25 bg-amber-500/5 px-3 py-2.5 text-xs text-amber-100/80 sm:px-4">
          {marketStale ? <AlertTriangle className="size-4 shrink-0 text-red-400" /> : <ShieldCheck className="size-4 shrink-0 text-amber-400" />}
          <span>{marketStale ? "Execution is suspended until the latest closed BTC candle is available." : meta.provenance?.warnings[0] ?? "Private-source point-in-time coverage is incomplete."}</span>
        </div>
      )}
      <div className="flex items-center gap-2 border-t border-border px-3 py-2 text-[10px] text-muted-foreground sm:px-4">
        <TrendingUp className="size-3 text-primary" /> Forecast coverage remains daily. A directional forecast is not an order unless the execution field says TRADE.
      </div>
    </section>
  );
}
