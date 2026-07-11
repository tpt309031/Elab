"use client";

import { useEffect, useMemo, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  createChart,
  createSeriesMarkers,
  LineSeries,
  type BusinessDay,
  type CandlestickData,
  type LineData,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";

import type { ForecastRow, IndexRow, MarketRow } from "@/lib/types";

interface MarketChartProps {
  market: MarketRow[];
  indices: IndexRow[];
  forecasts: ForecastRow[];
}

function timeLabel(time: Time | undefined): string {
  if (!time) return "";
  if (typeof time === "string") return time;
  if (typeof time === "number") return new Date(time * 1000).toISOString().slice(0, 10);
  const day = time as BusinessDay;
  return `${day.year}-${String(day.month).padStart(2, "0")}-${String(day.day).padStart(2, "0")}`;
}

export function MarketChart({ market, indices, forecasts }: MarketChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const indexByDate = useMemo(() => new Map(indices.map((row) => [row.date, row])), [indices]);

  useEffect(() => {
    const container = containerRef.current;
    const tooltip = tooltipRef.current;
    if (!container || !tooltip || market.length === 0) return;
    const chart = createChart(container, {
      autoSize: true,
      height: 620,
      layout: {
        background: { type: ColorType.Solid, color: "#101010" },
        textColor: "#9a9a9a",
        panes: { separatorColor: "#303030", separatorHoverColor: "#f7931a" },
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.045)" },
        horzLines: { color: "rgba(255,255,255,0.045)" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "rgba(255,255,255,0.66)", labelBackgroundColor: "#f7931a" },
        horzLine: { color: "rgba(255,255,255,0.24)", labelBackgroundColor: "#262626" },
      },
      rightPriceScale: { borderColor: "#292929", scaleMargins: { top: 0.08, bottom: 0.08 } },
      timeScale: { borderColor: "#292929", timeVisible: false, rightOffset: 2, barSpacing: 8, minBarSpacing: 3 },
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    });
    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#34d399",
      downColor: "#ef4444",
      wickUpColor: "#34d399",
      wickDownColor: "#ef4444",
      borderVisible: false,
      priceFormat: { type: "price", precision: 0, minMove: 1 },
    }, 0);
    const btcIndex = chart.addSeries(LineSeries, {
      color: "#f7931a",
      lineWidth: 3,
      title: "BTC Psychology",
      priceScaleId: "right",
      priceFormat: { type: "price", precision: 0, minMove: 1 },
    }, 1);
    const traderIndex = chart.addSeries(LineSeries, {
      color: "#4f9dff",
      lineWidth: 2,
      title: "Trader Energy",
      priceScaleId: "right",
      priceFormat: { type: "price", precision: 0, minMove: 1 },
    }, 1);
    const candleData: CandlestickData[] = market.map((row) => ({
      time: row.timestamp as Time,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
    }));
    const btcIndexData: LineData[] = [];
    const traderIndexData: LineData[] = [];
    for (const row of indices) {
      if (row.index_BTC != null) btcIndexData.push({ time: row.date as Time, value: row.index_BTC });
      if (row.index_me != null) traderIndexData.push({ time: row.date as Time, value: row.index_me });
    }
    candles.setData(candleData);
    btcIndex.setData(btcIndexData);
    traderIndex.setData(traderIndexData);
    const marketDates = new Set(market.map((row) => row.timestamp.slice(0, 10)));
    const markerData: SeriesMarker<Time>[] = forecasts
      .filter((row) => row.forecast !== "no-call" && marketDates.has(row.date))
      .map((row) => ({
        time: row.date as Time,
        position: row.forecast === "up" ? "belowBar" : row.forecast === "down" ? "aboveBar" : "inBar",
        shape: row.forecast === "up" ? "arrowUp" : row.forecast === "down" ? "arrowDown" : "circle",
        color: row.forecast === "up" ? "#34d399" : row.forecast === "down" ? "#ef4444" : "#fbbf24",
        text: `${row.forecast.toUpperCase()} ${Math.round((row.expected_score ?? 0) * 100)}%`,
      }));
    createSeriesMarkers(candles, markerData);
    const panes = chart.panes();
    panes[0]?.setHeight(410);
    panes[1]?.setHeight(210);
    chart.timeScale().fitContent();

    chart.subscribeCrosshairMove((parameter) => {
      const date = timeLabel(parameter.time);
      const candle = parameter.seriesData.get(candles) as CandlestickData | undefined;
      const indexRow = indexByDate.get(date);
      if (!date || !candle || !parameter.point) {
        tooltip.style.opacity = "0";
        return;
      }
      tooltip.style.opacity = "1";
      tooltip.style.left = `${Math.min(parameter.point.x + 18, Math.max(12, container.clientWidth - 240))}px`;
      tooltip.style.top = `${Math.max(12, parameter.point.y - 42)}px`;
      const move = (candle.close / candle.open - 1) * 100;
      tooltip.textContent = `${date}  O ${candle.open.toFixed(0)}  H ${candle.high.toFixed(0)}  L ${candle.low.toFixed(0)}  C ${candle.close.toFixed(0)}  ${move >= 0 ? "+" : ""}${move.toFixed(2)}%  BTC-I ${indexRow?.index_BTC ?? "—"}  ME-I ${indexRow?.index_me ?? "—"}`;
    });
    const observer = new ResizeObserver(() => chart.applyOptions({ width: container.clientWidth }));
    observer.observe(container);
    return () => {
      observer.disconnect();
      chart.remove();
    };
  }, [forecasts, indexByDate, indices, market]);

  return (
    <div className="relative min-h-[620px] w-full overflow-hidden border border-border bg-card">
      <div ref={containerRef} className="h-[620px] w-full" aria-label="BTC candlestick chart with private indices" />
      <div
        ref={tooltipRef}
        className="pointer-events-none absolute z-20 max-w-56 border border-[#464646] bg-[#111]/95 px-3 py-2 font-mono text-[10px] leading-5 text-[#f5f5f5] opacity-0 shadow-2xl transition-opacity"
      />
      <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap gap-2 text-[10px]">
        <span className="border border-border bg-black/70 px-2 py-1 text-muted-foreground"><i className="mr-1 inline-block size-2 bg-primary" />BTC Psychology</span>
        <span className="border border-border bg-black/70 px-2 py-1 text-muted-foreground"><i className="mr-1 inline-block size-2 bg-[#4f9dff]" />Trader Energy</span>
      </div>
    </div>
  );
}
