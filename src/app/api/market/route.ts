import { NextRequest, NextResponse } from "next/server";

type IntervalKey = "5m" | "1h" | "4h" | "1d";

const intervals: Record<IntervalKey, { binance: string; okx: string; milliseconds: number; limit: number }> = {
  "5m": { binance: "5m", okx: "5m", milliseconds: 5 * 60_000, limit: 300 },
  "1h": { binance: "1h", okx: "1H", milliseconds: 60 * 60_000, limit: 500 },
  "4h": { binance: "4h", okx: "4H", milliseconds: 4 * 60 * 60_000, limit: 500 },
  "1d": { binance: "1d", okx: "1Dutc", milliseconds: 24 * 60 * 60_000, limit: 1000 },
};

interface NormalizedMarketRow {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}
function normalizeRows(rows: Array<{ timestamp: number; open: unknown; high: unknown; low: unknown; close: unknown; volume: unknown }>, timeframe: IntervalKey): NormalizedMarketRow[] {
  return rows
    .map((row) => ({
      timestamp: timeframe === "1d" ? new Date(row.timestamp).toISOString().slice(0, 10) : new Date(row.timestamp).toISOString(),
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
      volume: Number(row.volume),
    }))
    .filter((row) => [row.open, row.high, row.low, row.close, row.volume].every(Number.isFinite))
    .sort((left, right) => left.timestamp.localeCompare(right.timestamp));
}

async function fromBinance(timeframe: IntervalKey): Promise<NormalizedMarketRow[]> {
  const config = intervals[timeframe];
  const url = new URL("https://api.binance.com/api/v3/klines");
  url.searchParams.set("symbol", "BTCUSDT");
  url.searchParams.set("interval", config.binance);
  url.searchParams.set("limit", String(config.limit));
  const response = await fetch(url, {
    headers: { "user-agent": "elab-market-monitor/2.0" },
    next: { revalidate: 300 },
  });
  if (!response.ok) throw new Error(`Binance ${response.status}`);
  const payload = (await response.json()) as unknown[][];
  return normalizeRows(payload.map((row) => ({
    timestamp: Number(row[0]),
    open: row[1],
    high: row[2],
    low: row[3],
    close: row[4],
    volume: row[5],
  })), timeframe);
}

async function fromOkx(timeframe: IntervalKey): Promise<NormalizedMarketRow[]> {
  const config = intervals[timeframe];
  const url = new URL("https://www.okx.com/api/v5/market/history-candles");
  url.searchParams.set("instId", "BTC-USDT");
  url.searchParams.set("bar", config.okx);
  url.searchParams.set("limit", String(Math.min(config.limit, 300)));
  const response = await fetch(url, {
    headers: { "user-agent": "elab-market-monitor/2.0" },
    next: { revalidate: 300 },
  });
  if (!response.ok) throw new Error(`OKX ${response.status}`);
  const payload = (await response.json()) as { data?: string[][] };
  return normalizeRows((payload.data ?? []).map((row) => ({
    timestamp: Number(row[0]),
    open: row[1],
    high: row[2],
    low: row[3],
    close: row[4],
    volume: row[5],
  })), timeframe);
}

export async function GET(request: NextRequest) {
  const requested = request.nextUrl.searchParams.get("timeframe") ?? "5m";
  if (!(requested in intervals)) {
    return NextResponse.json({ error: "Unsupported timeframe" }, { status: 400 });
  }
  const timeframe = requested as IntervalKey;
  try {
    const rows = await fromBinance(timeframe);
    return NextResponse.json(
      { provider: "Binance", timeframe, rows, fetchedAt: new Date().toISOString() },
      { headers: { "Cache-Control": "s-maxage=300, stale-while-revalidate=60" } },
    );
  } catch (binanceError) {
    try {
      const rows = await fromOkx(timeframe);
      return NextResponse.json(
        { provider: "OKX", timeframe, rows, fetchedAt: new Date().toISOString() },
        { headers: { "Cache-Control": "s-maxage=300, stale-while-revalidate=60" } },
      );
    } catch (okxError) {
      return NextResponse.json(
        {
          error: "Market data unavailable",
          detail: `${binanceError instanceof Error ? binanceError.message : "Binance failed"}; ${okxError instanceof Error ? okxError.message : "OKX failed"}`,
        },
        { status: 502 },
      );
    }
  }
}
