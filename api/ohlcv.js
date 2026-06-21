const INTERVALS = {
  "5m": { binance: "5m", okx: "5m", ms: 5 * 60 * 1000 },
  "1h": { binance: "1h", okx: "1H", ms: 60 * 60 * 1000 },
  "4h": { binance: "4h", okx: "4H", ms: 4 * 60 * 60 * 1000 },
  "1d": { binance: "1d", okx: "1Dutc", ms: 24 * 60 * 60 * 1000 },
};

function normalizeRows(rows, timeframe) {
  const intervalMs = INTERVALS[timeframe].ms;
  const now = Date.now();
  return rows
    .map((row) => ({
      timestamp: new Date(Number(row.timestamp)).toISOString().slice(0, 10),
      openTime: Number(row.timestamp),
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
      volume: Number(row.volume),
    }))
    .filter((row) => Number.isFinite(row.close) && row.openTime + intervalMs <= now)
    .map(({ openTime, ...row }) => row)
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));
}

async function fetchBinance(timeframe, startMs) {
  const interval = INTERVALS[timeframe].binance;
  const url = new URL("https://api.binance.com/api/v3/klines");
  url.searchParams.set("symbol", "BTCUSDT");
  url.searchParams.set("interval", interval);
  url.searchParams.set("startTime", String(startMs));
  url.searchParams.set("limit", "1000");
  const response = await fetch(url, { headers: { "user-agent": "btc-energy-dashboard" } });
  if (!response.ok) throw new Error(`Binance ${response.status}`);
  const data = await response.json();
  return normalizeRows(
    data.map((item) => ({
      timestamp: item[0],
      open: item[1],
      high: item[2],
      low: item[3],
      close: item[4],
      volume: item[5],
    })),
    timeframe,
  );
}

async function fetchOkx(timeframe, startMs) {
  const bar = INTERVALS[timeframe].okx;
  const url = new URL("https://www.okx.com/api/v5/market/history-candles");
  url.searchParams.set("instId", "BTC-USDT");
  url.searchParams.set("bar", bar);
  url.searchParams.set("after", String(Date.now()));
  url.searchParams.set("limit", "300");
  const response = await fetch(url, { headers: { "user-agent": "btc-energy-dashboard" } });
  if (!response.ok) throw new Error(`OKX ${response.status}`);
  const payload = await response.json();
  return normalizeRows(
    (payload.data || [])
      .filter((item) => Number(item[0]) >= startMs)
      .map((item) => ({
        timestamp: item[0],
        open: item[1],
        high: item[2],
        low: item[3],
        close: item[4],
        volume: item[5],
      })),
    timeframe,
  );
}

function getQueryParam(request, key, fallback) {
  try {
    return new URL(request.url || "", "https://btc-energy.local").searchParams.get(key) || fallback;
  } catch {
    return fallback;
  }
}

module.exports = async function handler(request, response) {
  const timeframe = String(getQueryParam(request, "timeframe", "1d"));
  if (!INTERVALS[timeframe]) {
    response.status(400).json({ error: "Unsupported timeframe" });
    return;
  }
  const startText = String(getQueryParam(request, "start", "2024-01-01"));
  const startMs = Number.isFinite(Date.parse(startText)) ? Date.parse(startText) : Date.parse("2024-01-01");
  response.setHeader("Cache-Control", "s-maxage=300, stale-while-revalidate=3600");
  try {
    const rows = await fetchBinance(timeframe, startMs);
    response.status(200).json({ provider: "Binance", timeframe, rows });
  } catch (binanceError) {
    try {
      const rows = await fetchOkx(timeframe, startMs);
      response.status(200).json({ provider: "OKX", timeframe, rows });
    } catch (okxError) {
      response.status(502).json({
        error: "OHLCV fetch failed",
        detail: `${binanceError.message}; ${okxError.message}`,
      });
    }
  }
};
