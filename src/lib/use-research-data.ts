"use client";

import useSWR from "swr";

import type { LiveMarketResponse, ResearchArtifact } from "@/lib/types";

async function jsonFetcher<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}
export function useResearchData() {
  const research = useSWR<ResearchArtifact>("/data/hybrid_research.json", jsonFetcher, {
    revalidateOnFocus: false,
    dedupingInterval: 60_000,
  });
  const live = useSWR<LiveMarketResponse>("/api/market?timeframe=5m", jsonFetcher, {
    refreshInterval: 300_000,
    revalidateOnFocus: true,
    dedupingInterval: 60_000,
  });
  return { research, live };
}
